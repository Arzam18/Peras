//! Board representation with incremental state: bitboards plus a mailbox, a stack of
//! per-ply `StateInfo` (hash keys, castling, checkers, pins), make/unmake, legality,
//! check detection, static exchange evaluation and repetition detection.

use crate::bitboard::*;
use crate::nnue::NnueState;
use crate::types::*;
use crate::zobrist::{ZOBRIST, cuckoo, h1, h2, psq_key};

/// Piece values used by SEE, move ordering and material heuristics. The evaluation
/// itself uses tapered PeSTO values; these sit on the same ~100cp-pawn scale.
pub const SEE_VALUE: [Value; 7] = [100, 320, 330, 500, 1000, 0, 0];

#[inline(always)]
pub const fn piece_value(pt: PieceType) -> Value {
    SEE_VALUE[pt as usize]
}

#[inline(always)]
pub const fn piece_value_of(pc: Piece) -> Value {
    if pc.is_none() { 0 } else { SEE_VALUE[(pc.0 % 6) as usize] }
}

#[derive(Clone, Copy, Debug)]
pub struct StateInfo {
    // Copied from the previous state when a move is made.
    pub castling_rights: u8,
    pub rule50: i32,
    pub plies_from_null: i32,
    pub ep_square: Square,
    pub pawn_key: u64,
    pub nonpawn_key: [u64; 2],
    pub minor_key: u64,
    pub material_key: u64,

    // Recomputed after every move.
    pub key: u64,
    pub checkers: Bitboard,
    pub blockers_for_king: [Bitboard; 2],
    pub pinners: [Bitboard; 2],
    pub check_squares: [Bitboard; PIECE_TYPE_NB],
    pub captured_piece: Piece,
    /// 0 = no repetition; +d = the position d plies ago repeats (first repeat);
    /// -d = that earlier position was itself already a repetition (a threefold).
    pub repetition: i32,
    /// Three-check: checks delivered so far by each colour.
    #[cfg(feature = "variants")]
    pub checks_given: [u8; 2],
    /// Antichess: the side to move has a capture available and must take.
    #[cfg(feature = "variants")]
    pub must_capture: bool,
    /// Crazyhouse pockets [color][piece type].
    #[cfg(feature = "variants")]
    pub hand: [[u8; 5]; 2],
    /// Crazyhouse: pieces that arrived by promotion (captured as pawns).
    #[cfg(feature = "variants")]
    pub promoted: Bitboard,
}

/// Zobrist contribution of the variant-only state. Zero without the feature.
#[cfg(not(feature = "variants"))]
#[inline(always)]
fn variant_state_key(_st: &StateInfo) -> u64 {
    0
}
#[cfg(feature = "variants")]
#[inline]
fn variant_state_key(st: &StateInfo) -> u64 {
    let mut key = 0;
    for c in 0..2 {
        key ^= ZOBRIST.checks[c][st.checks_given[c] as usize];
        for pt in 0..5 {
            key ^= ZOBRIST.hand[c][pt][st.hand[c][pt] as usize];
        }
    }
    let mut pr = st.promoted;
    while pr != 0 {
        key ^= ZOBRIST.promoted[pop_lsb(&mut pr) as usize];
    }
    key
}

impl StateInfo {
    const fn empty() -> StateInfo {
        StateInfo {
            castling_rights: 0,
            rule50: 0,
            plies_from_null: 0,
            ep_square: SQ_NONE,
            pawn_key: 0,
            nonpawn_key: [0; 2],
            minor_key: 0,
            material_key: 0,
            key: 0,
            checkers: 0,
            blockers_for_king: [0; 2],
            pinners: [0; 2],
            check_squares: [0; PIECE_TYPE_NB],
            captured_piece: Piece::NONE,
            repetition: 0,
            #[cfg(feature = "variants")]
            checks_given: [0; 2],
            #[cfg(feature = "variants")]
            must_capture: false,
            #[cfg(feature = "variants")]
            hand: [[0; 5]; 2],
            #[cfg(feature = "variants")]
            promoted: 0,
        }
    }
}

#[derive(Clone)]
pub struct Position {
    board: [Piece; 64],
    by_type: [Bitboard; PIECE_TYPE_NB],
    by_color: [Bitboard; 2],
    piece_count: [u8; PIECE_NB],
    side_to_move: Color,
    game_ply: i32,
    chess960: bool,
    variant: Variant,
    castling_rights_mask: [u8; 64],
    castling_rook_square: [Square; 16],
    castling_path: [Bitboard; 16],
    states: Vec<StateInfo>,
    nnue: NnueState,
}

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Default for Position {
    fn default() -> Self {
        Position::from_fen(START_FEN).expect("start position is valid")
    }
}

impl Position {
    fn empty() -> Position {
        let mut states = Vec::with_capacity(1024);
        states.push(StateInfo::empty());
        Position {
            board: [Piece::NONE; 64],
            by_type: [0; PIECE_TYPE_NB],
            by_color: [0; 2],
            piece_count: [0; PIECE_NB],
            side_to_move: Color::White,
            game_ply: 0,
            chess960: false,
            variant: Variant::Standard,
            castling_rights_mask: [0; 64],
            castling_rook_square: [SQ_NONE; 16],
            castling_path: [0; 16],
            states,
            nnue: NnueState::new(),
        }
    }

    pub fn startpos() -> Position {
        Position::default()
    }

    /// Parses a standard-chess FEN (Shredder / X-FEN castling files accepted).
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        Position::from_fen_variant(fen, Variant::Standard)
    }

    /// Parses a FEN for `variant`. Three-check accepts both the lichess form
    /// (`... - 3+3 0 1`, checks remaining) and the appended form (`... 0 1 +0+0`, checks given).
    pub fn from_fen_variant(fen: &str, variant: Variant) -> Result<Position, String> {
        crate::zobrist::init();
        let mut pos = Position::empty();
        pos.variant = variant;
        pos.chess960 = variant == Variant::Chess960;
        #[cfg_attr(not(feature = "variants"), allow(unused_mut))]
        let mut parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("FEN has too few fields: '{}'", fen));
        }

        // Three-check counters live in a token containing '+'; pull it out first.
        #[cfg_attr(not(feature = "variants"), allow(unused_mut))]
        let mut checks_given = [0u8; 2];
        #[cfg(feature = "variants")]
        if let Some(i) = parts.iter().skip(3).position(|t| t.contains('+')).map(|i| i + 3) {
            let tok = parts.remove(i);
            let nums: Vec<u8> = tok.split('+').filter(|x| !x.is_empty()).filter_map(|x| x.parse().ok()).collect();
            if nums.len() == 2 {
                if tok.starts_with('+') {
                    // "+w+b": checks already given.
                    checks_given = [nums[0].min(3), nums[1].min(3)];
                } else {
                    // "w+b": checks remaining.
                    checks_given = [3u8.saturating_sub(nums[0]), 3u8.saturating_sub(nums[1])];
                }
            }
        }

        // 1. Piece placement; crazyhouse adds "[pocket]" and marks promoted pieces with '~'.
        let (board_part, pocket_part) = match parts[0].find('[') {
            Some(i) => (&parts[0][..i], Some(parts[0][i + 1..].trim_end_matches(']'))),
            None => (parts[0], None),
        };
        let mut sq: i32 = squares::A8 as i32;
        let mut last_sq: i32 = -1;
        let mut promoted: Bitboard = 0;
        let mut hand = [[0u8; 5]; 2];
        for ch in board_part.chars() {
            if let Some(d) = ch.to_digit(10) {
                if !(1..=8).contains(&d) {
                    return Err(format!("bad digit '{}' in FEN", ch));
                }
                sq += d as i32;
            } else if ch == '/' {
                sq -= 16;
            } else if ch == '~' {
                if last_sq >= 0 {
                    promoted |= sq_bb(last_sq as Square);
                }
            } else if let Some(pc) = Piece::from_char(ch) {
                if !is_square_ok(sq) {
                    return Err("piece placement overflows the board".to_string());
                }
                pos.put_piece(pc, sq as Square);
                last_sq = sq;
                sq += 1;
            } else {
                return Err(format!("unexpected character '{}' in FEN", ch));
            }
        }
        if let Some(pocket) = pocket_part {
            for ch in pocket.chars() {
                if let Some(pc) = Piece::from_char(ch)
                    && pc.piece_type() != PieceType::King
                {
                    hand[pc.color().idx()][pc.piece_type().idx()] += 1;
                }
            }
        }
        pos.set_pockets(variant, hand, promoted);
        if variant.royal_king() && (pos.piece_count[Piece::W_KING.idx()] != 1 || pos.piece_count[Piece::B_KING.idx()] != 1) {
            return Err("position must have exactly one king per side".to_string());
        }

        // 2. Side to move
        pos.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("bad side to move '{}'", other)),
        };

        // 3. Castling availability
        if let Some(castling) = parts.get(2)
            && variant.castling_allowed()
        {
            for ch in castling.chars() {
                if ch == '-' {
                    continue;
                }
                let c = if ch.is_ascii_uppercase() { Color::White } else { Color::Black };
                let rook = Piece::make(c, PieceType::Rook);
                let king_sq = pos.king_square(c);
                let up = ch.to_ascii_uppercase();
                let rsq: Option<Square> = match up {
                    'K' => {
                        let mut s = relative_square(c, squares::H1);
                        while s > king_sq && pos.piece_on(s) != rook {
                            s -= 1;
                        }
                        if pos.piece_on(s) == rook && s > king_sq { Some(s) } else { None }
                    }
                    'Q' => {
                        let mut s = relative_square(c, squares::A1);
                        while s < king_sq && pos.piece_on(s) != rook {
                            s += 1;
                        }
                        if pos.piece_on(s) == rook && s < king_sq { Some(s) } else { None }
                    }
                    'A'..='H' => {
                        let s = make_square(up as u8 - b'A', relative_rank(Color::White, relative_square(c, squares::A1)));
                        if pos.piece_on(s) == rook { Some(s) } else { None }
                    }
                    _ => None,
                };
                if let Some(rsq) = rsq {
                    pos.set_castling_right(c, rsq);
                }
            }
        }

        // 4. En passant square (only kept if a capture is actually possible)
        if let Some(ep) = parts.get(3)
            && *ep != "-"
            && let Some(ep_sq) = square_from_str(ep)
        {
            let us = pos.side_to_move;
            let them = us.flip();
            let ok = relative_rank(us, ep_sq) == 5
                && pawn_attacks(them, ep_sq) & pos.pieces_cp(us, PieceType::Pawn) != 0
                && pos.pieces_cp(them, PieceType::Pawn) & sq_bb((ep_sq as i32 - us.forward()) as Square) != 0
                && pos.pieces() & (sq_bb(ep_sq) | sq_bb((ep_sq as i32 + us.forward()) as Square)) == 0;
            if ok {
                pos.state_mut().ep_square = ep_sq;
            }
        }

        // 5-6. Halfmove clock and fullmove number
        let rule50 = parts.get(4).and_then(|s| s.parse::<i32>().ok()).unwrap_or(0);
        let fullmove = parts.get(5).and_then(|s| s.parse::<i32>().ok()).unwrap_or(1);
        pos.state_mut().rule50 = rule50;
        pos.game_ply = (2 * (fullmove - 1)).max(0) + (pos.side_to_move == Color::Black) as i32;
        pos.set_checks_given(variant, checks_given);

        pos.set_state();
        pos.nnue.reset();
        Ok(pos)
    }

    pub fn set_chess960(&mut self, on: bool) {
        self.chess960 = on;
    }

    pub fn is_chess960(&self) -> bool {
        self.chess960
    }

    #[inline(always)]
    fn state(&self) -> &StateInfo {
        // The stack always holds at least the root state.
        unsafe { self.states.get_unchecked(self.states.len() - 1) }
    }

    #[inline(always)]
    fn state_mut(&mut self) -> &mut StateInfo {
        let n = self.states.len() - 1;
        unsafe { self.states.get_unchecked_mut(n) }
    }

    #[inline(always)]
    fn state_at(&self, back: usize) -> &StateInfo {
        &self.states[self.states.len() - 1 - back]
    }

    fn set_castling_right(&mut self, c: Color, rfrom: Square) {
        let kfrom = self.king_square(c);
        let kingside = rfrom > kfrom;
        let cr = castling_rights_of(c) & if kingside { WHITE_OO | BLACK_OO } else { WHITE_OOO | BLACK_OOO };
        self.state_mut().castling_rights |= cr;
        self.castling_rights_mask[kfrom as usize] |= cr;
        self.castling_rights_mask[rfrom as usize] |= cr;
        self.castling_rook_square[cr as usize] = rfrom;
        let kto = relative_square(c, if kingside { squares::G1 } else { squares::C1 });
        let rto = relative_square(c, if kingside { squares::F1 } else { squares::D1 });
        self.castling_path[cr as usize] =
            (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & !(sq_bb(kfrom) | sq_bb(rfrom));
    }

    /// Recomputes every derived field of the current state from the board.
    fn set_state(&mut self) {
        let mut key = 0u64;
        let mut pawn_key = ZOBRIST.no_pawns;
        let mut material_key = 0u64;
        let mut nonpawn_key = [0u64; 2];
        let mut minor_key = 0u64;

        let mut b = self.pieces();
        while b != 0 {
            let s = pop_lsb(&mut b);
            let pc = self.piece_on(s);
            key ^= psq_key(pc, s);
            match pc.piece_type() {
                PieceType::Pawn => pawn_key ^= psq_key(pc, s),
                pt => {
                    nonpawn_key[pc.color().idx()] ^= psq_key(pc, s);
                    if matches!(pt, PieceType::Knight | PieceType::Bishop | PieceType::King) {
                        minor_key ^= psq_key(pc, s);
                    }
                }
            }
        }
        for pc in 0..PIECE_NB {
            for cnt in 0..self.piece_count[pc] as usize {
                material_key ^= ZOBRIST.psq[pc][cnt];
            }
        }

        let st = self.state();
        if st.ep_square != SQ_NONE {
            key ^= ZOBRIST.en_passant[file_of(st.ep_square) as usize];
        }
        if self.side_to_move == Color::Black {
            key ^= ZOBRIST.side;
        }
        key ^= ZOBRIST.castling[st.castling_rights as usize];
        key ^= variant_state_key(st);

        let us = self.side_to_move;
        let checkers = if self.variant.royal_king() {
            self.attackers_to(self.king_square(us), self.pieces()) & self.pieces_c(us.flip())
        } else {
            0
        };
        {
            let st = self.state_mut();
            st.key = key;
            st.pawn_key = pawn_key;
            st.material_key = material_key;
            st.nonpawn_key = nonpawn_key;
            st.minor_key = minor_key;
            st.checkers = checkers;
            st.repetition = 0;
        }
        self.set_check_info();
    }

    fn set_check_info(&mut self) {
        #[cfg(feature = "variants")]
        if !self.variant.royal_king() {
            let must = self.compute_must_capture();
            let st = self.state_mut();
            st.blockers_for_king = [0; 2];
            st.pinners = [0; 2];
            st.check_squares = [0; PIECE_TYPE_NB];
            st.must_capture = must;
            return;
        }
        let (wb, bp) = self.slider_blockers(self.pieces_c(Color::Black), self.king_square(Color::White));
        let (bb, wp) = self.slider_blockers(self.pieces_c(Color::White), self.king_square(Color::Black));
        let them = self.side_to_move.flip();
        let ksq = self.king_square(them);
        let occ = self.pieces();
        let bishop = bishop_attacks(ksq, occ);
        let rook = rook_attacks(ksq, occ);
        let st = self.state_mut();
        st.blockers_for_king[Color::White.idx()] = wb;
        st.blockers_for_king[Color::Black.idx()] = bb;
        st.pinners[Color::White.idx()] = wp;
        st.pinners[Color::Black.idx()] = bp;
        st.check_squares[PieceType::Pawn.idx()] = pawn_attacks(them, ksq);
        st.check_squares[PieceType::Knight.idx()] = knight_attacks(ksq);
        st.check_squares[PieceType::Bishop.idx()] = bishop;
        st.check_squares[PieceType::Rook.idx()] = rook;
        st.check_squares[PieceType::Queen.idx()] = bishop | rook;
        st.check_squares[PieceType::King.idx()] = 0;
    }

    /// Pieces of either colour that are the sole blocker between `s` and one of
    /// `sliders`, plus the sliders that pin a piece of `s`'s colour.
    fn slider_blockers(&self, sliders: Bitboard, s: Square) -> (Bitboard, Bitboard) {
        let mut blockers = 0;
        let mut pinners = 0;
        let mut snipers = ((pseudo_attacks(PieceType::Rook, s) & self.pieces_pp(PieceType::Queen, PieceType::Rook))
            | (pseudo_attacks(PieceType::Bishop, s) & self.pieces_pp(PieceType::Queen, PieceType::Bishop)))
            & sliders;
        let occupancy = self.pieces() ^ snipers;
        let target_color = self.piece_on(s).color();
        while snipers != 0 {
            let sniper_sq = pop_lsb(&mut snipers);
            let b = between_bb(s, sniper_sq) & occupancy;
            if b != 0 && !more_than_one(b) {
                blockers |= b;
                if b & self.pieces_c(target_color) != 0 {
                    pinners |= sq_bb(sniper_sq);
                }
            }
        }
        (blockers, pinners)
    }

    // Accessors

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline(always)]
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// Three-check: checks delivered by `c` so far. Always zero in a standard build.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub fn checks_given(&self, _c: Color) -> u8 {
        0
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub fn checks_given(&self, c: Color) -> u8 {
        self.state().checks_given[c.idx()]
    }

    /// Antichess: whether the side to move is obliged to capture.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub fn must_capture(&self) -> bool {
        false
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub fn must_capture(&self) -> bool {
        self.state().must_capture
    }

    /// Crazyhouse: pieces of type `pt` in `c`'s pocket.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub fn hand(&self, _c: Color, _pt: PieceType) -> u8 {
        0
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub fn hand(&self, c: Color, pt: PieceType) -> u8 {
        self.state().hand[c.idx()][pt.idx()]
    }

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub fn promoted(&self) -> Bitboard {
        0
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub fn promoted(&self) -> Bitboard {
        self.state().promoted
    }

    /// Whether a pawn may promote to a king (antichess).
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn king_promotions(&self) -> bool {
        false
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    fn king_promotions(&self) -> bool {
        self.variant == Variant::Antichess
    }

    /// Antichess: standing pat is not allowed while a capture is available.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub fn captures_forced(&self) -> bool {
        false
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub fn captures_forced(&self) -> bool {
        self.variant == Variant::Antichess && self.must_capture()
    }

    /// Whether material-draw knowledge (insufficient material, scaling) applies.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn material_draws_apply(&self) -> bool {
        true
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    fn material_draws_apply(&self) -> bool {
        self.variant.royal_king() && !matches!(self.variant, Variant::RacingKings | Variant::Crazyhouse | Variant::KingOfTheHill)
    }

    // Variant hooks. Each has a no-op twin so the call sites stay uniform and vanish
    // entirely when the feature is off.

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn set_pockets(&mut self, _variant: Variant, _hand: [[u8; 5]; 2], _promoted: Bitboard) {}
    #[cfg(feature = "variants")]
    #[inline]
    fn set_pockets(&mut self, variant: Variant, hand: [[u8; 5]; 2], promoted: Bitboard) {
        if variant == Variant::Crazyhouse {
            self.state_mut().hand = hand;
            self.state_mut().promoted = promoted;
        }
    }

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn set_checks_given(&mut self, _variant: Variant, _checks: [u8; 2]) {}
    #[cfg(feature = "variants")]
    #[inline]
    fn set_checks_given(&mut self, variant: Variant, checks: [u8; 2]) {
        if variant == Variant::ThreeCheck {
            self.state_mut().checks_given = checks;
        }
    }

    /// Crazyhouse: a captured piece joins the capturer's pocket, as a pawn if it had
    /// itself been promoted.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn variant_on_capture(&self, _st: &mut StateInfo, _us: Color, _captured: Piece, _capsq: Square) {}
    #[cfg(feature = "variants")]
    #[inline]
    fn variant_on_capture(&self, st: &mut StateInfo, us: Color, captured: Piece, capsq: Square) {
        if self.variant != Variant::Crazyhouse {
            return;
        }
        let hpt = if st.promoted & sq_bb(capsq) != 0 {
            st.promoted ^= sq_bb(capsq);
            st.key ^= ZOBRIST.promoted[capsq as usize];
            PieceType::Pawn
        } else {
            captured.piece_type()
        };
        let h = &mut st.hand[us.idx()][hpt.idx()];
        st.key ^= ZOBRIST.hand[us.idx()][hpt.idx()][*h as usize];
        *h += 1;
        st.key ^= ZOBRIST.hand[us.idx()][hpt.idx()][*h as usize];
    }

    /// Crazyhouse: the promoted-piece mark travels with the piece.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn variant_on_move(&self, _st: &mut StateInfo, _from: Square, _to: Square) {}
    #[cfg(feature = "variants")]
    #[inline]
    fn variant_on_move(&self, st: &mut StateInfo, from: Square, to: Square) {
        if st.promoted & sq_bb(from) != 0 {
            st.promoted ^= sq_bb(from) | sq_bb(to);
            st.key ^= ZOBRIST.promoted[from as usize] ^ ZOBRIST.promoted[to as usize];
        }
    }

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn variant_on_promotion(&self, _st: &mut StateInfo, _to: Square) {}
    #[cfg(feature = "variants")]
    #[inline]
    fn variant_on_promotion(&self, st: &mut StateInfo, to: Square) {
        if self.variant == Variant::Crazyhouse {
            st.promoted |= sq_bb(to);
            st.key ^= ZOBRIST.promoted[to as usize];
        }
    }

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    fn variant_on_check(&self, _st: &mut StateInfo, _us: Color) {}
    #[cfg(feature = "variants")]
    #[inline]
    fn variant_on_check(&self, st: &mut StateInfo, us: Color) {
        if self.variant == Variant::ThreeCheck && st.checks_given[us.idx()] < 3 {
            st.key ^= ZOBRIST.checks[us.idx()][st.checks_given[us.idx()] as usize];
            st.checks_given[us.idx()] += 1;
            st.key ^= ZOBRIST.checks[us.idx()][st.checks_given[us.idx()] as usize];
        }
    }

    /// Antichess only: does the side to move have any capture (including en passant)?
    #[cfg(feature = "variants")]
    fn compute_must_capture(&self) -> bool {
        let us = self.side_to_move;
        let them = us.flip();
        let enemy = self.pieces_c(them);
        let occ = self.pieces();
        let mut targets = enemy;
        if self.ep_square() != SQ_NONE {
            targets |= sq_bb(self.ep_square());
        }
        if pawn_attacks_bb(us, self.pieces_cp(us, PieceType::Pawn)) & targets != 0 {
            return true;
        }
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King] {
            let mut b = self.pieces_cp(us, pt);
            while b != 0 {
                let s = pop_lsb(&mut b);
                if attacks_bb(pt, s, occ) & enemy != 0 {
                    return true;
                }
            }
        }
        false
    }

    /// The castling move type doubles as "promote to king" in antichess, where real
    /// castling does not exist.
    #[inline(always)]
    pub fn is_real_castling(&self, m: Move) -> bool {
        m.is_castling() && self.variant.castling_allowed()
    }

    /// Piece a promotion creates, if `m` promotes (king promotions in antichess included).
    #[inline(always)]
    pub fn promotion_piece(&self, m: Move) -> Option<PieceType> {
        if m.is_promotion() {
            Some(m.promotion())
        } else if m.is_castling() && self.king_promotions() {
            Some(PieceType::King)
        } else {
            None
        }
    }

    /// UCI text of a move under this position's variant.
    pub fn move_to_uci(&self, m: Move) -> String {
        if m.is_castling() && self.king_promotions() {
            return format!("{}{}k", square_to_string(m.from_sq()), square_to_string(m.to_sq()));
        }
        m.to_uci(self.chess960)
    }

    /// Racing Kings: `Some(value)` from the side to move's perspective if the race is decided.
    #[cfg(feature = "variants")]
    pub fn racing_kings_result(&self, ply: usize) -> Option<Value> {
        let wk = rank_of(self.king_square(Color::White)) == 7;
        let bk = rank_of(self.king_square(Color::Black)) == 7;
        match self.side_to_move {
            Color::White => {
                if wk && bk {
                    Some(VALUE_DRAW)
                } else if bk {
                    Some(mated_in(ply))
                } else if wk {
                    // Black had its reply and did not reach the eighth rank.
                    Some(mate_in(ply))
                } else {
                    None
                }
            }
            Color::Black => {
                // White reaching first leaves Black one move to equalise, so only a Black
                // king already on the eighth rank decides here.
                if bk { Some(mate_in(ply)) } else { None }
            }
        }
    }

    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Piece {
        unsafe { *self.board.get_unchecked(sq as usize) }
    }

    #[inline(always)]
    pub fn is_empty_sq(&self, sq: Square) -> bool {
        self.piece_on(sq).is_none()
    }

    #[inline(always)]
    pub fn pieces(&self) -> Bitboard {
        self.by_color[0] | self.by_color[1]
    }

    #[inline(always)]
    pub fn pieces_c(&self, c: Color) -> Bitboard {
        self.by_color[c.idx()]
    }

    #[inline(always)]
    pub fn pieces_p(&self, pt: PieceType) -> Bitboard {
        self.by_type[pt.idx()]
    }

    #[inline(always)]
    pub fn pieces_pp(&self, a: PieceType, b: PieceType) -> Bitboard {
        self.by_type[a.idx()] | self.by_type[b.idx()]
    }

    #[inline(always)]
    pub fn pieces_cp(&self, c: Color, pt: PieceType) -> Bitboard {
        self.by_color[c.idx()] & self.by_type[pt.idx()]
    }

    #[inline(always)]
    pub fn pieces_cpp(&self, c: Color, a: PieceType, b: PieceType) -> Bitboard {
        self.by_color[c.idx()] & (self.by_type[a.idx()] | self.by_type[b.idx()])
    }

    #[inline(always)]
    pub fn king_square(&self, c: Color) -> Square {
        lsb(self.pieces_cp(c, PieceType::King))
    }

    #[inline(always)]
    pub fn count(&self, pc: Piece) -> i32 {
        self.piece_count[pc.idx()] as i32
    }

    #[inline(always)]
    pub fn count_cp(&self, c: Color, pt: PieceType) -> i32 {
        self.piece_count[Piece::make(c, pt).idx()] as i32
    }

    #[inline(always)]
    pub fn count_all(&self) -> i32 {
        popcount(self.pieces())
    }

    #[inline(always)]
    pub fn ep_square(&self) -> Square {
        self.state().ep_square
    }

    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.state().castling_rights
    }

    #[inline(always)]
    pub fn can_castle(&self, cr: u8) -> bool {
        self.state().castling_rights & cr != 0
    }

    #[inline(always)]
    pub fn castling_impeded(&self, cr: u8) -> bool {
        self.pieces() & self.castling_path[cr as usize] != 0
    }

    #[inline(always)]
    pub fn castling_rook_square(&self, cr: u8) -> Square {
        self.castling_rook_square[cr as usize]
    }

    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.state().checkers
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.state().checkers != 0
    }

    #[inline(always)]
    pub fn blockers_for_king(&self, c: Color) -> Bitboard {
        self.state().blockers_for_king[c.idx()]
    }

    #[inline(always)]
    pub fn pinners(&self, c: Color) -> Bitboard {
        self.state().pinners[c.idx()]
    }

    #[inline(always)]
    pub fn check_squares(&self, pt: PieceType) -> Bitboard {
        self.state().check_squares[pt.idx()]
    }

    #[inline(always)]
    pub fn key(&self) -> u64 {
        self.state().key
    }

    #[inline(always)]
    pub fn pawn_key(&self) -> u64 {
        self.state().pawn_key
    }

    #[inline(always)]
    pub fn material_key(&self) -> u64 {
        self.state().material_key
    }

    #[inline(always)]
    pub fn nonpawn_key(&self, c: Color) -> u64 {
        self.state().nonpawn_key[c.idx()]
    }

    #[inline(always)]
    pub fn minor_key(&self) -> u64 {
        self.state().minor_key
    }

    #[inline(always)]
    pub fn rule50_count(&self) -> i32 {
        self.state().rule50
    }

    #[inline(always)]
    pub fn plies_from_null(&self) -> i32 {
        self.state().plies_from_null
    }

    #[inline(always)]
    pub fn game_ply(&self) -> i32 {
        self.game_ply
    }

    #[inline(always)]
    pub fn captured_piece(&self) -> Piece {
        self.state().captured_piece
    }

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub fn moved_piece(&self, m: Move) -> Piece {
        self.piece_on(m.from_sq())
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub fn moved_piece(&self, m: Move) -> Piece {
        if m.is_drop() { Piece::make(self.side_to_move, m.drop_piece()) } else { self.piece_on(m.from_sq()) }
    }

    /// True if the move captures (en passant included).
    #[inline(always)]
    pub fn is_capture(&self, m: Move) -> bool {
        (self.piece_on(m.to_sq()).is_some() && !self.is_real_castling(m)) || m.is_en_passant()
    }

    /// Captures and queen promotions: the moves the capture stage generates.
    #[inline(always)]
    pub fn is_capture_stage(&self, m: Move) -> bool {
        self.is_capture(m) || (m.is_promotion() && m.promotion() == PieceType::Queen)
    }

    /// Piece type captured by `m`, if any (pawn for en passant).
    #[inline(always)]
    pub fn captured_type(&self, m: Move) -> Option<PieceType> {
        if m.is_en_passant() {
            Some(PieceType::Pawn)
        } else if self.is_real_castling(m) {
            None
        } else {
            let pc = self.piece_on(m.to_sq());
            if pc.is_some() { Some(pc.piece_type()) } else { None }
        }
    }

    #[inline(always)]
    pub fn non_pawn_material(&self, c: Color) -> Value {
        piece_value(PieceType::Knight) * self.count_cp(c, PieceType::Knight)
            + piece_value(PieceType::Bishop) * self.count_cp(c, PieceType::Bishop)
            + piece_value(PieceType::Rook) * self.count_cp(c, PieceType::Rook)
            + piece_value(PieceType::Queen) * self.count_cp(c, PieceType::Queen)
    }

    #[inline(always)]
    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        self.pieces_c(c) & !self.pieces_pp(PieceType::Pawn, PieceType::King) != 0
    }

    // Attack queries

    /// All pieces (both colours) attacking `s` with the given occupancy.
    #[inline]
    pub fn attackers_to(&self, s: Square, occupied: Bitboard) -> Bitboard {
        (pawn_attacks(Color::Black, s) & self.pieces_cp(Color::White, PieceType::Pawn))
            | (pawn_attacks(Color::White, s) & self.pieces_cp(Color::Black, PieceType::Pawn))
            | (knight_attacks(s) & self.pieces_p(PieceType::Knight))
            | (rook_attacks(s, occupied) & self.pieces_pp(PieceType::Rook, PieceType::Queen))
            | (bishop_attacks(s, occupied) & self.pieces_pp(PieceType::Bishop, PieceType::Queen))
            | (king_attacks(s) & self.pieces_p(PieceType::King))
    }

    /// Whether any piece of colour `c` attacks `s`, with early exits.
    #[inline]
    pub fn attackers_to_exist(&self, s: Square, occupied: Bitboard, c: Color) -> bool {
        let rq = self.pieces_cpp(c, PieceType::Rook, PieceType::Queen);
        if pseudo_attacks(PieceType::Rook, s) & rq != 0 && rook_attacks(s, occupied) & rq != 0 {
            return true;
        }
        let bq = self.pieces_cpp(c, PieceType::Bishop, PieceType::Queen);
        if pseudo_attacks(PieceType::Bishop, s) & bq != 0 && bishop_attacks(s, occupied) & bq != 0 {
            return true;
        }
        ((pawn_attacks(c.flip(), s) & self.pieces_p(PieceType::Pawn))
            | (knight_attacks(s) & self.pieces_p(PieceType::Knight))
            | (king_attacks(s) & self.pieces_p(PieceType::King)))
            & self.pieces_c(c)
            != 0
    }

    #[inline]
    pub fn is_square_attacked(&self, s: Square, by: Color) -> bool {
        self.attackers_to_exist(s, self.pieces(), by)
    }

    /// Full legality test for a pseudo-legal move.
    pub fn legal(&self, m: Move) -> bool {
        #[cfg(feature = "variants")]
        match self.variant {
            // No royalty: every pseudo-legal move is legal.
            Variant::Antichess => return true,
            // Giving check is forbidden.
            Variant::RacingKings => {
                if self.gives_check(m) {
                    return false;
                }
            }
            _ => {}
        }
        // Drops are generated onto blocking squares only, so nothing is left to test.
        if m.is_drop() {
            return true;
        }
        let us = self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq();
        debug_assert!(self.moved_piece(m).is_some() && self.moved_piece(m).color() == us);

        if m.is_en_passant() {
            // Test the king directly: the capture removes two pieces from the ray.
            let ksq = self.king_square(us);
            let capsq = (to as i32 - us.forward()) as Square;
            let occupied = (self.pieces() ^ sq_bb(from) ^ sq_bb(capsq)) | sq_bb(to);
            return rook_attacks(ksq, occupied) & self.pieces_cpp(us.flip(), PieceType::Queen, PieceType::Rook) == 0
                && bishop_attacks(ksq, occupied) & self.pieces_cpp(us.flip(), PieceType::Queen, PieceType::Bishop) == 0;
        }

        if m.is_castling() {
            // Generation only checked the path was empty; attacks are checked here.
            let kto = relative_square(us, if to > from { squares::G1 } else { squares::C1 });
            let step: i32 = if kto > from { -1 } else { 1 };
            let mut s = kto as i32;
            while s != from as i32 {
                if self.attackers_to_exist(s as Square, self.pieces(), us.flip()) {
                    return false;
                }
                s += step;
            }
            // Chess960: the rook may have been shielding the king from a slider.
            return !self.chess960 || self.blockers_for_king(us) & sq_bb(to) == 0;
        }

        if self.piece_on(from).piece_type() == PieceType::King {
            return !self.attackers_to_exist(to, self.pieces() ^ sq_bb(from), us.flip());
        }

        // A pinned piece may only move along the pin ray.
        self.blockers_for_king(us) & sq_bb(from) == 0 || aligned(from, to, self.king_square(us))
    }

    /// Validates a move that may come from the TT or a killer slot (possibly stale).
    pub fn pseudo_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq();
        if from >= 64 || to >= 64 {
            return false;
        }
        #[cfg(feature = "variants")]
        if m.is_drop() {
            let pt = m.drop_piece();
            if self.variant != Variant::Crazyhouse || self.hand(us, pt) == 0 || !self.is_empty_sq(to) {
                return false;
            }
            if pt == PieceType::Pawn && (RANK_1_BB | RANK_8_BB) & sq_bb(to) != 0 {
                return false;
            }
            if self.in_check() {
                return !more_than_one(self.checkers()) && between_bb(self.king_square(us), lsb(self.checkers())) & sq_bb(to) != 0;
            }
            return true;
        }
        let pc = self.piece_on(from);
        if pc.is_none() || pc.color() != us {
            return false;
        }

        // Special moves are rare: verify them by generation.
        if m.move_type() != MoveType::Normal {
            let mut list = MoveList::new();
            if self.in_check() {
                crate::movegen::generate_evasions(self, &mut list);
            } else {
                crate::movegen::generate_all(self, &mut list);
            }
            return list.contains(m);
        }

        if self.pieces_c(us) & sq_bb(to) != 0 {
            return false;
        }

        // Antichess: a quiet move is illegal while a capture exists.
        #[cfg(feature = "variants")]
        if self.variant == Variant::Antichess && self.must_capture() && self.piece_on(to).is_none() {
            return false;
        }

        if pc.piece_type() == PieceType::Pawn {
            // Promotions were handled above, so the target cannot be a back rank.
            if (RANK_8_BB | RANK_1_BB) & sq_bb(to) != 0 {
                return false;
            }
            let push = us.forward();
            let is_capture = pawn_attacks(us, from) & self.pieces_c(us.flip()) & sq_bb(to) != 0;
            let is_single = from as i32 + push == to as i32 && self.is_empty_sq(to);
            let is_double = from as i32 + 2 * push == to as i32
                && relative_rank(us, from) == 1
                && self.is_empty_sq(to)
                && self.is_empty_sq((to as i32 - push) as Square);
            if !(is_capture || is_single || is_double) {
                return false;
            }
        } else if attacks_bb(pc.piece_type(), from, self.pieces()) & sq_bb(to) == 0 {
            return false;
        }

        if self.in_check() && pc.piece_type() != PieceType::King {
            if more_than_one(self.checkers()) {
                return false;
            }
            if between_bb(self.king_square(us), lsb(self.checkers())) & sq_bb(to) == 0 {
                return false;
            }
        }
        true
    }

    /// Whether a pseudo-legal move gives check.
    pub fn gives_check(&self, m: Move) -> bool {
        if !self.variant.royal_king() {
            return false;
        }
        #[cfg(feature = "variants")]
        if m.is_drop() {
            return self.check_squares(m.drop_piece()) & sq_bb(m.to_sq()) != 0;
        }
        let us = self.side_to_move;
        let them = us.flip();
        let from = m.from_sq();
        let to = m.to_sq();
        let pt = self.piece_on(from).piece_type();

        // Direct check
        if self.check_squares(pt) & sq_bb(to) != 0 {
            return true;
        }

        // Discovered check
        let their_king = self.pieces_cp(them, PieceType::King);
        if self.blockers_for_king(them) & sq_bb(from) != 0 {
            return line_bb(from, to) & their_king == 0 || m.is_castling();
        }

        match m.move_type() {
            MoveType::Normal => false,
            MoveType::Promotion => attacks_bb(m.promotion(), to, self.pieces() ^ sq_bb(from)) & their_king != 0,
            MoveType::EnPassant => {
                let capsq = make_square(file_of(to), rank_of(from));
                let b = (self.pieces() ^ sq_bb(from) ^ sq_bb(capsq)) | sq_bb(to);
                let ksq = self.king_square(them);
                rook_attacks(ksq, b) & self.pieces_cpp(us, PieceType::Queen, PieceType::Rook) != 0
                    || bishop_attacks(ksq, b) & self.pieces_cpp(us, PieceType::Queen, PieceType::Bishop) != 0
            }
            MoveType::Castling => {
                let rto = relative_square(us, if to > from { squares::F1 } else { squares::D1 });
                self.check_squares(PieceType::Rook) & sq_bb(rto) != 0
            }
        }
    }

    // Board mutation

    #[inline(always)]
    fn put_piece(&mut self, pc: Piece, sq: Square) {
        self.board[sq as usize] = pc;
        self.by_type[pc.piece_type().idx()] |= sq_bb(sq);
        self.by_color[pc.color().idx()] |= sq_bb(sq);
        self.piece_count[pc.idx()] += 1;
        self.nnue.record(pc, SQ_NONE, sq);
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) {
        let pc = self.board[sq as usize];
        self.by_type[pc.piece_type().idx()] ^= sq_bb(sq);
        self.by_color[pc.color().idx()] ^= sq_bb(sq);
        self.board[sq as usize] = Piece::NONE;
        self.piece_count[pc.idx()] -= 1;
        self.nnue.record(pc, sq, SQ_NONE);
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square) {
        let pc = self.board[from as usize];
        let ft = sq_bb(from) | sq_bb(to);
        self.by_type[pc.piece_type().idx()] ^= ft;
        self.by_color[pc.color().idx()] ^= ft;
        self.board[from as usize] = Piece::NONE;
        self.board[to as usize] = pc;
        self.nnue.record(pc, from, to);
    }

    /// Makes `m`, which must be legal. `gives_check` is the precomputed check flag.
    pub fn make_move(&mut self, m: Move, gives_check: bool) {
        debug_assert!(m.is_ok());
        self.nnue.begin();
        let prev = *self.state();
        let mut st = StateInfo {
            castling_rights: prev.castling_rights,
            rule50: prev.rule50 + 1,
            plies_from_null: prev.plies_from_null + 1,
            ep_square: prev.ep_square,
            pawn_key: prev.pawn_key,
            nonpawn_key: prev.nonpawn_key,
            minor_key: prev.minor_key,
            material_key: prev.material_key,
            key: prev.key ^ ZOBRIST.side,
            checkers: 0,
            blockers_for_king: [0; 2],
            pinners: [0; 2],
            check_squares: [0; PIECE_TYPE_NB],
            captured_piece: Piece::NONE,
            repetition: 0,
            #[cfg(feature = "variants")]
            checks_given: prev.checks_given,
            #[cfg(feature = "variants")]
            must_capture: false,
            #[cfg(feature = "variants")]
            hand: prev.hand,
            #[cfg(feature = "variants")]
            promoted: prev.promoted,
        };
        self.game_ply += 1;

        #[cfg(feature = "variants")]
        if m.is_drop() {
            return self.make_drop(m, gives_check, st);
        }

        let us = self.side_to_move;
        let them = us.flip();
        let from = m.from_sq();
        let mut to = m.to_sq();
        let pc = self.piece_on(from);
        let mut captured = if m.is_en_passant() { Piece::make(them, PieceType::Pawn) } else { self.piece_on(to) };
        debug_assert!(pc.is_some() && pc.color() == us);
        let real_castle = self.is_real_castling(m);
        let promo_type = self.promotion_piece(m);

        if real_castle {
            debug_assert!(captured == Piece::make(us, PieceType::Rook));
            let (rfrom, rto) = self.do_castling(us, from, &mut to, true);
            st.key ^= psq_key(captured, rfrom) ^ psq_key(captured, rto);
            st.nonpawn_key[us.idx()] ^= psq_key(captured, rfrom) ^ psq_key(captured, rto);
            captured = Piece::NONE;
        }

        if captured.is_some() {
            let mut capsq = to;
            if captured.piece_type() == PieceType::Pawn {
                if m.is_en_passant() {
                    capsq = (to as i32 - us.forward()) as Square;
                }
                st.pawn_key ^= psq_key(captured, capsq);
            } else {
                st.nonpawn_key[them.idx()] ^= psq_key(captured, capsq);
                if matches!(captured.piece_type(), PieceType::Knight | PieceType::Bishop) {
                    st.minor_key ^= psq_key(captured, capsq);
                }
            }
            self.remove_piece(capsq);
            st.key ^= psq_key(captured, capsq);
            st.material_key ^= ZOBRIST.psq[captured.idx()][self.piece_count[captured.idx()] as usize];
            st.rule50 = 0;
            self.variant_on_capture(&mut st, us, captured, capsq);
        }

        st.key ^= psq_key(pc, from) ^ psq_key(pc, to);

        if st.ep_square != SQ_NONE {
            st.key ^= ZOBRIST.en_passant[file_of(st.ep_square) as usize];
            st.ep_square = SQ_NONE;
        }

        if st.castling_rights != 0
            && (self.castling_rights_mask[from as usize] | self.castling_rights_mask[to as usize]) != 0
        {
            st.key ^= ZOBRIST.castling[st.castling_rights as usize];
            st.castling_rights &= !(self.castling_rights_mask[from as usize] | self.castling_rights_mask[to as usize]);
            st.key ^= ZOBRIST.castling[st.castling_rights as usize];
        }

        if !real_castle {
            self.move_piece(from, to);
            self.variant_on_move(&mut st, from, to);
        }

        if pc.piece_type() == PieceType::Pawn {
            if (to ^ from) == 16
                && pawn_attacks(us, (to as i32 - us.forward()) as Square) & self.pieces_cp(them, PieceType::Pawn) != 0
            {
                st.ep_square = (to as i32 - us.forward()) as Square;
                st.key ^= ZOBRIST.en_passant[file_of(st.ep_square) as usize];
            } else if let Some(promo_pt) = promo_type {
                let promo = Piece::make(us, promo_pt);
                self.remove_piece(to);
                self.put_piece(promo, to);
                self.variant_on_promotion(&mut st, to);
                st.key ^= psq_key(pc, to) ^ psq_key(promo, to);
                st.pawn_key ^= psq_key(pc, to);
                st.material_key ^= ZOBRIST.psq[promo.idx()][self.piece_count[promo.idx()] as usize - 1]
                    ^ ZOBRIST.psq[pc.idx()][self.piece_count[pc.idx()] as usize];
                st.nonpawn_key[us.idx()] ^= psq_key(promo, to);
                if matches!(promo_pt, PieceType::Knight | PieceType::Bishop | PieceType::King) {
                    st.minor_key ^= psq_key(promo, to);
                }
            }
            st.pawn_key ^= psq_key(pc, from) ^ psq_key(pc, to);
            st.rule50 = 0;
        } else {
            st.nonpawn_key[us.idx()] ^= psq_key(pc, from) ^ psq_key(pc, to);
            if matches!(pc.piece_type(), PieceType::Knight | PieceType::Bishop | PieceType::King) {
                st.minor_key ^= psq_key(pc, from) ^ psq_key(pc, to);
            }
        }

        st.captured_piece = captured;
        self.side_to_move = them;
        st.checkers = if gives_check { self.attackers_to(self.king_square(them), self.pieces()) & self.pieces_c(us) } else { 0 };
        if gives_check {
            self.variant_on_check(&mut st, us);
        }

        self.finish_make(st);
    }

    /// Repetition bookkeeping, push, and derived check info shared by all move kinds.
    fn finish_make(&mut self, mut st: StateInfo) {
        let end = st.rule50.min(st.plies_from_null);
        if end >= 4 {
            let n = self.states.len(); // index of `st` once pushed
            let mut i = 4;
            while i <= end {
                let idx = n - i as usize;
                let prior = &self.states[idx];
                if prior.key == st.key {
                    st.repetition = if prior.repetition != 0 { -i } else { i };
                    break;
                }
                i += 2;
            }
        }
        self.states.push(st);
        self.nnue.push();
        self.set_check_info();
    }

    /// Crazyhouse drop: the piece leaves the pocket and appears on `to`.
    #[cfg(feature = "variants")]
    fn make_drop(&mut self, m: Move, gives_check: bool, mut st: StateInfo) {
        let us = self.side_to_move;
        let them = us.flip();
        let pt = m.drop_piece();
        let to = m.to_sq();
        let pc = Piece::make(us, pt);
        debug_assert!(st.hand[us.idx()][pt.idx()] > 0 && self.is_empty_sq(to));

        let h = &mut st.hand[us.idx()][pt.idx()];
        st.key ^= ZOBRIST.hand[us.idx()][pt.idx()][*h as usize];
        *h -= 1;
        st.key ^= ZOBRIST.hand[us.idx()][pt.idx()][*h as usize];

        self.put_piece(pc, to);
        st.key ^= psq_key(pc, to);
        if pt == PieceType::Pawn {
            st.pawn_key ^= psq_key(pc, to);
        } else {
            st.nonpawn_key[us.idx()] ^= psq_key(pc, to);
            if matches!(pt, PieceType::Knight | PieceType::Bishop) {
                st.minor_key ^= psq_key(pc, to);
            }
        }
        st.material_key ^= ZOBRIST.psq[pc.idx()][self.piece_count[pc.idx()] as usize - 1];

        if st.ep_square != SQ_NONE {
            st.key ^= ZOBRIST.en_passant[file_of(st.ep_square) as usize];
            st.ep_square = SQ_NONE;
        }
        st.captured_piece = Piece::NONE;
        self.side_to_move = them;
        st.checkers = if gives_check { self.attackers_to(self.king_square(them), self.pieces()) & self.pieces_c(us) } else { 0 };
        self.finish_make(st);
    }

    pub fn unmake_move(&mut self, m: Move) {
        self.side_to_move = self.side_to_move.flip();
        let us = self.side_to_move;
        #[cfg(feature = "variants")]
        if m.is_drop() {
            self.remove_piece(m.to_sq());
            self.states.pop();
            self.nnue.pop();
            self.game_ply -= 1;
            return;
        }
        let from = m.from_sq();
        let mut to = m.to_sq();

        if self.promotion_piece(m).is_some() {
            self.remove_piece(to);
            self.put_piece(Piece::make(us, PieceType::Pawn), to);
        }

        if self.is_real_castling(m) {
            self.do_castling(us, from, &mut to, false);
        } else {
            self.move_piece(to, from);
            let captured = self.state().captured_piece;
            if captured.is_some() {
                let mut capsq = to;
                if m.is_en_passant() {
                    capsq = (to as i32 - us.forward()) as Square;
                }
                self.put_piece(captured, capsq);
            }
        }

        self.states.pop();
        self.nnue.pop();
        self.game_ply -= 1;
    }

    /// Moves king and rook for castling (`do_it = false` reverses it). `to` holds the
    /// rook square on entry and the king's destination on exit.
    fn do_castling(&mut self, us: Color, from: Square, to: &mut Square, do_it: bool) -> (Square, Square) {
        let kingside = *to > from;
        let rfrom = *to;
        let rto = relative_square(us, if kingside { squares::F1 } else { squares::D1 });
        *to = relative_square(us, if kingside { squares::G1 } else { squares::C1 });
        let kto = *to;
        let king = Piece::make(us, PieceType::King);
        let rook = Piece::make(us, PieceType::Rook);
        if do_it {
            self.remove_piece(from);
            self.remove_piece(rfrom);
            self.put_piece(king, kto);
            self.put_piece(rook, rto);
        } else {
            self.remove_piece(kto);
            self.remove_piece(rto);
            self.put_piece(king, from);
            self.put_piece(rook, rfrom);
        }
        (rfrom, rto)
    }

    pub fn make_null_move(&mut self) {
        debug_assert!(!self.in_check());
        let prev = *self.state();
        let mut st = prev;
        st.key ^= ZOBRIST.side;
        if st.ep_square != SQ_NONE {
            st.key ^= ZOBRIST.en_passant[file_of(st.ep_square) as usize];
            st.ep_square = SQ_NONE;
        }
        st.rule50 += 1;
        st.plies_from_null = 0;
        st.captured_piece = Piece::NONE;
        st.repetition = 0;
        st.checkers = 0;
        self.side_to_move = self.side_to_move.flip();
        self.states.push(st);
        self.nnue.begin();
        self.nnue.push();
        self.set_check_info();
    }

    pub fn unmake_null_move(&mut self) {
        self.states.pop();
        self.nnue.pop();
        self.side_to_move = self.side_to_move.flip();
    }

    /// Network evaluation from the side to move's point of view.
    #[inline]
    pub fn nnue_evaluate(&mut self) -> Value {
        let kings = [self.king_square(Color::White), self.king_square(Color::Black)];
        let mut pieces = [0 as Bitboard; 12];
        for (i, bb) in pieces.iter_mut().enumerate() {
            *bb = self.by_color[i / 6] & self.by_type[i % 6];
        }
        let count = popcount(self.pieces());
        self.nnue.evaluate(self.side_to_move, kings, &pieces, count)
    }

    /// Hash key of the position after `m`, for prefetching the child's TT bucket.
    #[inline]
    pub fn key_after(&self, m: Move) -> u64 {
        let from = m.from_sq();
        let to = m.to_sq();
        #[cfg(feature = "variants")]
        if m.is_drop() {
            return self.key() ^ ZOBRIST.side ^ psq_key(Piece::make(self.side_to_move, m.drop_piece()), to);
        }
        let pc = self.piece_on(from);
        let captured = self.piece_on(to);
        let mut k = self.key() ^ ZOBRIST.side;
        if captured.is_some() {
            k ^= psq_key(captured, to);
        }
        k ^ psq_key(pc, from) ^ psq_key(pc, to)
    }

    // Draw detection

    /// Draw by the fifty-move rule (unless it is checkmate) or by repetition.
    pub fn is_draw(&self, ply: usize) -> bool {
        if self.rule50_count() > 99 && (!self.in_check() || self.has_legal_moves()) {
            return true;
        }
        self.is_repetition(ply)
    }

    /// A repetition strictly inside the search tree, or any threefold.
    #[inline(always)]
    pub fn is_repetition(&self, ply: usize) -> bool {
        let r = self.state().repetition;
        r != 0 && r < ply as i32
    }

    pub fn has_repeated(&self) -> bool {
        let mut end = self.rule50_count().min(self.plies_from_null());
        let mut back = 0usize;
        while end >= 4 {
            if self.state_at(back).repetition != 0 {
                return true;
            }
            back += 1;
            end -= 1;
        }
        false
    }

    /// Whether a reversible move by the side to move recreates an earlier position.
    /// Mirrors `is_draw` over all legal moves via the cuckoo tables.
    pub fn upcoming_repetition(&self, ply: usize) -> bool {
        let end = self.rule50_count().min(self.plies_from_null());
        if end < 3 {
            return false;
        }
        let n = self.states.len() - 1;
        let original_key = self.states[n].key;
        let mut other = original_key ^ self.states[n - 1].key ^ ZOBRIST.side;
        let ck = cuckoo();
        let occupied = self.pieces();

        let mut i = 3;
        while i <= end {
            let stp = n - i as usize; // position i plies ago (same side to move)
            other ^= self.states[stp + 1].key ^ self.states[stp].key ^ ZOBRIST.side;
            if other == 0 {
                let move_key = original_key ^ self.states[stp].key;
                let mut j = h1(move_key);
                if ck.keys[j] != move_key {
                    j = h2(move_key);
                }
                if ck.keys[j] == move_key {
                    let mv = ck.moves[j];
                    let s1 = mv.from_sq();
                    let s2 = mv.to_sq();
                    if (between_bb(s1, s2) ^ sq_bb(s2)) & occupied == 0 {
                        if ply > i as usize {
                            return true;
                        }
                        // At or before the root only a genuine repetition counts.
                        if self.states[stp].repetition != 0 {
                            return true;
                        }
                    }
                }
            }
            i += 2;
        }
        false
    }

    pub fn has_legal_moves(&self) -> bool {
        let mut list = MoveList::new();
        crate::movegen::generate_legal(self, &mut list);
        !list.is_empty()
    }

    // Static exchange evaluation

    /// True if the exchange started by `m` nets at least `threshold` (swap-list
    /// algorithm with pin awareness and x-ray discovery).
    pub fn see_ge(&self, m: Move, threshold: Value) -> bool {
        if m.is_drop() || m.move_type() != MoveType::Normal {
            return VALUE_ZERO >= threshold;
        }
        let from = m.from_sq();
        let to = m.to_sq();

        let mut swap = piece_value_of(self.piece_on(to)) - threshold;
        if swap < 0 {
            return false;
        }
        swap = piece_value_of(self.piece_on(from)) - swap;
        if swap <= 0 {
            return true;
        }

        let mut occupied = self.pieces() ^ sq_bb(from) ^ sq_bb(to);
        let mut stm = self.side_to_move;
        let mut attackers = self.attackers_to(to, occupied);
        let mut res = 1;

        loop {
            stm = stm.flip();
            attackers &= occupied;

            let mut stm_attackers = attackers & self.pieces_c(stm);
            if stm_attackers == 0 {
                break;
            }

            // Pinned pieces cannot recapture while their pinner is still on the board.
            if self.pinners(stm.flip()) & occupied != 0 {
                stm_attackers &= !self.blockers_for_king(stm);
                if stm_attackers == 0 {
                    break;
                }
            }

            res ^= 1;

            // Locate and remove the least valuable attacker, adding any x-ray
            // attacker revealed behind it.
            let bb = stm_attackers & self.pieces_p(PieceType::Pawn);
            if bb != 0 {
                swap = piece_value(PieceType::Pawn) - swap;
                if swap < res {
                    break;
                }
                occupied ^= least_significant_square_bb(bb);
                attackers |= bishop_attacks(to, occupied) & self.pieces_pp(PieceType::Bishop, PieceType::Queen);
                continue;
            }
            let bb = stm_attackers & self.pieces_p(PieceType::Knight);
            if bb != 0 {
                swap = piece_value(PieceType::Knight) - swap;
                if swap < res {
                    break;
                }
                occupied ^= least_significant_square_bb(bb);
                continue;
            }
            let bb = stm_attackers & self.pieces_p(PieceType::Bishop);
            if bb != 0 {
                swap = piece_value(PieceType::Bishop) - swap;
                if swap < res {
                    break;
                }
                occupied ^= least_significant_square_bb(bb);
                attackers |= bishop_attacks(to, occupied) & self.pieces_pp(PieceType::Bishop, PieceType::Queen);
                continue;
            }
            let bb = stm_attackers & self.pieces_p(PieceType::Rook);
            if bb != 0 {
                swap = piece_value(PieceType::Rook) - swap;
                if swap < res {
                    break;
                }
                occupied ^= least_significant_square_bb(bb);
                attackers |= rook_attacks(to, occupied) & self.pieces_pp(PieceType::Rook, PieceType::Queen);
                continue;
            }
            let bb = stm_attackers & self.pieces_p(PieceType::Queen);
            if bb != 0 {
                swap = piece_value(PieceType::Queen) - swap;
                if swap < res {
                    break;
                }
                occupied ^= least_significant_square_bb(bb);
                attackers |= (bishop_attacks(to, occupied) & self.pieces_pp(PieceType::Bishop, PieceType::Queen))
                    | (rook_attacks(to, occupied) & self.pieces_pp(PieceType::Rook, PieceType::Queen));
                continue;
            }
            // Only the king is left: it may capture only if no enemy attacker remains.
            return if attackers & !self.pieces_c(stm) != 0 { res == 0 } else { res != 0 };
        }
        res != 0
    }

    /// Signed SEE value: the material the side to move nets by starting the exchange.
    pub fn see(&self, m: Move) -> Value {
        // Binary search over see_ge is unnecessary here: a coarse scan over the
        // possible outcomes (bounded by a queen) is cheap and exact enough.
        let cap = self.captured_type(m).map_or(0, piece_value)
            + if m.is_promotion() { piece_value(m.promotion()) - piece_value(PieceType::Pawn) } else { 0 };
        let mover = if m.is_promotion() { piece_value(m.promotion()) } else { piece_value_of(self.piece_on(m.from_sq())) };
        let mut lo = cap - mover;
        let mut hi = cap;
        if !self.see_ge(m, lo) {
            return lo - 1;
        }
        while lo < hi {
            let mid = (lo + hi + 1) >> 1;
            if self.see_ge(m, mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    // Insufficient material

    /// Positions no side can ever win: bare kings, a single minor, or bishops all on
    /// one square colour with no pawns.
    pub fn is_insufficient_material(&self) -> bool {
        if !self.material_draws_apply() {
            return false;
        }
        if self.pieces_p(PieceType::Pawn) | self.pieces_p(PieceType::Rook) | self.pieces_p(PieceType::Queen) != 0 {
            return false;
        }
        let knights = self.pieces_p(PieceType::Knight);
        let bishops = self.pieces_p(PieceType::Bishop);
        let minors = knights | bishops;
        if !more_than_one(minors) {
            return true;
        }
        if knights == 0 {
            // Only bishops: drawn if they all share a square colour.
            return bishops & LIGHT_SQUARES == 0 || bishops & DARK_SQUARES == 0;
        }
        false
    }

    // Text output

    pub fn fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                let pc = self.piece_on(make_square(f, r));
                if pc.is_none() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push_str(&empty.to_string());
                        empty = 0;
                    }
                    s.push(pc.to_char());
                    if self.promoted() & sq_bb(make_square(f, r)) != 0 {
                        s.push('~');
                    }
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if r > 0 {
                s.push('/');
            }
        }
        #[cfg(feature = "variants")]
        if self.variant == Variant::Crazyhouse {
            s.push('[');
            for c in [Color::White, Color::Black] {
                for pt in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight, PieceType::Pawn] {
                    for _ in 0..self.hand(c, pt) {
                        s.push(Piece::make(c, pt).to_char());
                    }
                }
            }
            s.push(']');
        }
        s.push(' ');
        s.push(if self.side_to_move == Color::White { 'w' } else { 'b' });
        s.push(' ');
        let cr = self.castling_rights();
        if cr == 0 {
            s.push('-');
        } else {
            for (bit, ch) in [(WHITE_OO, 'K'), (WHITE_OOO, 'Q'), (BLACK_OO, 'k'), (BLACK_OOO, 'q')] {
                if cr & bit != 0 {
                    if self.chess960 {
                        let f = (b'A' + file_of(self.castling_rook_square(bit))) as char;
                        s.push(if bit & WHITE_CASTLING != 0 { f } else { f.to_ascii_lowercase() });
                    } else {
                        s.push(ch);
                    }
                }
            }
        }
        s.push(' ');
        s.push_str(&square_to_string(self.ep_square()));
        #[cfg(feature = "variants")]
        if self.variant == Variant::ThreeCheck {
            s.push_str(&format!(" {}+{}", 3 - self.checks_given(Color::White), 3 - self.checks_given(Color::Black)));
        }
        s.push_str(&format!(" {} {}", self.rule50_count(), 1 + (self.game_ply - (self.side_to_move == Color::Black) as i32) / 2));
        s
    }

    pub fn pretty(&self) -> String {
        let mut s = String::from("\n +---+---+---+---+---+---+---+---+\n");
        for r in (0..8).rev() {
            for f in 0..8 {
                s.push_str(&format!(" | {}", self.piece_on(make_square(f, r)).to_char()));
            }
            s.push_str(&format!(" | {}\n +---+---+---+---+---+---+---+---+\n", r + 1));
        }
        s.push_str("   a   b   c   d   e   f   g   h\n\n");
        s.push_str(&format!("Variant: {}\nFen: {}\nKey: {:016X}\nCheckers:", self.variant.name(), self.fen(), self.key()));
        let mut c = self.checkers();
        while c != 0 {
            s.push_str(&format!(" {}", square_to_string(pop_lsb(&mut c))));
        }
        s.push('\n');
        s
    }

    /// Parses a UCI move against the current legal move list.
    pub fn parse_uci_move(&self, s: &str) -> Option<Move> {
        let mut list = MoveList::new();
        crate::movegen::generate_legal(self, &mut list);
        let s = s.trim();
        for &m in list.iter() {
            if self.move_to_uci(m) == s {
                return Some(m);
            }
            // Accept king-takes-rook notation for castling in standard chess too.
            if self.is_real_castling(m) && m.to_uci(true) == s {
                return Some(m);
            }
        }
        None
    }
}
