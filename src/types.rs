//! Core value types: colors, pieces, squares, moves and score constants.
//!
//! Squares use little-endian rank-file mapping (a1 = 0, b1 = 1, ..., h8 = 63), so
//! "north" is `+8` and the bit layout of a bitboard reads like the board seen from
//! White's side rotated 180 degrees.

pub type Bitboard = u64;
pub type Square = u8;
pub type Value = i32;
pub type Depth = i32;

pub const SQ_NONE: Square = 64;

// Search score constants. Mate scores are `VALUE_MATE - ply`; anything at or beyond
// `VALUE_MATE_IN_MAX_PLY` is a proven win, so evaluation output is clamped below it.
pub const VALUE_ZERO: Value = 0;
pub const VALUE_DRAW: Value = 0;
pub const VALUE_MATE: Value = 32000;
pub const VALUE_INFINITE: Value = 32001;
pub const VALUE_NONE: Value = 32002;
pub const MAX_PLY: usize = 128;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY as Value;
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;
/// Largest magnitude a static evaluation may take.
pub const VALUE_EVAL_MAX: Value = VALUE_MATE_IN_MAX_PLY - 1;

#[inline(always)]
pub const fn mate_in(ply: usize) -> Value {
    VALUE_MATE - ply as Value
}

#[inline(always)]
pub const fn mated_in(ply: usize) -> Value {
    -VALUE_MATE + ply as Value
}

#[inline(always)]
pub const fn is_win(v: Value) -> bool {
    v >= VALUE_MATE_IN_MAX_PLY
}

#[inline(always)]
pub const fn is_loss(v: Value) -> bool {
    v <= VALUE_MATED_IN_MAX_PLY
}

#[inline(always)]
pub const fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

#[inline(always)]
pub const fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn from_idx(i: usize) -> Color {
        if i == 0 { Color::White } else { Color::Black }
    }

    /// Pawn push direction as a signed square offset.
    #[inline(always)]
    pub const fn forward(self) -> i32 {
        match self {
            Color::White => 8,
            Color::Black => -8,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

pub const PIECE_TYPE_NB: usize = 6;

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn from_idx(i: usize) -> PieceType {
        match i {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            _ => PieceType::King,
        }
    }

    pub const fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }
}

/// A colored piece packed into a byte: 0..5 white P..K, 6..11 black p..k, 12 = none.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece(pub u8);

pub const PIECE_NB: usize = 12;

impl Piece {
    pub const W_PAWN: Piece = Piece(0);
    pub const W_KNIGHT: Piece = Piece(1);
    pub const W_BISHOP: Piece = Piece(2);
    pub const W_ROOK: Piece = Piece(3);
    pub const W_QUEEN: Piece = Piece(4);
    pub const W_KING: Piece = Piece(5);
    pub const B_PAWN: Piece = Piece(6);
    pub const B_KNIGHT: Piece = Piece(7);
    pub const B_BISHOP: Piece = Piece(8);
    pub const B_ROOK: Piece = Piece(9);
    pub const B_QUEEN: Piece = Piece(10);
    pub const B_KING: Piece = Piece(11);
    pub const NONE: Piece = Piece(12);

    #[inline(always)]
    pub const fn make(c: Color, pt: PieceType) -> Piece {
        Piece((c as u8) * 6 + pt as u8)
    }

    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::from_idx((self.0 % 6) as usize)
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 < 6 { Color::White } else { Color::Black }
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 12
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != 12
    }

    pub const fn to_char(self) -> char {
        match self.0 {
            0 => 'P',
            1 => 'N',
            2 => 'B',
            3 => 'R',
            4 => 'Q',
            5 => 'K',
            6 => 'p',
            7 => 'n',
            8 => 'b',
            9 => 'r',
            10 => 'q',
            11 => 'k',
            _ => '.',
        }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        Some(match c {
            'P' => Piece::W_PAWN,
            'N' => Piece::W_KNIGHT,
            'B' => Piece::W_BISHOP,
            'R' => Piece::W_ROOK,
            'Q' => Piece::W_QUEEN,
            'K' => Piece::W_KING,
            'p' => Piece::B_PAWN,
            'n' => Piece::B_KNIGHT,
            'b' => Piece::B_BISHOP,
            'r' => Piece::B_ROOK,
            'q' => Piece::B_QUEEN,
            'k' => Piece::B_KING,
            _ => return None,
        })
    }
}

// Square helpers

#[inline(always)]
pub const fn make_square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

#[inline(always)]
pub const fn file_of(sq: Square) -> u8 {
    sq & 7
}

#[inline(always)]
pub const fn rank_of(sq: Square) -> u8 {
    sq >> 3
}

/// Rank as seen from `c`'s side (0 = its own back rank).
#[inline(always)]
pub const fn relative_rank(c: Color, sq: Square) -> u8 {
    rank_of(sq) ^ (c as u8 * 7)
}

#[inline(always)]
pub const fn relative_square(c: Color, sq: Square) -> Square {
    sq ^ (c as u8 * 56)
}

#[inline(always)]
pub const fn flip_rank(sq: Square) -> Square {
    sq ^ 56
}

#[inline(always)]
pub const fn is_square_ok(sq: i32) -> bool {
    sq >= 0 && sq < 64
}

pub fn square_to_string(sq: Square) -> String {
    if sq >= 64 {
        return "-".to_string();
    }
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{}{}", f, r)
}

pub fn square_from_str(s: &str) -> Option<Square> {
    let b = s.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let f = b[0];
    let r = b[1];
    if !(b'a'..=b'h').contains(&f) || !(b'1'..=b'8').contains(&r) {
        return None;
    }
    Some(make_square(f - b'a', r - b'1'))
}

#[allow(dead_code)]
pub mod squares {
    use super::Square;
    pub const A1: Square = 0;
    pub const B1: Square = 1;
    pub const C1: Square = 2;
    pub const D1: Square = 3;
    pub const E1: Square = 4;
    pub const F1: Square = 5;
    pub const G1: Square = 6;
    pub const H1: Square = 7;
    pub const A2: Square = 8;
    pub const B2: Square = 9;
    pub const C2: Square = 10;
    pub const D2: Square = 11;
    pub const E2: Square = 12;
    pub const F2: Square = 13;
    pub const G2: Square = 14;
    pub const H2: Square = 15;
    pub const A3: Square = 16;
    pub const D3: Square = 19;
    pub const E3: Square = 20;
    pub const H3: Square = 23;
    pub const A4: Square = 24;
    pub const D4: Square = 27;
    pub const E4: Square = 28;
    pub const H4: Square = 31;
    pub const A5: Square = 32;
    pub const D5: Square = 35;
    pub const E5: Square = 36;
    pub const H5: Square = 39;
    pub const A6: Square = 40;
    pub const D6: Square = 43;
    pub const E6: Square = 44;
    pub const H6: Square = 47;
    pub const A7: Square = 48;
    pub const B7: Square = 49;
    pub const C7: Square = 50;
    pub const D7: Square = 51;
    pub const E7: Square = 52;
    pub const F7: Square = 53;
    pub const G7: Square = 54;
    pub const H7: Square = 55;
    pub const A8: Square = 56;
    pub const B8: Square = 57;
    pub const C8: Square = 58;
    pub const D8: Square = 59;
    pub const E8: Square = 60;
    pub const F8: Square = 61;
    pub const G8: Square = 62;
    pub const H8: Square = 63;
}

/// Supported chess variants. Everything past Chess960 exists only under the
/// `variants` feature, so a standard build knows of exactly two and every variant
/// test folds to a constant at compile time.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Variant {
    #[default]
    Standard,
    /// Standard rules with shuffled back ranks; castling is king-takes-rook.
    Chess960,
    /// Captures are mandatory, the king is an ordinary piece (promotion to king is
    /// allowed) and having no legal move or no pieces wins.
    #[cfg(feature = "variants")]
    Antichess,
    /// Standard chess where delivering the third check wins.
    #[cfg(feature = "variants")]
    ThreeCheck,
    /// No pawns, checks are illegal, and the first king to reach the eighth rank wins
    /// (White reaching first gives Black one move to equalise into a draw).
    #[cfg(feature = "variants")]
    RacingKings,
    /// Captured pieces change hands and may be dropped on empty squares.
    #[cfg(feature = "variants")]
    Crazyhouse,
    /// A king reaching one of the four centre squares wins.
    #[cfg(feature = "variants")]
    KingOfTheHill,
}

impl Variant {
    #[cfg(not(feature = "variants"))]
    pub const ALL: [Variant; 2] = [Variant::Standard, Variant::Chess960];
    #[cfg(feature = "variants")]
    pub const ALL: [Variant; 7] = [
        Variant::Standard,
        Variant::Chess960,
        Variant::Antichess,
        Variant::ThreeCheck,
        Variant::RacingKings,
        Variant::Crazyhouse,
        Variant::KingOfTheHill,
    ];

    pub fn parse(s: &str) -> Option<Variant> {
        Some(match s.to_ascii_lowercase().as_str() {
            "standard" | "chess" | "normal" => Variant::Standard,
            "chess960" | "fischerandom" | "fischer" | "960" => Variant::Chess960,
            #[cfg(feature = "variants")]
            "antichess" | "giveaway" | "suicide" => Variant::Antichess,
            #[cfg(feature = "variants")]
            "3check" | "threecheck" | "three-check" | "three_check" => Variant::ThreeCheck,
            #[cfg(feature = "variants")]
            "racingkings" | "racing-kings" | "racing_kings" | "racing" => Variant::RacingKings,
            #[cfg(feature = "variants")]
            "crazyhouse" | "zh" | "crazy" => Variant::Crazyhouse,
            #[cfg(feature = "variants")]
            "kingofthehill" | "koth" | "king-of-the-hill" | "king_of_the_hill" => Variant::KingOfTheHill,
            _ => return None,
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Variant::Standard => "standard",
            Variant::Chess960 => "chess960",
            #[cfg(feature = "variants")]
            Variant::Antichess => "antichess",
            #[cfg(feature = "variants")]
            Variant::ThreeCheck => "3check",
            #[cfg(feature = "variants")]
            Variant::RacingKings => "racingkings",
            #[cfg(feature = "variants")]
            Variant::Crazyhouse => "crazyhouse",
            #[cfg(feature = "variants")]
            Variant::KingOfTheHill => "kingofthehill",
        }
    }

    pub const fn start_fen(self) -> &'static str {
        match self {
            #[cfg(feature = "variants")]
            Variant::Crazyhouse => "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
            #[cfg(feature = "variants")]
            Variant::Antichess => "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            #[cfg(feature = "variants")]
            Variant::RacingKings => "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1",
            _ => "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        }
    }

    /// Whether the king is royal (check, mate and pins exist).
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn royal_king(self) -> bool {
        true
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn royal_king(self) -> bool {
        !matches!(self, Variant::Antichess)
    }

    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn castling_allowed(self) -> bool {
        true
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn castling_allowed(self) -> bool {
        matches!(self, Variant::Standard | Variant::Chess960 | Variant::ThreeCheck | Variant::Crazyhouse | Variant::KingOfTheHill)
    }

    /// Standard material/endgame knowledge (insufficient material, mop-up, scaling) applies.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn standard_material(self) -> bool {
        true
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn standard_material(self) -> bool {
        matches!(self, Variant::Standard | Variant::Chess960)
    }

    /// Null move pruning is unsound where having no move is a win.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn allows_null_move(self) -> bool {
        true
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn allows_null_move(self) -> bool {
        !matches!(self, Variant::Antichess)
    }

    /// Whether every pseudo-legal move needs the full legality test (racing kings
    /// forbids giving check, which no cheap pin test covers).
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn needs_full_legality(self) -> bool {
        false
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn needs_full_legality(self) -> bool {
        matches!(self, Variant::RacingKings)
    }

    /// Whether pieces can be dropped from a pocket.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn has_drops(self) -> bool {
        false
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn has_drops(self) -> bool {
        matches!(self, Variant::Crazyhouse)
    }
}

// Castling rights bitfield.
pub const WHITE_OO: u8 = 1;
pub const WHITE_OOO: u8 = 2;
pub const BLACK_OO: u8 = 4;
pub const BLACK_OOO: u8 = 8;
pub const WHITE_CASTLING: u8 = WHITE_OO | WHITE_OOO;
pub const BLACK_CASTLING: u8 = BLACK_OO | BLACK_OOO;
pub const ANY_CASTLING: u8 = 15;

#[inline(always)]
pub const fn castling_rights_of(c: Color) -> u8 {
    match c {
        Color::White => WHITE_CASTLING,
        Color::Black => BLACK_CASTLING,
    }
}

/// Move flavours packed into the top two bits of a `Move`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum MoveType {
    Normal = 0,
    Promotion = 1 << 14,
    EnPassant = 2 << 14,
    Castling = 3 << 14,
}

/// 16-bit move: bits 0-5 destination, 6-11 origin, 12-13 promotion piece
/// (knight = 0 ... queen = 3), 14-15 move type. Castling is encoded as "king
/// captures own rook" so the same code serves Chess960 later.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);
    /// All bits set: never a real move or a drop.
    pub const NULL: Move = Move(0xFFFF);

    /// Crazyhouse drop: from == to == square, piece type in the top nibble (pawn = 0 ..
    /// queen = 4). A pawn drop on a1 would collide with NONE but pawns never drop on rank 1.
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn drop(pt: PieceType, sq: Square) -> Move {
        Move(((pt as u16) << 12) | ((sq as u16) << 6) | sq as u16)
    }

    /// Always false without the `variants` feature, which folds every drop branch away.
    #[cfg(not(feature = "variants"))]
    #[inline(always)]
    pub const fn is_drop(self) -> bool {
        false
    }
    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn is_drop(self) -> bool {
        self.0 != 0 && (self.0 & 0x3f) == ((self.0 >> 6) & 0x3f) && (self.0 >> 12) <= 4
    }

    #[cfg(feature = "variants")]
    #[inline(always)]
    pub const fn drop_piece(self) -> PieceType {
        PieceType::from_idx((self.0 >> 12) as usize)
    }

    #[inline(always)]
    pub const fn new(from: Square, to: Square) -> Move {
        Move(((from as u16) << 6) | to as u16)
    }

    #[inline(always)]
    pub const fn make(from: Square, to: Square, mt: MoveType, promo: PieceType) -> Move {
        let p = (promo as u16).saturating_sub(PieceType::Knight as u16) & 3;
        Move(mt as u16 | (p << 12) | ((from as u16) << 6) | to as u16)
    }

    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        ((self.0 >> 6) & 0x3f) as Square
    }

    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        (self.0 & 0x3f) as Square
    }

    /// `from * 64 + to`, the butterfly index used by history tables.
    #[inline(always)]
    pub const fn from_to(self) -> usize {
        (self.0 & 0xfff) as usize
    }

    #[inline(always)]
    pub const fn move_type(self) -> MoveType {
        if self.is_drop() {
            return MoveType::Normal;
        }
        match self.0 & (3 << 14) {
            0 => MoveType::Normal,
            0x4000 => MoveType::Promotion,
            0x8000 => MoveType::EnPassant,
            _ => MoveType::Castling,
        }
    }

    #[inline(always)]
    pub const fn promotion(self) -> PieceType {
        PieceType::from_idx((((self.0 >> 12) & 3) + 1) as usize)
    }

    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.0 & (3 << 14) == MoveType::Promotion as u16 && !self.is_drop()
    }

    #[inline(always)]
    pub const fn is_castling(self) -> bool {
        self.0 & (3 << 14) == MoveType::Castling as u16 && !self.is_drop()
    }

    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.0 & (3 << 14) == MoveType::EnPassant as u16 && !self.is_drop()
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }

    /// A real move (or drop): anything but NONE and NULL.
    #[inline(always)]
    pub const fn is_ok(self) -> bool {
        self.0 != 0 && self.0 != 0xFFFF
    }

    /// UCI notation. Castling is stored as king-takes-rook, so for standard chess the
    /// king's real destination is printed; `chess960` prints the rook square instead.
    pub fn to_uci(self, chess960: bool) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        if self == Move::NULL {
            return "0000".to_string();
        }
        #[cfg(feature = "variants")]
        if self.is_drop() {
            return format!("{}@{}", self.drop_piece().to_char().to_ascii_uppercase(), square_to_string(self.to_sq()));
        }
        let from = self.from_sq();
        let mut to = self.to_sq();
        if self.is_castling() && !chess960 {
            let kingside = to > from;
            let rank = rank_of(from);
            to = make_square(if kingside { 6 } else { 2 }, rank);
        }
        let mut s = format!("{}{}", square_to_string(from), square_to_string(to));
        if self.is_promotion() {
            s.push(self.promotion().to_char());
        }
        s
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci(false))
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci(false))
    }
}

/// TT bound type (2 bits).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    /// Score <= alpha (fail low, all-node)
    Upper = 1,
    /// Score >= beta (fail high, cut-node)
    Lower = 2,
    Exact = 3,
}

impl Bound {
    #[inline(always)]
    pub const fn from_u8(v: u8) -> Bound {
        match v & 3 {
            0 => Bound::None,
            1 => Bound::Upper,
            2 => Bound::Lower,
            _ => Bound::Exact,
        }
    }

    #[inline(always)]
    pub const fn has_lower(self) -> bool {
        (self as u8) & (Bound::Lower as u8) != 0
    }

    #[inline(always)]
    pub const fn has_upper(self) -> bool {
        (self as u8) & (Bound::Upper as u8) != 0
    }
}

/// Capacity of the fixed move buffers. Standard chess peaks at 218 legal moves;
/// crazyhouse drops can take a position well past that.
#[cfg(not(feature = "variants"))]
pub const MAX_MOVES: usize = 256;
#[cfg(feature = "variants")]
pub const MAX_MOVES: usize = 512;

/// Fixed-capacity move list.
#[derive(Clone)]
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    #[inline(always)]
    pub const fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.moves[..self.len].iter()
    }

    #[inline(always)]
    pub fn contains(&self, m: Move) -> bool {
        self.moves[..self.len].contains(&m)
    }

    #[inline(always)]
    pub fn swap(&mut self, a: usize, b: usize) {
        self.moves.swap(a, b);
    }

    #[inline(always)]
    pub fn swap_remove(&mut self, i: usize) -> Move {
        let m = self.moves[i];
        self.len -= 1;
        self.moves[i] = self.moves[self.len];
        m
    }
}

impl std::ops::Index<usize> for MoveList {
    type Output = Move;
    #[inline(always)]
    fn index(&self, i: usize) -> &Move {
        &self.moves[i]
    }
}

impl std::ops::IndexMut<usize> for MoveList {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Move {
        &mut self.moves[i]
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
