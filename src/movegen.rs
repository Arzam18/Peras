//! Pseudo-legal move generation split by stage: captures (including every
//! capture-promotion and quiet queen promotions), quiets (including quiet
//! underpromotions and castling), check evasions, and a fully legal generator for
//! the root and perft. Pinned-piece and king-move legality is left to `Position::legal`.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenType {
    Captures,
    Quiets,
    Evasions,
    NonEvasions,
}

#[inline(always)]
fn push_promotions(list: &mut MoveList, from: Square, to: Square, kind: GenType, is_capture: bool) {
    let all = kind == GenType::Evasions || kind == GenType::NonEvasions;
    if kind == GenType::Captures || all {
        list.push(Move::make(from, to, MoveType::Promotion, PieceType::Queen));
        if is_capture && kind == GenType::Captures {
            list.push(Move::make(from, to, MoveType::Promotion, PieceType::Rook));
            list.push(Move::make(from, to, MoveType::Promotion, PieceType::Bishop));
            list.push(Move::make(from, to, MoveType::Promotion, PieceType::Knight));
        }
    }
    if (kind == GenType::Quiets && !is_capture) || all {
        list.push(Move::make(from, to, MoveType::Promotion, PieceType::Rook));
        list.push(Move::make(from, to, MoveType::Promotion, PieceType::Bishop));
        list.push(Move::make(from, to, MoveType::Promotion, PieceType::Knight));
    }
}

fn generate_pawn_moves(pos: &Position, us: Color, list: &mut MoveList, target: Bitboard, kind: GenType) {
    let them = us.flip();
    let rank7 = if us == Color::White { RANK_7_BB } else { RANK_2_BB };
    let rank3 = if us == Color::White { RANK_3_BB } else { RANK_6_BB };
    let up = us.forward();
    let up_right = if us == Color::White { NORTH_EAST } else { SOUTH_WEST };
    let up_left = if us == Color::White { NORTH_WEST } else { SOUTH_EAST };

    let empty = !pos.pieces();
    let enemies = if kind == GenType::Evasions { pos.checkers() } else { pos.pieces_c(them) };
    let pawns_on7 = pos.pieces_cp(us, PieceType::Pawn) & rank7;
    let pawns_not_on7 = pos.pieces_cp(us, PieceType::Pawn) & !rank7;

    // Single and double pushes (no promotions)
    if kind != GenType::Captures {
        let mut b1 = shift(pawns_not_on7, up) & empty;
        let mut b2 = shift(b1 & rank3, up) & empty;
        if kind == GenType::Evasions {
            b1 &= target;
            b2 &= target;
        }
        while b1 != 0 {
            let to = pop_lsb(&mut b1);
            list.push(Move::new((to as i32 - up) as Square, to));
        }
        while b2 != 0 {
            let to = pop_lsb(&mut b2);
            list.push(Move::new((to as i32 - up - up) as Square, to));
        }
    }

    // Promotions
    if pawns_on7 != 0 {
        let mut b1 = shift(pawns_on7, up_right) & enemies;
        let mut b2 = shift(pawns_on7, up_left) & enemies;
        let mut b3 = shift(pawns_on7, up) & empty;
        if kind == GenType::Evasions {
            b3 &= target;
        }
        while b1 != 0 {
            let to = pop_lsb(&mut b1);
            push_promotions(list, (to as i32 - up_right) as Square, to, kind, true);
        }
        while b2 != 0 {
            let to = pop_lsb(&mut b2);
            push_promotions(list, (to as i32 - up_left) as Square, to, kind, true);
        }
        while b3 != 0 {
            let to = pop_lsb(&mut b3);
            push_promotions(list, (to as i32 - up) as Square, to, kind, false);
        }
    }

    // Captures and en passant
    if kind != GenType::Quiets {
        let mut b1 = shift(pawns_not_on7, up_right) & enemies;
        let mut b2 = shift(pawns_not_on7, up_left) & enemies;
        while b1 != 0 {
            let to = pop_lsb(&mut b1);
            list.push(Move::new((to as i32 - up_right) as Square, to));
        }
        while b2 != 0 {
            let to = pop_lsb(&mut b2);
            list.push(Move::new((to as i32 - up_left) as Square, to));
        }

        let ep = pos.ep_square();
        if ep != SQ_NONE {
            // If the block-target set contains the double-pushed pawn's origin square,
            // the check is a discovered one that en passant cannot resolve.
            if kind == GenType::Evasions && target & sq_bb((ep as i32 + up) as Square) != 0 {
                return;
            }
            let mut b = pawns_not_on7 & pawn_attacks(them, ep);
            while b != 0 {
                let from = pop_lsb(&mut b);
                list.push(Move::make(from, ep, MoveType::EnPassant, PieceType::Knight));
            }
        }
    }
}

#[inline(always)]
fn generate_piece_moves(pos: &Position, us: Color, pt: PieceType, list: &mut MoveList, target: Bitboard) {
    let mut bb = pos.pieces_cp(us, pt);
    let occ = pos.pieces();
    while bb != 0 {
        let from = pop_lsb(&mut bb);
        let mut b = attacks_bb(pt, from, occ) & target;
        while b != 0 {
            list.push(Move::new(from, pop_lsb(&mut b)));
        }
    }
}

fn generate(pos: &Position, list: &mut MoveList, kind: GenType) {
    let us = pos.side_to_move();
    let ksq = pos.king_square(us);
    let checkers = pos.checkers();

    // Double check: only the king may move.
    if !(kind == GenType::Evasions && more_than_one(checkers)) {
        let target = match kind {
            GenType::Evasions => between_bb(ksq, lsb(checkers)),
            GenType::NonEvasions => !pos.pieces_c(us),
            GenType::Captures => pos.pieces_c(us.flip()),
            GenType::Quiets => !pos.pieces(),
        };
        generate_pawn_moves(pos, us, list, target, kind);
        generate_piece_moves(pos, us, PieceType::Knight, list, target);
        generate_piece_moves(pos, us, PieceType::Bishop, list, target);
        generate_piece_moves(pos, us, PieceType::Rook, list, target);
        generate_piece_moves(pos, us, PieceType::Queen, list, target);
    }

    let king_target = match kind {
        GenType::Evasions | GenType::NonEvasions => !pos.pieces_c(us),
        GenType::Captures => pos.pieces_c(us.flip()),
        GenType::Quiets => !pos.pieces(),
    };
    let mut b = king_attacks(ksq) & king_target;
    while b != 0 {
        list.push(Move::new(ksq, pop_lsb(&mut b)));
    }

    if (kind == GenType::Quiets || kind == GenType::NonEvasions) && pos.can_castle(castling_rights_of(us)) {
        let (oo, ooo) = if us == Color::White { (WHITE_OO, WHITE_OOO) } else { (BLACK_OO, BLACK_OOO) };
        for cr in [oo, ooo] {
            if pos.can_castle(cr) && !pos.castling_impeded(cr) {
                list.push(Move::make(ksq, pos.castling_rook_square(cr), MoveType::Castling, PieceType::Knight));
            }
        }
    }

    // Crazyhouse drops: any empty square, or only blocking squares against a single check.
    #[cfg(feature = "variants")]
    if pos.variant() == Variant::Crazyhouse && kind != GenType::Captures && !(kind == GenType::Evasions && more_than_one(checkers)) {
        let mut target = !pos.pieces();
        if kind == GenType::Evasions {
            target &= between_bb(ksq, lsb(checkers));
        }
        for pt in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            if pos.hand(us, pt) == 0 {
                continue;
            }
            let mut t = target;
            if pt == PieceType::Pawn {
                t &= !(RANK_1_BB | RANK_8_BB);
            }
            while t != 0 {
                list.push(Move::drop(pt, pop_lsb(&mut t)));
            }
        }
    }
}

// Antichess: the king is an ordinary piece, pawns may promote to a king (encoded with
// the otherwise unused castling move type), and captures are compulsory.

#[cfg(feature = "variants")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum AntiMode {
    Captures,
    Quiets,
    All,
}

#[cfg(feature = "variants")]
#[inline(always)]
fn push_anti_promotions(list: &mut MoveList, from: Square, to: Square) {
    for pt in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
        list.push(Move::make(from, to, MoveType::Promotion, pt));
    }
    list.push(Move::make(from, to, MoveType::Castling, PieceType::Knight));
}

#[cfg(feature = "variants")]
fn generate_antichess(pos: &Position, list: &mut MoveList, mode: AntiMode) {
    let us = pos.side_to_move();
    let them = us.flip();
    let occ = pos.pieces();
    let empty = !occ;
    let enemies = pos.pieces_c(them);
    let rank7 = if us == Color::White { RANK_7_BB } else { RANK_2_BB };
    let rank3 = if us == Color::White { RANK_3_BB } else { RANK_6_BB };
    let up = us.forward();
    let up_right = if us == Color::White { NORTH_EAST } else { SOUTH_WEST };
    let up_left = if us == Color::White { NORTH_WEST } else { SOUTH_EAST };
    let pawns = pos.pieces_cp(us, PieceType::Pawn);
    let pawns_on7 = pawns & rank7;
    let pawns_not_on7 = pawns & !rank7;

    if mode != AntiMode::Captures {
        let mut b1 = shift(pawns_not_on7, up) & empty;
        let mut b2 = shift(b1 & rank3, up) & empty;
        while b1 != 0 {
            let to = pop_lsb(&mut b1);
            list.push(Move::new((to as i32 - up) as Square, to));
        }
        while b2 != 0 {
            let to = pop_lsb(&mut b2);
            list.push(Move::new((to as i32 - up - up) as Square, to));
        }
        let mut b3 = shift(pawns_on7, up) & empty;
        while b3 != 0 {
            let to = pop_lsb(&mut b3);
            push_anti_promotions(list, (to as i32 - up) as Square, to);
        }
    }
    if mode != AntiMode::Quiets {
        for (bb, dir) in [(pawns_on7, up_right), (pawns_on7, up_left)] {
            let mut b = shift(bb, dir) & enemies;
            while b != 0 {
                let to = pop_lsb(&mut b);
                push_anti_promotions(list, (to as i32 - dir) as Square, to);
            }
        }
        for dir in [up_right, up_left] {
            let mut b = shift(pawns_not_on7, dir) & enemies;
            while b != 0 {
                let to = pop_lsb(&mut b);
                list.push(Move::new((to as i32 - dir) as Square, to));
            }
        }
        let ep = pos.ep_square();
        if ep != SQ_NONE {
            let mut b = pawns_not_on7 & pawn_attacks(them, ep);
            while b != 0 {
                let from = pop_lsb(&mut b);
                list.push(Move::make(from, ep, MoveType::EnPassant, PieceType::Knight));
            }
        }
    }

    let target = match mode {
        AntiMode::Captures => enemies,
        AntiMode::Quiets => empty,
        AntiMode::All => !pos.pieces_c(us),
    };
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King] {
        generate_piece_moves(pos, us, pt, list, target);
    }
}

/// Captures, en passant, every capture-promotion, and quiet queen promotions.
#[inline]
pub fn generate_captures(pos: &Position, list: &mut MoveList) {
    #[cfg(feature = "variants")]
    if pos.variant() == Variant::Antichess {
        return generate_antichess(pos, list, AntiMode::Captures);
    }
    debug_assert!(!pos.in_check());
    generate(pos, list, GenType::Captures);
}

/// Non-capturing moves: pushes, piece moves, castling and quiet underpromotions.
/// Empty in antichess while a capture is available.
#[inline]
pub fn generate_quiets(pos: &Position, list: &mut MoveList) {
    #[cfg(feature = "variants")]
    if pos.variant() == Variant::Antichess {
        if !pos.must_capture() {
            generate_antichess(pos, list, AntiMode::Quiets);
        }
        return;
    }
    debug_assert!(!pos.in_check());
    generate(pos, list, GenType::Quiets);
}

/// All pseudo-legal moves when not in check (only the captures in antichess when one exists).
#[inline]
pub fn generate_all(pos: &Position, list: &mut MoveList) {
    #[cfg(feature = "variants")]
    if pos.variant() == Variant::Antichess {
        let mode = if pos.must_capture() { AntiMode::Captures } else { AntiMode::All };
        return generate_antichess(pos, list, mode);
    }
    debug_assert!(!pos.in_check());
    generate(pos, list, GenType::NonEvasions);
}

/// Pseudo-legal check evasions (king moves, blocks, captures of the checker).
#[inline]
pub fn generate_evasions(pos: &Position, list: &mut MoveList) {
    #[cfg(feature = "variants")]
    if pos.variant() == Variant::Antichess {
        return generate_all(pos, list);
    }
    debug_assert!(pos.in_check());
    generate(pos, list, GenType::Evasions);
}

/// Strictly legal moves.
pub fn generate_legal(pos: &Position, list: &mut MoveList) {
    #[cfg(feature = "variants")]
    if pos.variant() == Variant::Antichess {
        return generate_all(pos, list);
    }
    let mut pseudo = MoveList::new();
    if pos.in_check() {
        generate_evasions(pos, &mut pseudo);
    } else {
        generate_all(pos, &mut pseudo);
    }
    let us = pos.side_to_move();
    let pinned = pos.blockers_for_king(us) & pos.pieces_c(us);
    let ksq = pos.king_square(us);
    // Racing kings forbids checks, so every move needs the full test.
    let check_all = pos.variant().needs_full_legality();
    for &m in pseudo.iter() {
        let from = m.from_sq();
        let needs_check = check_all || pinned & sq_bb(from) != 0 || from == ksq || m.is_en_passant();
        if !needs_check || pos.legal(m) {
            list.push(m);
        }
    }
}

/// Node count of the legal move tree to `depth` (perft).
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate_legal(pos, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for &m in list.iter() {
        let gives_check = pos.gives_check(m);
        pos.make_move(m, gives_check);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m);
    }
    nodes
}

/// Perft with a per-root-move breakdown printed to stdout.
pub fn perft_divide(pos: &mut Position, depth: u32) -> u64 {
    let mut list = MoveList::new();
    generate_legal(pos, &mut list);
    let mut total = 0;
    for &m in list.iter() {
        let gives_check = pos.gives_check(m);
        pos.make_move(m, gives_check);
        let n = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake_move(m);
        println!("{}: {}", pos.move_to_uci(m), n);
        total += n;
    }
    total
}
