//! KX-vs-K mating guidance. The board edge does the confining work, so a small
//! tiebreaker suffices: drive the bare king to the edge, keep our king close, and
//! for KBN aim at a corner the bishop controls.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

/// Manhattan distance from the board centre, 0 (d4/e4/d5/e5) to 6 (corners).
#[rustfmt::skip]
const CENTER_DIST: [i32; 8] = [3, 2, 1, 0, 0, 1, 2, 3];

/// Winning lead (in SEE material) before the guidance is applied at all.
const MIN_LEAD: Value = 250;
/// Ceiling of the term against a defended king so shaping never outweighs material.
const DEFENDED_CAP: Value = 200;
const BARE_CAP: Value = 450;

#[inline]
fn center_manhattan(sq: Square) -> i32 {
    CENTER_DIST[file_of(sq) as usize] + CENTER_DIST[rank_of(sq) as usize]
}

/// Number of non-king, non-pawn pieces.
#[inline]
fn officers(pos: &Position, c: Color) -> i32 {
    popcount(pos.pieces_c(c) & !pos.pieces_pp(PieceType::Pawn, PieceType::King))
}

/// Returns the mop-up term from the side to move's perspective and whether it is
/// active (the caller caps fifty-move damping while it is).
pub fn mop_up_term(pos: &Position) -> (Value, bool) {
    for winner in [Color::White, Color::Black] {
        let loser = winner.flip();
        let w_off = officers(pos, winner);
        let l_off = officers(pos, loser);
        let w_pawns = popcount(pos.pieces_cp(winner, PieceType::Pawn));
        let l_pawns = popcount(pos.pieces_cp(loser, PieceType::Pawn));

        // Loser: king plus at most one officer; winner: at least one officer; one side pawnless.
        if l_off > 1 || w_off < 1 || (w_pawns > 0 && l_pawns > 0) {
            continue;
        }
        let lead = pos.non_pawn_material(winner) - pos.non_pawn_material(loser) + 100 * (w_pawns - l_pawns);
        if lead < MIN_LEAD {
            continue;
        }
        if crate::eval::side_cannot_mate(pos, winner) && w_pawns == 0 {
            continue;
        }
        let bare = l_off == 0 && l_pawns == 0;
        let scale = if bare { 100 } else { 50 };

        let our_king = pos.king_square(winner);
        let their_king = pos.king_square(loser);

        // Push the bare king to the rim and bring our king toward it. The weights must
        // beat the loser's centralising king PST (~80cp centre-to-corner) by a margin.
        let mut bonus = 35 * center_manhattan(their_king);
        bonus += 10 * (14 - manhattan_distance(our_king, their_king) as i32);

        // K+B+N vs K: mate is only possible in a corner of the bishop's colour.
        let bishops = pos.pieces_cp(winner, PieceType::Bishop);
        let knights = pos.pieces_cp(winner, PieceType::Knight);
        if bare && w_off == 2 && popcount(bishops) == 1 && popcount(knights) == 1 {
            let light = bishops & LIGHT_SQUARES != 0;
            let (c1, c2) = if light { (squares::A8, squares::H1) } else { (squares::A1, squares::H8) };
            let d = distance(their_king, c1).min(distance(their_king, c2)) as i32;
            bonus += 20 * (7 - d);
        }

        let scaled = bonus * scale / 100;
        let cap = if bare { BARE_CAP } else { DEFENDED_CAP };
        let term = scaled.clamp(0, cap);
        let stm_term = if pos.side_to_move() == winner { term } else { -term };
        return (stm_term, true);
    }
    (0, false)
}
