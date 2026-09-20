//! KX-vs-K specialised evaluation.
//!
//! The network is poor at forcing these mates: it reads the material, saturates, and then
//! offers nothing to steer by once the bare king is loose, so the winning side drifts until
//! the fifty-move rule takes the win away. A bonus added on top of that cannot fix it,
//! because the network's own noise is the larger term. Instead the evaluation is replaced
//! outright, the way a mate is really scored: a constant saying the position is won, plus a
//! small gradient that always points at the mate.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

/// Base score for a position the winning side mates from by force. Far above any shaping
/// below, and far enough below the mate range to still read as an evaluation.
const KNOWN_WIN: Value = 2000;

/// Weight on the corner term for K+B+N, which has to outweigh everything else: the mate
/// exists in only two of the four corners, so steering into a wrong one is a lost game.
const CORNER_WEIGHT: i32 = 560;

/// Distance from the nearer edge, 0 on the rim to 3 in the middle.
#[inline]
fn edge_distance(x: u8) -> i32 {
    let x = x as i32;
    x.min(7 - x)
}

/// Drives the bare king to an edge: 90 on the rim, down to 28 in the centre.
#[inline]
fn push_to_edge(sq: Square) -> i32 {
    let rd = edge_distance(rank_of(sq));
    let fd = edge_distance(file_of(sq));
    90 - (7 * fd * fd / 2 + 7 * rd * rd / 2)
}

/// Brings the kings together: 120 when adjacent, falling to 0 across the board. The mate
/// needs our king as much as the piece, so this is weighted like the edge term, not below it.
#[inline]
fn push_close(a: Square, b: Square) -> i32 {
    140 - 20 * distance(a, b) as i32
}

/// Distance from the a1/h8 corners: 0 along the a8-h1 diagonal, 7 at a1 and h8.
#[inline]
fn push_to_corner(sq: Square) -> i32 {
    (7 - rank_of(sq) as i32 - file_of(sq) as i32).abs()
}

/// Mirrors a square left to right, turning a dark-corner gradient into a light-corner one.
#[inline]
fn flip_file(sq: Square) -> Square {
    sq ^ 7
}

/// A complete evaluation for KX vs K from the side to move's point of view, or `None` when
/// the position is not one and the network should speak instead.
pub fn kx_vs_k(pos: &Position) -> Option<Value> {
    for winner in [Color::White, Color::Black] {
        let loser = winner.flip();

        // The defender must be a bare king, and the winner must be able to force mate with
        // pieces alone. With a pawn anywhere the win runs through promotion, which the
        // network handles and which is not always a win at all.
        if pos.pieces_c(loser) != pos.pieces_cp(loser, PieceType::King) {
            continue;
        }
        if pos.pieces_cp(winner, PieceType::Pawn) != 0 {
            continue;
        }
        if crate::eval::side_cannot_mate(pos, winner) {
            continue;
        }

        let wk = pos.king_square(winner);
        let lk = pos.king_square(loser);
        let bishops = pos.pieces_cp(winner, PieceType::Bishop);
        let knights = pos.pieces_cp(winner, PieceType::Knight);

        let shaping = if popcount(bishops) == 1 && popcount(knights) == 1 {
            // K+B+N mates only in the two corners the bishop attacks, so distance to the
            // nearer of those is the whole objective; the generic edge term is dropped
            // because it pulls just as hard toward the two corners that cannot mate.
            let dark_bishop = bishops & DARK_SQUARES != 0;
            let target = if dark_bishop { lk } else { flip_file(lk) };
            CORNER_WEIGHT * push_to_corner(target) + push_close(wk, lk)
        } else {
            push_to_edge(lk) + push_close(wk, lk)
        };

        let v = KNOWN_WIN + pos.non_pawn_material(winner) + shaping;
        return Some(if pos.side_to_move() == winner { v } else { -v });
    }
    None
}
