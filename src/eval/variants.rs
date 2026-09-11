//! Variant evaluations: antichess (fewer pieces is better), racing kings (the race to
//! the eighth rank dominates) and the three-check bonus.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

/// Middlegame PeSTO material; the king counts as a piece in antichess.
const ANTI_VALUES: [Value; 6] = [82, 337, 365, 477, 1025, 300];
/// Per-check bonus toward the third check that ends the game.
const CHECK_BONUS: [Value; 4] = [0, 300, 900, 3000];
const RACE_RANK_BONUS: Value = 350;

/// Antichess: material is a liability. Mobility is mildly bad too (more moves means more
/// ways to be forced into captures), which the material term approximates.
pub fn antichess(pos: &Position) -> Value {
    let mut score = [0i32; 2];
    for c in [Color::White, Color::Black] {
        let mut total = 0;
        for pt in PieceType::ALL {
            total += ANTI_VALUES[pt.idx()] * popcount(pos.pieces_cp(c, pt));
        }
        score[c.idx()] = -total;
    }
    let stm = pos.side_to_move().idx();
    score[stm] - score[stm ^ 1]
}

/// Racing kings: king rank progress plus middlegame material.
pub fn racing_kings(pos: &Position) -> Value {
    let mut score = [0i32; 2];
    for c in [Color::White, Color::Black] {
        let ksq = pos.king_square(c);
        let mut v = RACE_RANK_BONUS * rank_of(ksq) as Value;
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            v += ANTI_VALUES[pt.idx()] * popcount(pos.pieces_cp(c, pt));
        }
        // King mobility: squares the king could step to without being attacked.
        let steps = king_attacks(ksq) & !pos.pieces_c(c);
        let mut b = steps;
        let mut free = 0;
        while b != 0 {
            let s = pop_lsb(&mut b);
            if !pos.attackers_to_exist(s, pos.pieces() ^ sq_bb(ksq), c.flip()) {
                free += 1;
            }
        }
        v += 12 * free;
        score[c.idx()] = v;
    }
    let stm = pos.side_to_move().idx();
    score[stm] - score[stm ^ 1]
}

/// Pocket pieces are worth about their board value; pawns and knights a little more
/// because they drop with tempo.
const HAND_VALUES: [Value; 5] = [120, 350, 330, 450, 900];

/// Crazyhouse: material in hand, side to move relative.
#[inline]
pub fn crazyhouse_hand(pos: &Position) -> Value {
    let mut v = [0i32; 2];
    for c in [Color::White, Color::Black] {
        for pt in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            v[c.idx()] += HAND_VALUES[pt.idx()] * pos.hand(c, pt) as Value;
        }
    }
    let stm = pos.side_to_move().idx();
    v[stm] - v[stm ^ 1]
}

#[rustfmt::skip]
const HILL_DIST: [i32; 8] = [3, 2, 1, 0, 0, 1, 2, 3];

/// King of the Hill: reward king proximity to the four centre squares.
#[inline]
pub fn king_of_the_hill_bonus(pos: &Position) -> Value {
    let dist = |c: Color| {
        let k = pos.king_square(c);
        HILL_DIST[file_of(k) as usize] + HILL_DIST[rank_of(k) as usize]
    };
    let us = pos.side_to_move();
    35 * (dist(us.flip()) - dist(us))
}

/// Three-check: added on top of the standard evaluation (side to move relative).
#[inline]
pub fn three_check_bonus(pos: &Position) -> Value {
    let us = pos.side_to_move();
    CHECK_BONUS[pos.checks_given(us) as usize] - CHECK_BONUS[pos.checks_given(us.flip()) as usize]
}
