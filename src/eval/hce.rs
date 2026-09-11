//! Hand-crafted evaluation: a bitboard port of Apeiron's standard-chess evaluator
//! (tapered PeSTO material and piece-square tables, non-linear mobility, pawn
//! structure, king shelter and attacker pressure, rook files, outposts, bishop pair).

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

const MG_VALUES: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUES: [i32; 6] = [94, 281, 297, 512, 936, 0];

#[rustfmt::skip]
const MG_KNIGHT_MOB: [i32; 9] = [-62, -36, -12,  0,  8, 14, 18, 20, 22];
#[rustfmt::skip]
const EG_KNIGHT_MOB: [i32; 9] = [-81, -46, -26, -8,  4, 10, 14, 16, 18];
#[rustfmt::skip]
const MG_BISHOP_MOB: [i32; 14] = [-48, -20,  6, 14, 20, 26, 30, 32, 32, 34, 36, 36, 38, 40];
#[rustfmt::skip]
const EG_BISHOP_MOB: [i32; 14] = [-59, -23, -3,  8, 16, 22, 28, 30, 34, 36, 38, 40, 42, 44];
#[rustfmt::skip]
const MG_ROOK_MOB: [i32; 15] = [-60, -20, 0, 2, 4,  8, 14, 18, 22, 22, 24, 26, 28, 28, 30];
#[rustfmt::skip]
const EG_ROOK_MOB: [i32; 15] = [-78, -17, 16, 28, 48, 62, 66, 74, 78, 80, 84, 86, 88, 88, 90];
#[rustfmt::skip]
const MG_QUEEN_MOB: [i32; 28] = [
    -30, -12, -8, -8, 10, 12, 12, 18, 18, 26, 30, 30,
     30, 30, 30, 30, 34, 34, 36, 36, 40, 42, 42, 42,
     44, 46, 46, 48,
];
#[rustfmt::skip]
const EG_QUEEN_MOB: [i32; 28] = [
    -48, -30,  -7, 14, 30, 40, 44, 50, 52, 58, 60, 64,
     72, 76, 80, 82, 84, 86, 90, 92, 94, 96, 96, 100,
    102, 104, 106, 112,
];

// Piece-square tables, listed from White's view with a8 first (index 0 = a8).
#[rustfmt::skip]
const MG_PAWN_PST: [i32; 64] = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
];
#[rustfmt::skip]
const EG_PAWN_PST: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const MG_KNIGHT_PST: [i32; 64] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
];
#[rustfmt::skip]
const EG_KNIGHT_PST: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];
#[rustfmt::skip]
const MG_BISHOP_PST: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];
#[rustfmt::skip]
const EG_BISHOP_PST: [i32; 64] = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
];
#[rustfmt::skip]
const MG_ROOK_PST: [i32; 64] = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
];
#[rustfmt::skip]
const EG_ROOK_PST: [i32; 64] = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
];
#[rustfmt::skip]
const MG_QUEEN_PST: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
];
#[rustfmt::skip]
const EG_QUEEN_PST: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];
#[rustfmt::skip]
const MG_KING_PST: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];
#[rustfmt::skip]
const EG_KING_PST: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

const MG_PST: [[i32; 64]; 6] = [MG_PAWN_PST, MG_KNIGHT_PST, MG_BISHOP_PST, MG_ROOK_PST, MG_QUEEN_PST, MG_KING_PST];
const EG_PST: [[i32; 64]; 6] = [EG_PAWN_PST, EG_KNIGHT_PST, EG_BISHOP_PST, EG_ROOK_PST, EG_QUEEN_PST, EG_KING_PST];

const MG_ISOLATED_PENALTY: i32 = 8;
const EG_ISOLATED_PENALTY: i32 = 16;
const MG_DOUBLED_PENALTY: i32 = 8;
const EG_DOUBLED_PENALTY: i32 = 16;
const MG_BACKWARD_PENALTY: i32 = 9;
const EG_BACKWARD_PENALTY: i32 = 20;

/// Connected pawn bonus by relative rank (1-based, index 0 unused).
#[rustfmt::skip]
const CONNECTED_BONUS: [i32; 8] = [0, 0, 7, 8, 12, 29, 48, 86];
/// Passed pawn bonus by relative rank (1-based).
#[rustfmt::skip]
const MG_PASSED_BONUS: [i32; 8] = [0, 0,  5, 10, 20,  40,  70, 120];
#[rustfmt::skip]
const EG_PASSED_BONUS: [i32; 8] = [0, 0, 10, 20, 40,  80, 140, 220];

const MG_BISHOP_PAIR: i32 = 30;
const EG_BISHOP_PAIR: i32 = 55;

const MG_ROOK_OPEN_FILE: i32 = 48;
const EG_ROOK_OPEN_FILE: i32 = 29;
const MG_ROOK_SEMI_OPEN_FILE: i32 = 19;
const EG_ROOK_SEMI_OPEN_FILE: i32 = 7;

const MG_OUTPOST_KNIGHT: i32 = 54;
const EG_OUTPOST_KNIGHT: i32 = 34;
const MG_OUTPOST_BISHOP: i32 = 28;
const EG_OUTPOST_BISHOP: i32 = 20;

const MG_MINOR_BEHIND_PAWN: i32 = 18;

/// King attack weights per piece type [pawn, knight, bishop, rook, queen, king].
const KING_ATTACK_WEIGHT: [i32; 6] = [0, 20, 20, 40, 80, 0];

/// Pawn shelter bonus by rank distance (1..=3) and file relation (0 = king file).
const SHELTER_BONUS: [[i32; 3]; 4] = [[0, 0, 0], [20, 14, 8], [10, 6, 2], [4, 2, 0]];
const SHELTER_MISSING_PENALTY: [i32; 3] = [22, 14, 6];

const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;

/// The PST index of `sq` for a piece of colour `c` (tables are written a8-first).
#[inline(always)]
const fn pst_index(c: Color, sq: Square) -> usize {
    match c {
        Color::White => (sq ^ 56) as usize,
        Color::Black => sq as usize,
    }
}

/// Files (bits 0-7) holding at least one pawn of `b`.
#[inline(always)]
const fn file_mask(b: Bitboard) -> u8 {
    let mut x = b;
    x |= x >> 32;
    x |= x >> 16;
    x |= x >> 8;
    (x & 0xFF) as u8
}

#[inline(always)]
fn adjacent_file_bits(f: u8) -> u8 {
    let mut m = 0u8;
    if f > 0 {
        m |= 1 << (f - 1);
    }
    if f < 7 {
        m |= 1 << (f + 1);
    }
    m
}

struct Side {
    mg: i32,
    eg: i32,
    danger: i32,
}

pub fn evaluate(pos: &Position) -> Value {
    let mut side = [Side { mg: 0, eg: 0, danger: 0 }, Side { mg: 0, eg: 0, danger: 0 }];
    let mut phase = 0;
    let occ = pos.pieces();
    let kings = [pos.king_square(Color::White), pos.king_square(Color::Black)];
    let pawns = [pos.pieces_cp(Color::White, PieceType::Pawn), pos.pieces_cp(Color::Black, PieceType::Pawn)];
    let pawn_files = [file_mask(pawns[0]), file_mask(pawns[1])];

    for c in [Color::White, Color::Black] {
        let us = c.idx();
        let them_c = c.flip();
        let them = them_c.idx();
        let own = pos.pieces_c(c);
        let enemy_king = kings[them];
        let own_files = pawn_files[us];
        let enemy_files = pawn_files[them];
        let outpost_ranks: Bitboard =
            if c == Color::White { RANK_4_BB | RANK_5_BB | RANK_6_BB } else { RANK_3_BB | RANK_4_BB | RANK_5_BB };

        // Pawns: material and PST only here (structure comes below).
        let mut b = pawns[us];
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let i = pst_index(c, sq);
            side[us].mg += MG_VALUES[0] + MG_PST[0][i];
            side[us].eg += EG_VALUES[0] + EG_PST[0][i];
        }

        // Knights
        let mut b = pos.pieces_cp(c, PieceType::Knight);
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let i = pst_index(c, sq);
            side[us].mg += MG_VALUES[1] + MG_PST[1][i];
            side[us].eg += EG_VALUES[1] + EG_PST[1][i];
            phase += PHASE_INC[1];

            let mob = popcount(knight_attacks(sq) & !own).min(8) as usize;
            side[us].mg += MG_KNIGHT_MOB[mob];
            side[us].eg += EG_KNIGHT_MOB[mob];

            if distance(sq, enemy_king) <= 3 {
                side[them].danger += KING_ATTACK_WEIGHT[1];
            }

            if sq_bb(sq) & outpost_ranks != 0 {
                let adj = adjacent_file_bits(file_of(sq));
                if own_files & adj != 0 && enemy_files & adj == 0 {
                    side[us].mg += MG_OUTPOST_KNIGHT;
                    side[us].eg += EG_OUTPOST_KNIGHT;
                }
            }

            let ahead = sq as i32 + c.forward();
            if is_square_ok(ahead) && pawns[us] & sq_bb(ahead as Square) != 0 {
                side[us].mg += MG_MINOR_BEHIND_PAWN;
            }
        }

        // Bishops
        let bishops = pos.pieces_cp(c, PieceType::Bishop);
        let mut b = bishops;
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let i = pst_index(c, sq);
            side[us].mg += MG_VALUES[2] + MG_PST[2][i];
            side[us].eg += EG_VALUES[2] + EG_PST[2][i];
            phase += PHASE_INC[2];

            let mob = popcount(bishop_attacks(sq, occ) & !own).min(13) as usize;
            side[us].mg += MG_BISHOP_MOB[mob];
            side[us].eg += EG_BISHOP_MOB[mob];

            if distance(sq, enemy_king) <= 4 {
                side[them].danger += KING_ATTACK_WEIGHT[2];
            }

            if sq_bb(sq) & outpost_ranks != 0 {
                let adj = adjacent_file_bits(file_of(sq));
                if own_files & adj != 0 && enemy_files & adj == 0 {
                    side[us].mg += MG_OUTPOST_BISHOP;
                    side[us].eg += EG_OUTPOST_BISHOP;
                }
            }
        }
        if bishops & LIGHT_SQUARES != 0 && bishops & DARK_SQUARES != 0 {
            side[us].mg += MG_BISHOP_PAIR;
            side[us].eg += EG_BISHOP_PAIR;
        }

        // Rooks
        let mut b = pos.pieces_cp(c, PieceType::Rook);
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let i = pst_index(c, sq);
            side[us].mg += MG_VALUES[3] + MG_PST[3][i];
            side[us].eg += EG_VALUES[3] + EG_PST[3][i];
            phase += PHASE_INC[3];

            let mob = popcount(rook_attacks(sq, occ) & !own).min(14) as usize;
            side[us].mg += MG_ROOK_MOB[mob];
            side[us].eg += EG_ROOK_MOB[mob];

            if distance(sq, enemy_king) <= 4 {
                side[them].danger += KING_ATTACK_WEIGHT[3];
            }

            let fbit = 1u8 << file_of(sq);
            if own_files & fbit == 0 {
                if enemy_files & fbit == 0 {
                    side[us].mg += MG_ROOK_OPEN_FILE;
                    side[us].eg += EG_ROOK_OPEN_FILE;
                } else {
                    side[us].mg += MG_ROOK_SEMI_OPEN_FILE;
                    side[us].eg += EG_ROOK_SEMI_OPEN_FILE;
                }
            }
        }

        // Queens
        let mut b = pos.pieces_cp(c, PieceType::Queen);
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let i = pst_index(c, sq);
            side[us].mg += MG_VALUES[4] + MG_PST[4][i];
            side[us].eg += EG_VALUES[4] + EG_PST[4][i];
            phase += PHASE_INC[4];

            let mob = popcount(queen_attacks(sq, occ) & !own).min(27) as usize;
            side[us].mg += MG_QUEEN_MOB[mob];
            side[us].eg += EG_QUEEN_MOB[mob];

            if distance(sq, enemy_king) <= 5 {
                side[them].danger += KING_ATTACK_WEIGHT[4];
            }
        }

        // King
        let i = pst_index(c, kings[us]);
        side[us].mg += MG_PST[5][i];
        side[us].eg += EG_PST[5][i];
    }

    // Pawn structure
    for c in [Color::White, Color::Black] {
        let us = c.idx();
        let them_c = c.flip();
        let them = them_c.idx();
        let own_pawns = pawns[us];
        let enemy_pawns = pawns[them];
        let own_files = pawn_files[us];
        let enemy_files = pawn_files[them];
        let (own_king, opp_king) = (kings[us], kings[them]);

        let mut b = own_pawns;
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let f = file_of(sq);
            let adj_files = adjacent_files_bb(sq);
            let has_neighbor = own_files & adjacent_file_bits(f) != 0;

            if !has_neighbor {
                side[us].mg -= MG_ISOLATED_PENALTY;
                side[us].eg -= EG_ISOLATED_PENALTY;
            }

            if own_pawns & file_bb(sq) & !sq_bb(sq) != 0 {
                side[us].mg -= MG_DOUBLED_PENALTY;
                side[us].eg -= EG_DOUBLED_PENALTY;
            }

            // Connected: phalanx neighbour or a defending pawn diagonally behind.
            let phalanx = own_pawns & adj_files & rank_bb(sq) != 0;
            let supported = own_pawns & pawn_attacks(them_c, sq) != 0;
            let rank_idx = (relative_rank(c, sq) as usize + 1).min(7);
            if phalanx || supported {
                let v = CONNECTED_BONUS[rank_idx];
                side[us].mg += v;
                side[us].eg += v * (rank_idx as i32 - 2).max(0) / 4;
            }

            // Backward: no friendly pawn behind on an adjacent file while an enemy pawn
            // contests the advance. Isolated pawns are already penalised, so skipped.
            if !supported && !phalanx && has_neighbor {
                let behind = forward_ranks_bb(them_c, sq);
                let no_support_behind = own_pawns & adj_files & behind == 0;
                let enemy_stop_file = enemy_files & adjacent_file_bits(f) != 0;
                if no_support_behind && enemy_stop_file {
                    side[us].mg -= MG_BACKWARD_PENALTY;
                    side[us].eg -= EG_BACKWARD_PENALTY;
                }
            }

            if enemy_pawns & passed_pawn_span(c, sq) == 0 {
                let mg_bonus = MG_PASSED_BONUS[rank_idx];
                let mut eg_bonus = EG_PASSED_BONUS[rank_idx];
                if rank_idx >= 3 {
                    let promo = make_square(f, if c == Color::White { 7 } else { 0 });
                    let own_dist = distance(own_king, promo).min(5) as i32;
                    let opp_dist = distance(opp_king, promo).min(5) as i32;
                    let w = (rank_idx as i32 - 2) * 5;
                    eg_bonus += (opp_dist - own_dist) * w;
                }
                side[us].mg += mg_bonus;
                side[us].eg += eg_bonus;
            }
        }
    }

    // King shelter (middlegame only): the king file and its neighbours.
    for c in [Color::White, Color::Black] {
        let us = c.idx();
        let ksq = kings[us];
        let kf = file_of(ksq) as i32;
        let own_pawns = pawns[us];
        let ahead = forward_ranks_bb(c, ksq);
        let mut shelter = 0;
        for df in -1i32..=1 {
            let f = (kf + df).clamp(0, 7) as u8;
            let file_rel = df.unsigned_abs().min(2) as usize;
            let on_file = own_pawns & file_bb_of(f) & ahead;
            if on_file != 0 {
                let closest = if c == Color::White { lsb(on_file) } else { msb(on_file) };
                let dist = rank_distance(ksq, closest) as usize;
                if dist <= 3 {
                    shelter += SHELTER_BONUS[dist][file_rel];
                    continue;
                }
            }
            shelter -= SHELTER_MISSING_PENALTY[file_rel];
        }
        side[us].mg += shelter;
    }

    // Attacker pressure: quadratic in the accumulated weight (middlegame only).
    for s in side.iter_mut() {
        if s.danger > 0 {
            s.mg -= s.danger * s.danger / 256;
        }
    }

    let stm = pos.side_to_move().idx();
    let other = stm ^ 1;
    let mg_score = side[stm].mg - side[other].mg;
    let eg_score = side[stm].eg - side[other].eg;
    let mg_phase = phase.min(MAX_PHASE);
    let eg_phase = MAX_PHASE - mg_phase;
    (mg_score * mg_phase + eg_score * eg_phase) / MAX_PHASE
}

/// Game phase in [0, 24]; 24 = full material.
#[inline]
pub fn game_phase(pos: &Position) -> i32 {
    let p = popcount(pos.pieces_p(PieceType::Knight)) * PHASE_INC[1]
        + popcount(pos.pieces_p(PieceType::Bishop)) * PHASE_INC[2]
        + popcount(pos.pieces_p(PieceType::Rook)) * PHASE_INC[3]
        + popcount(pos.pieces_p(PieceType::Queen)) * PHASE_INC[4];
    p.min(MAX_PHASE)
}
