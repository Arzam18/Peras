//! Bitboard primitives and precomputed attack tables (leapers, fancy magic
//! sliders, between/line rays, distances). `init()` must run once before use.

use crate::types::*;
use std::cell::UnsafeCell;
use std::sync::Once;

pub const FILE_A_BB: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_B_BB: Bitboard = FILE_A_BB << 1;
pub const FILE_C_BB: Bitboard = FILE_A_BB << 2;
pub const FILE_D_BB: Bitboard = FILE_A_BB << 3;
pub const FILE_E_BB: Bitboard = FILE_A_BB << 4;
pub const FILE_F_BB: Bitboard = FILE_A_BB << 5;
pub const FILE_G_BB: Bitboard = FILE_A_BB << 6;
pub const FILE_H_BB: Bitboard = FILE_A_BB << 7;

pub const RANK_1_BB: Bitboard = 0xFF;
pub const RANK_2_BB: Bitboard = RANK_1_BB << 8;
pub const RANK_3_BB: Bitboard = RANK_1_BB << 16;
pub const RANK_4_BB: Bitboard = RANK_1_BB << 24;
pub const RANK_5_BB: Bitboard = RANK_1_BB << 32;
pub const RANK_6_BB: Bitboard = RANK_1_BB << 40;
pub const RANK_7_BB: Bitboard = RANK_1_BB << 48;
pub const RANK_8_BB: Bitboard = RANK_1_BB << 56;

pub const LIGHT_SQUARES: Bitboard = 0x55AA_55AA_55AA_55AA;
pub const DARK_SQUARES: Bitboard = !LIGHT_SQUARES;
pub const QUEEN_SIDE: Bitboard = FILE_A_BB | FILE_B_BB | FILE_C_BB | FILE_D_BB;
pub const KING_SIDE: Bitboard = FILE_E_BB | FILE_F_BB | FILE_G_BB | FILE_H_BB;
pub const CENTER: Bitboard = (FILE_D_BB | FILE_E_BB) & (RANK_4_BB | RANK_5_BB);

pub const NORTH: i32 = 8;
pub const SOUTH: i32 = -8;
pub const EAST: i32 = 1;
pub const WEST: i32 = -1;
pub const NORTH_EAST: i32 = 9;
pub const NORTH_WEST: i32 = 7;
pub const SOUTH_EAST: i32 = -7;
pub const SOUTH_WEST: i32 = -9;

#[inline(always)]
pub const fn sq_bb(sq: Square) -> Bitboard {
    1u64 << sq
}

#[inline(always)]
pub const fn file_bb(sq: Square) -> Bitboard {
    FILE_A_BB << (sq & 7)
}

#[inline(always)]
pub const fn file_bb_of(file: u8) -> Bitboard {
    FILE_A_BB << file
}

#[inline(always)]
pub const fn rank_bb(sq: Square) -> Bitboard {
    RANK_1_BB << (sq & 56)
}

#[inline(always)]
pub const fn rank_bb_of(rank: u8) -> Bitboard {
    RANK_1_BB << (rank * 8)
}

#[inline(always)]
pub const fn shift(b: Bitboard, dir: i32) -> Bitboard {
    match dir {
        NORTH => b << 8,
        SOUTH => b >> 8,
        EAST => (b & !FILE_H_BB) << 1,
        WEST => (b & !FILE_A_BB) >> 1,
        NORTH_EAST => (b & !FILE_H_BB) << 9,
        NORTH_WEST => (b & !FILE_A_BB) << 7,
        SOUTH_EAST => (b & !FILE_H_BB) >> 7,
        SOUTH_WEST => (b & !FILE_A_BB) >> 9,
        16 => b << 16,
        -16 => b >> 16,
        _ => 0,
    }
}

#[inline(always)]
pub const fn lsb(b: Bitboard) -> Square {
    debug_assert!(b != 0);
    b.trailing_zeros() as Square
}

#[inline(always)]
pub const fn msb(b: Bitboard) -> Square {
    debug_assert!(b != 0);
    (63 - b.leading_zeros()) as Square
}

#[inline(always)]
pub fn pop_lsb(b: &mut Bitboard) -> Square {
    let s = lsb(*b);
    *b &= *b - 1;
    s
}

#[inline(always)]
pub const fn popcount(b: Bitboard) -> i32 {
    b.count_ones() as i32
}

#[inline(always)]
pub const fn more_than_one(b: Bitboard) -> bool {
    b & b.wrapping_sub(1) != 0
}

#[inline(always)]
pub const fn least_significant_square_bb(b: Bitboard) -> Bitboard {
    b & b.wrapping_neg()
}

/// Squares the pawns of `c` on `b` attack (set-wise).
#[inline(always)]
pub const fn pawn_attacks_bb(c: Color, b: Bitboard) -> Bitboard {
    match c {
        Color::White => shift(b, NORTH_WEST) | shift(b, NORTH_EAST),
        Color::Black => shift(b, SOUTH_WEST) | shift(b, SOUTH_EAST),
    }
}

/// Squares on `sq`'s file in front of it from `c`'s point of view.
#[inline(always)]
pub const fn forward_ranks_bb(c: Color, sq: Square) -> Bitboard {
    match c {
        Color::White => !RANK_1_BB << (8 * (rank_of(sq) as u32)),
        Color::Black => !RANK_8_BB >> (8 * (7 - rank_of(sq) as u32)),
    }
}

#[inline(always)]
pub const fn forward_file_bb(c: Color, sq: Square) -> Bitboard {
    forward_ranks_bb(c, sq) & file_bb(sq)
}

#[inline(always)]
pub const fn adjacent_files_bb(sq: Square) -> Bitboard {
    shift(file_bb(sq), EAST) | shift(file_bb(sq), WEST)
}

/// Squares a pawn on `sq` could attack as it advances (adjacent files, ahead).
#[inline(always)]
pub const fn pawn_attack_span(c: Color, sq: Square) -> Bitboard {
    forward_ranks_bb(c, sq) & adjacent_files_bb(sq)
}

/// Squares that must be free of enemy pawns for a pawn on `sq` to be passed.
#[inline(always)]
pub const fn passed_pawn_span(c: Color, sq: Square) -> Bitboard {
    pawn_attack_span(c, sq) | forward_file_bb(c, sq)
}

/// Chebyshev distance.
#[inline(always)]
pub fn distance(a: Square, b: Square) -> u8 {
    tables().square_distance[a as usize][b as usize]
}

#[inline(always)]
pub fn file_distance(a: Square, b: Square) -> u8 {
    (file_of(a) as i8 - file_of(b) as i8).unsigned_abs()
}

#[inline(always)]
pub fn rank_distance(a: Square, b: Square) -> u8 {
    (rank_of(a) as i8 - rank_of(b) as i8).unsigned_abs()
}

#[inline(always)]
pub fn manhattan_distance(a: Square, b: Square) -> u8 {
    file_distance(a, b) + rank_distance(a, b)
}

#[derive(Clone, Copy)]
pub struct Magic {
    pub mask: Bitboard,
    pub magic: u64,
    pub shift: u32,
    pub offset: u32,
}

impl Magic {
    const EMPTY: Magic = Magic {
        mask: 0,
        magic: 0,
        shift: 0,
        offset: 0,
    };

    #[inline(always)]
    pub const fn index(&self, occupied: Bitboard) -> usize {
        (((occupied & self.mask).wrapping_mul(self.magic)) >> self.shift) as usize
            + self.offset as usize
    }
}

const ROOK_TABLE_SIZE: usize = 0x19000;
const BISHOP_TABLE_SIZE: usize = 0x1480;

pub struct Tables {
    pub pawn_attacks: [[Bitboard; 64]; 2],
    pub pseudo_attacks: [[Bitboard; 64]; 6],
    pub between_bb: [[Bitboard; 64]; 64],
    pub line_bb: [[Bitboard; 64]; 64],
    pub square_distance: [[u8; 64]; 64],
    pub rook_magics: [Magic; 64],
    pub bishop_magics: [Magic; 64],
    pub rook_table: [Bitboard; ROOK_TABLE_SIZE],
    pub bishop_table: [Bitboard; BISHOP_TABLE_SIZE],
}

impl Tables {
    const EMPTY: Tables = Tables {
        pawn_attacks: [[0; 64]; 2],
        pseudo_attacks: [[0; 64]; 6],
        between_bb: [[0; 64]; 64],
        line_bb: [[0; 64]; 64],
        square_distance: [[0; 64]; 64],
        rook_magics: [Magic::EMPTY; 64],
        bishop_magics: [Magic::EMPTY; 64],
        rook_table: [0; ROOK_TABLE_SIZE],
        bishop_table: [0; BISHOP_TABLE_SIZE],
    };
}

struct SyncCell<T>(UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}

static TABLES: SyncCell<Tables> = SyncCell(UnsafeCell::new(Tables::EMPTY));
static INIT: Once = Once::new();

#[inline(always)]
pub fn tables() -> &'static Tables {
    // Written exactly once inside `init()` (guarded by `Once`) before any reader
    // exists; afterwards the table is immutable, so shared references are sound.
    unsafe { &*TABLES.0.get() }
}

/// Relevant occupancy bit counts per square (fancy magic shift = 64 - bits).
#[rustfmt::skip]
const BISHOP_BITS: [u32; 64] = [
    6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6,
];
#[rustfmt::skip]
const ROOK_BITS: [u32; 64] = [
    12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12,
];

// Magic multipliers, indexed by bit index. The relevant-occupancy mask of bit `i` is
// a pure bit pattern (row i/8, column i%8), so these work for any rank labelling.
#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0xA180022080400230, 0x0040100040022000, 0x0080088020001002, 0x0080080280841000,
    0x4200042010460008, 0x04800A0003040080, 0x0400110082041008, 0x008000A041000880,
    0x10138001A080C010, 0x0000804008200480, 0x00010011012000C0, 0x0022004128102200,
    0x000200081201200C, 0x202A001048460004, 0x0081000100420004, 0x4000800380004500,
    0x0000208002904001, 0x0090004040026008, 0x0208808010002001, 0x2002020020704940,
    0x8048010008110005, 0x6820808004002200, 0x0A80040008023011, 0x00B1460000811044,
    0x4204400080008EA0, 0xB002400180200184, 0x2020200080100380, 0x0010080080100080,
    0x2204080080800400, 0x0000A40080360080, 0x02040604002810B1, 0x008C218600004104,
    0x8180004000402000, 0x488C402000401001, 0x4018A00080801004, 0x1230002105001008,
    0x8904800800800400, 0x0042000C42003810, 0x008408110400B012, 0x0018086182000401,
    0x2240088020C28000, 0x001001201040C004, 0x0A02008010420020, 0x0010003009010060,
    0x0004008008008014, 0x0080020004008080, 0x0282020001008080, 0x50000181204A0004,
    0x48FFFE99FECFAA00, 0x48FFFE99FECFAA00, 0x497FFFADFF9C2E00, 0x613FFFDDFFCE9200,
    0xFFFFFFE9FFE7CE00, 0xFFFFFFF5FFF3E600, 0x0010301802830400, 0x510FFFF5F63C96A0,
    0xEBFFFFB9FF9FC526, 0x61FFFEDDFEEDAEAE, 0x53BFFFEDFFDEB1A2, 0x127FFFB9FFDFB5F6,
    0x411FFFDDFFDBF4D6, 0x0801000804000603, 0x0003FFEF27EEBE74, 0x7645FFFECBFEA79E,
];
#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0xFFEDF9FD7CFCFFFF, 0xFC0962854A77F576, 0x5822022042000000, 0x2CA804A100200020,
    0x0204042200000900, 0x2002121024000002, 0xFC0A66C64A7EF576, 0x7FFDFDFCBD79FFFF,
    0xFC0846A64A34FFF6, 0xFC087A874A3CF7F6, 0x1001080204002100, 0x1810080489021800,
    0x0062040420010A00, 0x5028043004300020, 0xFC0864AE59B4FF76, 0x3C0860AF4B35FF76,
    0x73C01AF56CF4CFFB, 0x41A01CFAD64AAFFC, 0x040C0422080A0598, 0x4228020082004050,
    0x0200800400E00100, 0x020B001230021040, 0x7C0C028F5B34FF76, 0xFC0A028E5AB4DF76,
    0x0020208050A42180, 0x001004804B280200, 0x2048020024040010, 0x0102C04004010200,
    0x020408204C002010, 0x02411100020080C1, 0x102A008084042100, 0x0941030000A09846,
    0x0244100800400200, 0x4000901010080696, 0x0000280404180020, 0x0800042008240100,
    0x0220008400088020, 0x04020182000904C9, 0x0023010400020600, 0x0041040020110302,
    0xDCEFD9B54BFCC09F, 0xF95FFA765AFD602B, 0x1401210240484800, 0x0022244208010080,
    0x1105040104000210, 0x2040088800C40081, 0x43FF9A5CF4CA0C01, 0x4BFFCD8E7C587601,
    0xFC0FF2865334F576, 0xFC0BF6CE5924F576, 0x80000B0401040402, 0x0020004821880A00,
    0x8200002022440100, 0x0009431801010068, 0xC3FFB7DC36CA8C89, 0xC3FF8A54F4CA2C89,
    0xFFFFFCFCFD79EDFF, 0xFC0863FCCB147576, 0x040C000022013020, 0x2000104000420600,
    0x0400000260142410, 0x0800633408100500, 0xFC087E8E4BB2F736, 0x43FF9E4EF4CA2C89,
];

/// Destination of stepping `step` from `sq`, or `None` if it wraps off the board.
#[inline]
fn safe_destination(sq: Square, step: i32) -> Option<Square> {
    let to = sq as i32 + step;
    if !is_square_ok(to) {
        return None;
    }
    let to = to as Square;
    let fd = (file_of(sq) as i32 - file_of(to) as i32).abs();
    let rd = (rank_of(sq) as i32 - rank_of(to) as i32).abs();
    if fd.max(rd) <= 2 { Some(to) } else { None }
}

/// Sliding attacks computed square by square (used only at init).
fn sliding_attack(dirs: &[i32], sq: Square, occupied: Bitboard) -> Bitboard {
    let mut attacks = 0;
    for &d in dirs {
        let mut s = sq;
        while let Some(to) = safe_destination(s, d) {
            attacks |= sq_bb(to);
            if occupied & sq_bb(to) != 0 {
                break;
            }
            s = to;
        }
    }
    attacks
}

const ROOK_DIRS: [i32; 4] = [NORTH, SOUTH, EAST, WEST];
const BISHOP_DIRS: [i32; 4] = [NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST];

fn init_magics(t: &mut Tables, rook: bool) {
    let dirs: &[i32] = if rook { &ROOK_DIRS } else { &BISHOP_DIRS };
    let mut offset: u32 = 0;
    for sq in 0..64u8 {
        // Edges never matter for occupancy, only as destinations.
        let edges = ((RANK_1_BB | RANK_8_BB) & !rank_bb(sq)) | ((FILE_A_BB | FILE_H_BB) & !file_bb(sq));
        let mask = sliding_attack(dirs, sq, 0) & !edges;
        let bits = if rook { ROOK_BITS[sq as usize] } else { BISHOP_BITS[sq as usize] };
        debug_assert_eq!(popcount(mask) as u32, bits, "relevant bit count mismatch on {}", sq);
        let magic = Magic {
            mask,
            magic: if rook { ROOK_MAGICS[sq as usize] } else { BISHOP_MAGICS[sq as usize] },
            shift: 64 - bits,
            offset,
        };
        // Enumerate every subset of the mask (Carry-Rippler) and fill the table.
        let mut occ: Bitboard = 0;
        loop {
            let idx = magic.index(occ);
            let att = sliding_attack(dirs, sq, occ);
            let slot = if rook { &mut t.rook_table[idx] } else { &mut t.bishop_table[idx] };
            debug_assert!(*slot == 0 || *slot == att, "magic collision on square {}", sq);
            *slot = att;
            occ = occ.wrapping_sub(mask) & mask;
            if occ == 0 {
                break;
            }
        }
        if rook {
            t.rook_magics[sq as usize] = magic;
        } else {
            t.bishop_magics[sq as usize] = magic;
        }
        offset += 1 << bits;
    }
    debug_assert_eq!(offset as usize, if rook { ROOK_TABLE_SIZE } else { BISHOP_TABLE_SIZE });
}

/// Builds every table. Idempotent and thread-safe.
pub fn init() {
    INIT.call_once(|| {
        // SAFETY: the Once guarantees this closure runs exactly once, and callers of
        // `tables()` are documented to run after `init()` returns.
        let t = unsafe { &mut *TABLES.0.get() };

        for a in 0..64u8 {
            for b in 0..64u8 {
                t.square_distance[a as usize][b as usize] = file_distance(a, b).max(rank_distance(a, b));
            }
        }

        init_magics(t, true);
        init_magics(t, false);

        for sq in 0..64u8 {
            let b = sq_bb(sq);
            t.pawn_attacks[Color::White.idx()][sq as usize] = pawn_attacks_bb(Color::White, b);
            t.pawn_attacks[Color::Black.idx()][sq as usize] = pawn_attacks_bb(Color::Black, b);

            for step in [-9, -7, 7, 9, -8, 8, -1, 1] {
                if let Some(to) = safe_destination(sq, step) {
                    t.pseudo_attacks[PieceType::King.idx()][sq as usize] |= sq_bb(to);
                }
            }
            for step in [-17, -15, -10, -6, 6, 10, 15, 17] {
                if let Some(to) = safe_destination(sq, step) {
                    t.pseudo_attacks[PieceType::Knight.idx()][sq as usize] |= sq_bb(to);
                }
            }
            t.pseudo_attacks[PieceType::Bishop.idx()][sq as usize] = sliding_attack(&BISHOP_DIRS, sq, 0);
            t.pseudo_attacks[PieceType::Rook.idx()][sq as usize] = sliding_attack(&ROOK_DIRS, sq, 0);
            t.pseudo_attacks[PieceType::Queen.idx()][sq as usize] =
                t.pseudo_attacks[PieceType::Bishop.idx()][sq as usize] | t.pseudo_attacks[PieceType::Rook.idx()][sq as usize];
        }

        for s1 in 0..64u8 {
            for s2 in 0..64u8 {
                for pt in [PieceType::Bishop, PieceType::Rook] {
                    if t.pseudo_attacks[pt.idx()][s1 as usize] & sq_bb(s2) != 0 {
                        t.line_bb[s1 as usize][s2 as usize] =
                            (t.pseudo_attacks[pt.idx()][s1 as usize] & t.pseudo_attacks[pt.idx()][s2 as usize])
                                | sq_bb(s1)
                                | sq_bb(s2);
                        let dirs: &[i32] = if pt == PieceType::Rook { &ROOK_DIRS } else { &BISHOP_DIRS };
                        t.between_bb[s1 as usize][s2 as usize] =
                            sliding_attack(dirs, s1, sq_bb(s2)) & sliding_attack(dirs, s2, sq_bb(s1));
                    }
                }
                // Unaligned squares: the "between" set is just the destination, so
                // `between_bb(king, checker) & to` also accepts capturing the checker.
                t.between_bb[s1 as usize][s2 as usize] |= sq_bb(s2);
            }
        }
    });
}

#[inline(always)]
pub fn pawn_attacks(c: Color, sq: Square) -> Bitboard {
    tables().pawn_attacks[c.idx()][sq as usize]
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    tables().pseudo_attacks[PieceType::Knight.idx()][sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    tables().pseudo_attacks[PieceType::King.idx()][sq as usize]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let t = tables();
    let m = &t.bishop_magics[sq as usize];
    // The magic index is bounded by construction: (occ & mask) * magic >> shift < 2^bits.
    unsafe { *t.bishop_table.get_unchecked(m.index(occupied)) }
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let t = tables();
    let m = &t.rook_magics[sq as usize];
    unsafe { *t.rook_table.get_unchecked(m.index(occupied)) }
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
}

/// Attacks of a non-pawn piece type from `sq` given the occupancy.
#[inline(always)]
pub fn attacks_bb(pt: PieceType, sq: Square, occupied: Bitboard) -> Bitboard {
    match pt {
        PieceType::Bishop => bishop_attacks(sq, occupied),
        PieceType::Rook => rook_attacks(sq, occupied),
        PieceType::Queen => queen_attacks(sq, occupied),
        _ => tables().pseudo_attacks[pt.idx()][sq as usize],
    }
}

/// Attacks of a piece type on an empty board.
#[inline(always)]
pub fn pseudo_attacks(pt: PieceType, sq: Square) -> Bitboard {
    tables().pseudo_attacks[pt.idx()][sq as usize]
}

/// Squares strictly between `a` and `b` plus `b` itself; just `b` when unaligned.
#[inline(always)]
pub fn between_bb(a: Square, b: Square) -> Bitboard {
    tables().between_bb[a as usize][b as usize]
}

/// Full line through `a` and `b` (0 if not aligned).
#[inline(always)]
pub fn line_bb(a: Square, b: Square) -> Bitboard {
    tables().line_bb[a as usize][b as usize]
}

#[inline(always)]
pub fn aligned(a: Square, b: Square, c: Square) -> bool {
    line_bb(a, b) & sq_bb(c) != 0
}

pub fn pretty(b: Bitboard) -> String {
    let mut s = String::from("+---+---+---+---+---+---+---+---+\n");
    for r in (0..8).rev() {
        for f in 0..8 {
            s.push_str(if b & sq_bb(make_square(f, r)) != 0 { "| X " } else { "|   " });
        }
        s.push_str(&format!("| {}\n+---+---+---+---+---+---+---+---+\n", r + 1));
    }
    s.push_str("  a   b   c   d   e   f   g   h\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magics_match_on_the_fly() {
        init();
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..20000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let occ = rng & (rng.rotate_left(23));
            let sq = (rng >> 58) as Square;
            assert_eq!(rook_attacks(sq, occ), sliding_attack(&ROOK_DIRS, sq, occ));
            assert_eq!(bishop_attacks(sq, occ), sliding_attack(&BISHOP_DIRS, sq, occ));
        }
    }

    #[test]
    fn between_and_line() {
        init();
        use crate::types::squares::*;
        assert_eq!(between_bb(A1, A8), (FILE_A_BB & !sq_bb(A1)));
        assert_eq!(between_bb(A1, C2), sq_bb(C2)); // unaligned
        assert!(aligned(A1, H8, D4));
        assert!(!aligned(A1, H8, D5));
        assert_eq!(knight_attacks(A1), sq_bb(B1 + 16) | sq_bb(C1 + 8));
    }
}
