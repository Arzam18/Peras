//! NNUE evaluation.
//!
//! Architecture, per perspective: `768 x 10 king buckets` piece-square inputs, `59808`
//! threat inputs (which piece attacks which, from where) and `4560` pawn-pair inputs, all
//! horizontally mirrored onto the king's half of the board, feeding a `1024`-wide
//! accumulator; `(72048 -> 1024)x2 -> 8 output buckets` with SCReLU activation, trained with
//! bullet. Piece-square weights are `i16` and the threat and pawn-pair weights `i8`, both
//! quantised by 255; the output layer is `i16` quantised by 64. The net is embedded in the
//! binary.
//!
//! While a move is made, the board primitives record every threat it creates or destroys
//! (a piece's own attacks, attacks on it, and the lines it opens or closes for sliders
//! behind it). Each ply also keeps a snapshot of the board and
//! pawns. The accumulators are brought up to date lazily on the first evaluation that
//! needs them: piece-square and pawn-pair changes come from diffing snapshots, threat
//! changes from the recorded list, and the weight rows are applied to the accumulator a
//! register-resident tile at a time. When the king crosses into another bucket (or over the
//! mirror line) the accumulator is instead rebuilt from a cache holding the last accumulator
//! seen in that bucket, diffing full attack sets against the snapshot cached with it.

use crate::bitboard::*;
use crate::types::*;

/// Cycle counters per stage, compiled in with the `nnue-profile` feature; `bench` prints them.
#[cfg(feature = "nnue-profile")]
pub mod profile {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static SNAPSHOT: AtomicU64 = AtomicU64::new(0);
    pub static THREATS: AtomicU64 = AtomicU64::new(0);
    pub static DELTA: AtomicU64 = AtomicU64::new(0);
    pub static UPDATE: AtomicU64 = AtomicU64::new(0);
    pub static DOT: AtomicU64 = AtomicU64::new(0);
    pub static PUSHES: AtomicU64 = AtomicU64::new(0);
    pub static EVALS: AtomicU64 = AtomicU64::new(0);
    pub static REFRESHES: AtomicU64 = AtomicU64::new(0);
    pub static AUX_ROWS: AtomicU64 = AtomicU64::new(0);
    pub static AUX_ROWS_REFRESH: AtomicU64 = AtomicU64::new(0);
    pub static PSQT_ROWS_REFRESH: AtomicU64 = AtomicU64::new(0);

    #[inline(always)]
    pub fn now() -> u64 {
        unsafe { std::arch::x86_64::_rdtsc() }
    }

    #[inline(always)]
    pub fn add(c: &AtomicU64, start: u64) {
        c.fetch_add(now() - start, Relaxed);
    }

    pub fn report() -> String {
        let pushes = PUSHES.load(Relaxed).max(1);
        let evals = EVALS.load(Relaxed).max(1);
        let cyc = |c: &AtomicU64| c.load(Relaxed) as f64;
        format!(
            "nnue profile: pushes {} evals {} refreshes {} aux rows/eval {:.1} (per refresh {:.1} aux, {:.1} psqt)
  per push: snapshot {:.0} cyc
  per eval: threats {:.0}, delta {:.0}, update {:.0}, dot {:.0} cyc",
            pushes,
            evals,
            REFRESHES.load(Relaxed),
            cyc(&AUX_ROWS) / evals as f64,
            cyc(&AUX_ROWS_REFRESH) / REFRESHES.load(Relaxed).max(1) as f64,
            cyc(&PSQT_ROWS_REFRESH) / REFRESHES.load(Relaxed).max(1) as f64,
            cyc(&SNAPSHOT) / pushes as f64,
            cyc(&THREATS) / evals as f64,
            cyc(&DELTA) / evals as f64,
            cyc(&UPDATE) / evals as f64,
            cyc(&DOT) / evals as f64,
        )
    }
}

macro_rules! timed {
    ($ctr:ident, $e:expr) => {{
        #[cfg(feature = "nnue-profile")]
        let __t = profile::now();
        let __r = $e;
        #[cfg(feature = "nnue-profile")]
        profile::add(&profile::$ctr, __t);
        __r
    }};
}

macro_rules! count {
    ($ctr:ident, $n:expr) => {{
        #[cfg(feature = "nnue-profile")]
        profile::$ctr.fetch_add($n as u64, std::sync::atomic::Ordering::Relaxed);
    }};
}

pub const HL: usize = 1024;
/// Half the transformer, after the two halves are multiplied together.
pub const PAIRED: usize = HL / 2;
const INPUT_BUCKETS: usize = 16;
const OUTPUT_BUCKETS: usize = 8;
const L2: usize = 32;
const L3: usize = 32;
const PSQT_FEATURES: usize = 768 * INPUT_BUCKETS;
const PP_FEATURES: usize = 96 * 95 / 2;
const THREATS_PER_SIDE: usize = 29904;
const AUX_FEATURES: usize = PP_FEATURES + 2 * THREATS_PER_SIDE;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

/// The transformer's outputs are multiplied in pairs and taken down by this much. Nine bits
/// caps them at 127, so a pair of them fits an unsigned byte each and the first layer can
/// use `maddubs` without saturating; it also lands the layer's output on exactly the scale
/// its bias was stored at. The weights themselves were rescaled with a shift of 8 at save
/// time, which is a separate constant and must not be changed here.
const FT_SHIFT: u32 = 9;
/// Scale of the first affine layer's output, and of its stored bias.
const S1: i32 = 128 * 128;
/// Scale of the second affine layer's output, and of the third's.
const S2: i64 = (QB as i64).pow(3);
const S3: i64 = (QB as i64).pow(4);

/// King bucket by square (own perspective, a1 = 0), given for the a-d files only; the
/// e-h files mirror onto them.
#[rustfmt::skip]
const BUCKET_LAYOUT: [u8; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

const fn expand_buckets() -> [u8; 64] {
    let mirror = [0, 1, 2, 3, 3, 2, 1, 0];
    let mut out = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        out[sq] = BUCKET_LAYOUT[(sq / 8) * 4 + mirror[sq % 8]];
        sq += 1;
    }
    out
}

static KING_BUCKET: [u8; 64] = expand_buckets();

// ---------------------------------------------------------------------------------------
// Threat feature tables.
//
// A threat feature is (attacker type, attacker square, attacked square, attacked piece).
// Attacked squares are numbered within the attacker's empty-board attack set, and each
// attacker type only has features for some target types: pawns for knights and rooks,
// bishops and rooks for pawns, knights, bishops and rooks, knights and queens for all
// but kings. A piece attacking a piece of its own type is only counted from the higher
// square, since the relation is symmetric. Kings never attack or are attacked here.

const NO_TARGET: u8 = u8::MAX;

struct PieceThreats {
    attacks: [Bitboard; 64],
    index: [u32; 64],
    count: u32,
    targets: [u8; 12],
    offset: u32,
}

const fn pseudo_attacks_const(pt: usize, sq: usize) -> Bitboard {
    let (f, r) = ((sq % 8) as i32, (sq / 8) as i32);
    let dirs: &[(i32, i32)] = match pt {
        1 => &[(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)],
        2 => &[(1, 1), (1, -1), (-1, -1), (-1, 1)],
        3 => &[(1, 0), (0, 1), (-1, 0), (0, -1)],
        _ => &[(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)],
    };
    let slider = pt != 1;
    let mut bb = 0u64;
    let mut i = 0;
    while i < dirs.len() {
        let (df, dr) = dirs[i];
        let (mut x, mut y) = (f + df, r + dr);
        while x >= 0 && x < 8 && y >= 0 && y < 8 {
            bb |= 1u64 << (y * 8 + x);
            if !slider {
                break;
            }
            x += df;
            y += dr;
        }
        i += 1;
    }
    bb
}

/// Target classes 0..6 are own pieces by type, 6..12 the opponent's.
const fn make_targets(valid: &[usize]) -> [u8; 12] {
    let mut t = [NO_TARGET; 12];
    let mut i = 0;
    while i < valid.len() {
        t[valid[i]] = i as u8;
        t[valid[i] + 6] = (i + valid.len()) as u8;
        i += 1;
    }
    t
}

const fn piece_threats(pt: usize, valid: &[usize], offset: u32) -> PieceThreats {
    let mut attacks = [0u64; 64];
    let mut index = [0u32; 64];
    let mut count = 0u32;
    let mut sq = 0;
    while sq < 64 {
        attacks[sq] = pseudo_attacks_const(pt, sq);
        index[sq] = count;
        count += attacks[sq].count_ones();
        sq += 1;
    }
    PieceThreats { attacks, index, count, targets: make_targets(valid), offset }
}

const PAWN_THREATS: usize = 4 * 84;
static PAWN_TARGETS: [u8; 12] = make_targets(&[1, 3]);
static KNIGHT_THREATS: PieceThreats = piece_threats(1, &[0, 1, 2, 3, 4], PAWN_THREATS as u32);
static BISHOP_THREATS: PieceThreats = piece_threats(2, &[0, 1, 2, 3], KNIGHT_THREATS.offset + 10 * KNIGHT_THREATS.count);
static ROOK_THREATS: PieceThreats = piece_threats(3, &[0, 1, 2, 3], BISHOP_THREATS.offset + 8 * BISHOP_THREATS.count);
static QUEEN_THREATS: PieceThreats = piece_threats(4, &[0, 1, 2, 3, 4], ROOK_THREATS.offset + 8 * ROOK_THREATS.count);
const _: () = assert!(QUEEN_THREATS.offset + 10 * QUEEN_THREATS.count == THREATS_PER_SIDE as u32);

/// Index of `d` within the attacker's empty-board attack set from `s`, plus that square's
/// cumulative offset, for each attacker type. Precomputing this is what removes the mask and
/// popcount from the hot path; entries for squares the attacker cannot reach are never read.
const fn build_sq_idx() -> [[[i16; 64]; 64]; 6] {
    let tables = all_piece_threats();
    let mut out = [[[0i16; 64]; 64]; 6];
    let mut pt = 0;
    while pt < 6 {
        let mut s = 0;
        while s < 64 {
            let mut d = 0;
            while d < 64 {
                out[pt][s][d] = if pt == 0 {
                    // Pawn features are laid out by rank, file and capture direction. The
                    // offset is kept signed and unguarded: a pawn on relative rank 0 (which
                    // occurs transiently for the opponent's perspective while a promotion is
                    // being made) underflowed in the original arithmetic, and the resulting
                    // index is reproduced here exactly. It is harmless because the same
                    // index is both added and subtracted within the move, so it cancels.
                    let rank = (s / 8) as i32;
                    let diff = if d > s { d - s } else { s - d };
                    let id = if diff == if d > s { 7 } else { 9 } { 0 } else { 1 };
                    let attack = 2 * (s % 8) as i32 + id - 1;
                    ((rank - 1) * 14 + attack) as i16
                } else if pt < 5 {
                    let t = &tables[pt - 1];
                    let below = if d == 63 { u64::MAX >> 1 } else { (1u64 << d) - 1 };
                    (t.index[s] + (t.attacks[s] & below).count_ones()) as i16
                } else {
                    0
                };
                d += 1;
            }
            s += 1;
        }
        pt += 1;
    }
    out
}

/// The four non-pawn attacker tables, rebuilt once in const context so the lookup tables
/// below do not have to read the statics.
const fn all_piece_threats() -> [PieceThreats; 4] {
    let knight = piece_threats(1, &[0, 1, 2, 3, 4], PAWN_THREATS as u32);
    let bishop = piece_threats(2, &[0, 1, 2, 3], knight.offset + 10 * knight.count);
    let rook = piece_threats(3, &[0, 1, 2, 3], bishop.offset + 8 * bishop.count);
    let queen = piece_threats(4, &[0, 1, 2, 3, 4], rook.offset + 8 * rook.count);
    [knight, bishop, rook, queen]
}

/// Everything about a (attacker, target, direction) triple that does not depend on the
/// squares: the side block, the attacker's block, and the target's slot within it.
/// `u32::MAX` marks a pair the network has no input for. Pieces are perspective-relative,
/// so 0..6 are the perspective's own and 6..12 the opponent's.
const fn build_lut1() -> [[[u32; 2]; 12]; 12] {
    let tables = all_piece_threats();
    let mut out = [[[u32::MAX; 2]; 12]; 12];
    let mut ra = 0;
    while ra < 12 {
        let pt = ra % 6;
        let side = (ra / 6) as u32;
        let mut rt = 0;
        while rt < 12 {
            let mut dir = 0;
            while dir < 2 {
                // dir == 1 means the target square is above the attacker's.
                let excluded_same_type = dir == 1 && rt % 6 == pt;
                let value = if pt == 0 {
                    let m = make_targets(&[1, 3])[rt];
                    if m == NO_TARGET { u32::MAX } else { m as u32 * 84 }
                } else if pt < 5 {
                    let t = &tables[pt - 1];
                    let m = t.targets[rt];
                    if m == NO_TARGET || excluded_same_type {
                        u32::MAX
                    } else {
                        t.offset + m as u32 * t.count
                    }
                } else {
                    u32::MAX
                };
                out[ra][rt][dir] = if value == u32::MAX {
                    u32::MAX
                } else {
                    PP_FEATURES as u32 + THREATS_PER_SIDE as u32 * side + value
                };
                dir += 1;
            }
            rt += 1;
        }
        ra += 1;
    }
    out
}

static SQ_IDX: [[[i16; 64]; 64]; 6] = build_sq_idx();
static THREAT_LUT: [[[u32; 2]; 12]; 12] = build_lut1();

/// `pc` as seen from `p`: the perspective's own pieces occupy 0..6 and the opponent's 6..12.
#[inline(always)]
fn relative_piece(pc: Piece, p: Color) -> usize {
    let i = pc.idx();
    if p == Color::White { i } else { (i + 6) % 12 }
}

/// Threat feature index from `p`'s perspective, or `None` for pairs the net has no input for.
#[inline(always)]
fn threat_feature(attacker: Piece, src: Square, dest: Square, target: Piece, p: Color, flip: u8) -> Option<usize> {
    let s = (relative_square(p, src) ^ flip) as usize;
    let d = (relative_square(p, dest) ^ flip) as usize;
    let ra = relative_piece(attacker, p);
    let rt = relative_piece(target, p);
    threat_index(ra, rt, s, d)
}

/// The table lookup itself: perspective-relative attacker and target, oriented squares.
#[inline(always)]
fn threat_index(ra: usize, rt: usize, s: usize, d: usize) -> Option<usize> {
    let base = THREAT_LUT[ra][rt][(d > s) as usize];
    if base == u32::MAX {
        return None;
    }
    Some((base as i64 + SQ_IDX[ra % 6][s][d] as i64) as usize)
}

/// Pawn-pair feature index from `p`'s perspective for pawns `a` and `b` (any colours).
#[inline]
fn pp_feature(ca: Color, sa: Square, cb: Color, sb: Square, p: Color, flip: u8) -> usize {
    let id = |c: Color, s: Square| ((c != p) as usize) * 48 + (relative_square(p, s) ^ flip) as usize - 8;
    let (a, b) = (id(ca, sa), id(cb, sb));
    let (lo, hi) = (a.min(b), a.max(b));
    debug_assert!(lo != hi);
    hi * (hi - 1) / 2 + lo
}

/// Pawns pair with pawns on the same or an adjacent file.
#[inline(always)]
fn pawn_band(sq: Square) -> Bitboard {
    file_bb(sq) | adjacent_files_bb(sq)
}

/// Squares on the line from `a` through `b` to the edge, from `b` onward but also the
/// squares strictly between `a` and `b`; empty when the two are not aligned.
const fn ray_pass_table() -> [[Bitboard; 64]; 64] {
    let mut t = [[0u64; 64]; 64];
    let mut a = 0;
    while a < 64 {
        let mut b = 0;
        while b < 64 {
            if a != b {
                let (df, dr) = ((b % 8) as i32 - (a % 8) as i32, (b / 8) as i32 - (a / 8) as i32);
                if df == 0 || dr == 0 || df.abs() == dr.abs() {
                    let (sf, sr) = (df.signum(), dr.signum());
                    let (mut f, mut r) = ((a % 8) as i32 + sf, (a / 8) as i32 + sr);
                    while f >= 0 && f < 8 && r >= 0 && r < 8 {
                        t[a][b] |= 1u64 << (r * 8 + f);
                        f += sf;
                        r += sr;
                    }
                }
            }
            b += 1;
        }
        a += 1;
    }
    t
}

static RAY_PASS: [[Bitboard; 64]; 64] = ray_pass_table();

/// One threat that appeared or vanished during a move: `attacker` on `from` attacking
/// `target` on `to`. Packed so a ply's list copies cheaply.
#[derive(Clone, Copy)]
pub struct DirtyThreat(u32);

impl DirtyThreat {
    #[inline(always)]
    fn new(add: bool, attacker: Piece, target: Piece, from: Square, to: Square) -> DirtyThreat {
        DirtyThreat(
            (add as u32) << 31 | (attacker.0 as u32) << 24 | (target.0 as u32) << 16 | (from as u32) << 8 | to as u32,
        )
    }
    #[inline(always)]
    fn add(self) -> bool {
        self.0 >> 31 != 0
    }
    #[inline(always)]
    fn attacker(self) -> Piece {
        Piece((self.0 >> 24 & 0x7f) as u8)
    }
    #[inline(always)]
    fn target(self) -> Piece {
        Piece((self.0 >> 16 & 0xff) as u8)
    }
    #[inline(always)]
    fn from(self) -> Square {
        (self.0 >> 8 & 0xff) as Square
    }
    #[inline(always)]
    fn to(self) -> Square {
        (self.0 & 0xff) as Square
    }
}

pub const MAX_DIRTY_THREATS: usize = 96;

/// The threats a move changed, recorded by the board primitives as they run.
#[derive(Clone, Copy)]
pub struct DirtyThreats {
    list: [DirtyThreat; MAX_DIRTY_THREATS],
    n: u8,
    /// More changed than fit: the ply falls back to diffing snapshots.
    overflow: bool,
}

impl DirtyThreats {
    const EMPTY: DirtyThreats = DirtyThreats { list: [DirtyThreat(0); MAX_DIRTY_THREATS], n: 0, overflow: false };

    #[inline(always)]
    pub fn clear(&mut self) {
        self.n = 0;
        self.overflow = false;
    }

    #[inline(always)]
    pub fn push(&mut self, add: bool, attacker: Piece, target: Piece, from: Square, to: Square) {
        if (self.n as usize) < MAX_DIRTY_THREATS {
            self.list[self.n as usize] = DirtyThreat::new(add, attacker, target, from, to);
            self.n += 1;
        } else {
            self.overflow = true;
        }
    }

    fn as_slice(&self) -> &[DirtyThreat] {
        &self.list[..self.n as usize]
    }
}

/// Whether `slider` has a threat input for attacking `target`: queens are only targets
/// of queens.
#[inline(always)]
pub fn can_slider_threat(target: Piece, slider: Piece) -> bool {
    target.piece_type() != PieceType::Queen || slider.piece_type() == PieceType::Queen
}

/// Squares from `a` (exclusive) through `b` to the edge of the board.
#[inline(always)]
pub fn ray_pass(a: Square, b: Square) -> Bitboard {
    RAY_PASS[a as usize][b as usize]
}

// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    v: [i16; HL],
}

/// The head's weights are stored input-major: for one input, every bucket's outputs sit
/// next to each other, so a bucket's row is a contiguous slice and the sparse first layer
/// can accumulate a whole bucket per non-zero input.
#[repr(C)]
struct Network {
    psqt_weights: [Accumulator; PSQT_FEATURES],
    aux_weights: [[i8; HL]; AUX_FEATURES],
    ft_bias: Accumulator,
    l1_weights: [[i8; OUTPUT_BUCKETS * L2]; HL],
    l1_bias: [i32; OUTPUT_BUCKETS * L2],
    l2_weights: [[i32; OUTPUT_BUCKETS * L3]; L2 * 2],
    l2_bias: [i32; OUTPUT_BUCKETS * L3],
    l3_weights: [[i32; OUTPUT_BUCKETS]; L3],
    l3_bias: [i32; OUTPUT_BUCKETS],
}

/// The network file is padded to a 64-byte multiple, which is exactly the struct's size.
static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("PERAS_NET"))) };

/// The layer stack's weights, dequantised once. Past the first affine the tensors are tiny
/// (~67 KB) but the arithmetic is awkward in integers: the scales need three divisions and
/// the products do not fit 32 bits, which leaves no vector form on AVX2. In floats it is
/// plain multiply-add, which is what every engine shipping a head this deep does.
struct StackF {
    l1_bias: [f32; OUTPUT_BUCKETS * L2],
    l2_weights: [[f32; OUTPUT_BUCKETS * L3]; L2 * 2],
    l2_bias: [f32; OUTPUT_BUCKETS * L3],
    l3_weights: [[f32; OUTPUT_BUCKETS]; L3],
    l3_bias: [f32; OUTPUT_BUCKETS],
}

/// Inputs consumed per iteration of the first affine layer.
const CHUNK4: usize = 4;
const CHUNKS: usize = HL / CHUNK4;

/// The first layer's weights, reordered so one iteration handles four inputs at once.
///
/// Stored as `[bucket][chunk]` of 128 bytes: for each of a bucket's 32 outputs, that
/// output's four weights sit together. `maddubs` against a broadcast of the four input
/// bytes then yields two partial sums per output, and `madd` folds them into the full
/// four-input dot product. The whole chunk is one contiguous read, which is why this shape
/// needs no prefetching.
struct L1Chunked {
    w: Box<[[i8; L2 * CHUNK4]; OUTPUT_BUCKETS * CHUNKS]>,
}

static L1_CHUNKED: std::sync::OnceLock<L1Chunked> = std::sync::OnceLock::new();

fn l1_chunked() -> &'static L1Chunked {
    L1_CHUNKED.get_or_init(|| {
        let mut w = vec![[0i8; L2 * CHUNK4]; OUTPUT_BUCKETS * CHUNKS].into_boxed_slice();
        for bucket in 0..OUTPUT_BUCKETS {
            for c in 0..CHUNKS {
                let dst = &mut w[bucket * CHUNKS + c];
                for o in 0..L2 {
                    for k in 0..CHUNK4 {
                        dst[o * CHUNK4 + k] = NET.l1_weights[c * CHUNK4 + k][bucket * L2 + o];
                    }
                }
            }
        }
        L1Chunked { w: w.try_into().unwrap() }
    })
}

static STACK_F: std::sync::OnceLock<Box<StackF>> = std::sync::OnceLock::new();

fn stack_f() -> &'static StackF {
    STACK_F.get_or_init(|| {
        let mut s = Box::new(StackF {
            l1_bias: [0.0; OUTPUT_BUCKETS * L2],
            l2_weights: [[0.0; OUTPUT_BUCKETS * L3]; L2 * 2],
            l2_bias: [0.0; OUTPUT_BUCKETS * L3],
            l3_weights: [[0.0; OUTPUT_BUCKETS]; L3],
            l3_bias: [0.0; OUTPUT_BUCKETS],
        });
        let qb = QB as f32;
        for i in 0..OUTPUT_BUCKETS * L2 {
            s.l1_bias[i] = NET.l1_bias[i] as f32 / S1 as f32;
        }
        for i in 0..L2 * 2 {
            for o in 0..OUTPUT_BUCKETS * L3 {
                s.l2_weights[i][o] = NET.l2_weights[i][o] as f32 / qb;
            }
        }
        for o in 0..OUTPUT_BUCKETS * L3 {
            s.l2_bias[o] = NET.l2_bias[o] as f32 / S2 as f32;
        }
        for i in 0..L3 {
            for o in 0..OUTPUT_BUCKETS {
                s.l3_weights[i][o] = NET.l3_weights[i][o] as f32 / qb;
            }
        }
        for o in 0..OUTPUT_BUCKETS {
            s.l3_bias[o] = NET.l3_bias[o] as f32 / S3 as f32;
        }
        s
    })
}

/// Multiplies each perspective's two halves together, then runs the bucket's layer stack.
/// Returns centipawns from the side to move's point of view.
///
/// Most of the transformer's outputs are zero once clipped, so the first layer only visits
/// the ones that are not. The last two layers accumulate in 64 bits: the third reaches about
/// 2.1e9, which an `i32` cannot hold.
fn head(us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
    #[cfg(target_feature = "avx2")]
    {
        unsafe { simd::head_avx2(us, them, bucket) }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        head_scalar(us, them, bucket)
    }
}

/// Reference implementation of [`head`]. The vectorised path must agree with it exactly.
#[allow(dead_code)]
fn head_scalar(us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
    let mut x = [0i32; HL];
    for (h, acc) in [us, them].into_iter().enumerate() {
        for j in 0..PAIRED {
            let lo = i32::from(acc.v[j].clamp(0, QA as i16));
            let hi = i32::from(acc.v[j + PAIRED].clamp(0, QA as i16));
            x[h * PAIRED + j] = (lo * hi) >> FT_SHIFT;
        }
    }

    let b1 = bucket * L2;
    let mut a1 = [0i32; L2];
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0 {
            continue;
        }
        let row = &NET.l1_weights[i][b1..b1 + L2];
        for (a, &w) in a1.iter_mut().zip(row) {
            *a += xi * i32::from(w);
        }
    }

    stack(&a1, bucket)
}

/// Second affine layer's weight sums, before its bias.
#[inline]
fn l2_accumulate(h2: &[f32; L2 * 2], b2: usize) -> [f32; L3] {
    #[cfg(target_feature = "avx2")]
    {
        unsafe { simd::l2_avx2(h2, b2) }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        l2_scalar(h2, b2)
    }
}

/// Reference for [`l2_accumulate`]; the vectorised path must agree with it closely.
#[allow(dead_code)]
fn l2_scalar(h2: &[f32; L2 * 2], b2: usize) -> [f32; L3] {
    let s = stack_f();
    let mut a2 = [0f32; L3];
    for (i, &h) in h2.iter().enumerate() {
        let row = &s.l2_weights[i][b2..b2 + L3];
        for (a, &w) in a2.iter_mut().zip(row) {
            *a += h * w;
        }
    }
    a2
}

/// The two layers after the first affine, shared by both code paths. `a1` is the first
/// layer's weight sum, before its bias.
fn stack(a1: &[i32; L2], bucket: usize) -> i32 {
    stack_flt(a1, bucket)
}

/// Integer layer stack. Kept alongside [`stack_flt`] so the two can be timed against each
/// other in one process; this machine's wall clock cannot resolve the difference between
/// separate runs.
#[allow(dead_code)]
fn stack_int(a1: &[i32; L2], bucket: usize) -> i32 {
    let b1 = bucket * L2;
    let mut h2 = [0i64; L2 * 2];
    for k in 0..L2 {
        let v = a1[k] + NET.l1_bias[b1 + k];
        h2[k] = i64::from(v.clamp(0, S1));
        let c = i64::from(v.clamp(-S1, S1));
        h2[L2 + k] = (c * c / i64::from(S1)).min(i64::from(S1));
    }

    let b2 = bucket * L3;
    let mut a2 = [0i64; L3];
    for (i, &h) in h2.iter().enumerate() {
        let row = &NET.l2_weights[i][b2..b2 + L3];
        for (a, &w) in a2.iter_mut().zip(row) {
            *a += h * i64::from(w);
        }
    }

    let mut h3 = [0i64; L3];
    for k in 0..L3 {
        let v = a2[k] * i64::from(QB) * i64::from(QB) / i64::from(S1) + i64::from(NET.l2_bias[b2 + k]);
        h3[k] = v.clamp(0, S2);
    }

    let mut out = i64::from(NET.l3_bias[bucket]);
    for (i, &h) in h3.iter().enumerate() {
        out += h * i64::from(NET.l3_weights[i][bucket]);
    }
    (out * i64::from(SCALE) / S3) as i32
}

fn stack_flt(a1: &[i32; L2], bucket: usize) -> i32 {
    let s = stack_f();
    let b1 = bucket * L2;
    let mut h2 = [0f32; L2 * 2];
    for k in 0..L2 {
        let v = a1[k] as f32 / S1 as f32 + s.l1_bias[b1 + k];
        h2[k] = v.clamp(0.0, 1.0);
        // The square is taken before clipping, so a negative output still contributes.
        h2[L2 + k] = (v * v).min(1.0);
    }

    let b2 = bucket * L3;
    let mut a2 = l2_accumulate(&h2, b2);

    for (k, v) in a2.iter_mut().enumerate() {
        *v = (*v + s.l2_bias[b2 + k]).clamp(0.0, 1.0);
    }

    let mut out = s.l3_bias[bucket];
    for (i, &h) in a2.iter().enumerate() {
        out += h * s.l3_weights[i][bucket];
    }
    (out * SCALE as f32) as i32
}

/// What the network sees of a position: the board, every non-king piece's attacks on
/// non-king pieces, and where the pawns and kings are.
#[derive(Clone, Copy)]
pub struct Snapshot {
    board: [Piece; 64],
    /// Attack sets, filled in by `ensure_threats` the first time a diff needs them.
    threats: [Bitboard; 64],
    threats_ready: bool,
    /// All pieces but the kings: the possible attackers and targets.
    attackers: Bitboard,
    occ: Bitboard,
    pawns: [Bitboard; 2],
    kings: [Square; 2],
    by_type: [Bitboard; 6],
    by_color: [Bitboard; 2],
}

impl Snapshot {
    pub const EMPTY: Snapshot = Snapshot {
        board: [Piece::NONE; 64],
        threats: [0; 64],
        threats_ready: true,
        attackers: 0,
        occ: 0,
        pawns: [0; 2],
        kings: [SQ_NONE; 2],
        by_type: [0; 6],
        by_color: [0; 2],
    };

    #[inline]
    pub fn build(board: &[Piece; 64], by_type: &[Bitboard; 6], by_color: &[Bitboard; 2], kings: [Square; 2]) -> Snapshot {
        count!(PUSHES, 1);
        timed!(SNAPSHOT, Self::build_inner(board, by_type, by_color, kings))
    }

    #[inline]
    fn build_inner(board: &[Piece; 64], by_type: &[Bitboard; 6], by_color: &[Bitboard; 2], kings: [Square; 2]) -> Snapshot {
        let occ = by_color[0] | by_color[1];
        let pawns = [by_type[0] & by_color[0], by_type[0] & by_color[1]];
        let mut king_bb = 0;
        for k in kings {
            if k != SQ_NONE {
                king_bb |= sq_bb(k);
            }
        }
        Snapshot {
            board: *board,
            threats: [0; 64],
            threats_ready: false,
            attackers: occ & !king_bb,
            occ,
            pawns,
            kings,
            by_type: *by_type,
            by_color: *by_color,
        }
    }

    /// Non-king pieces attacking `t`.
    #[inline]
    fn attackers_of(&self, t: Square) -> Bitboard {
        let q = self.by_type[PieceType::Queen.idx()];
        let p = self.by_type[PieceType::Pawn.idx()];
        ((rook_attacks(t, self.occ) & (self.by_type[PieceType::Rook.idx()] | q))
            | (bishop_attacks(t, self.occ) & (self.by_type[PieceType::Bishop.idx()] | q))
            | (knight_attacks(t) & self.by_type[PieceType::Knight.idx()])
            | (pawn_attacks(Color::Black, t) & p & self.by_color[0])
            | (pawn_attacks(Color::White, t) & p & self.by_color[1]))
            & self.attackers
    }

    fn ensure_threats(&mut self) {
        if self.threats_ready {
            return;
        }
        timed!(THREATS, self.compute_threats());
    }

    fn compute_threats(&mut self) {
        let mut b = self.attackers;
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let pc = self.board[sq as usize];
            let att = match pc.piece_type() {
                PieceType::Pawn => pawn_attacks(pc.color(), sq),
                pt => attacks_bb(pt, sq, self.occ),
            };
            self.threats[sq as usize] = att & self.attackers;
        }
        self.threats_ready = true;
    }
}

/// Bitboard of the squares whose piece differs between snapshots.
#[inline]
fn board_diff(old: &Snapshot, new: &Snapshot) -> Bitboard {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let mut board_changed = 0u64;
        for half in 0..2 {
            let a = _mm256_loadu_si256(old.board.as_ptr().add(32 * half) as *const __m256i);
            let b = _mm256_loadu_si256(new.board.as_ptr().add(32 * half) as *const __m256i);
            let same = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) as u32;
            board_changed |= u64::from(!same) << (32 * half);
        }
        board_changed
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let mut board_changed = 0u64;
        for sq in 0..64 {
            if old.board[sq] != new.board[sq] {
                board_changed |= 1 << sq;
            }
        }
        board_changed
    }
}

/// Bitboards of the squares whose piece, and whose attack set, differ between snapshots.
#[inline]
fn snapshot_diff(old: &Snapshot, new: &Snapshot) -> (Bitboard, Bitboard) {
    debug_assert!(old.threats_ready && new.threats_ready);
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let mut board_changed = 0u64;
        for half in 0..2 {
            let a = _mm256_loadu_si256(old.board.as_ptr().add(32 * half) as *const __m256i);
            let b = _mm256_loadu_si256(new.board.as_ptr().add(32 * half) as *const __m256i);
            let same = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) as u32;
            board_changed |= u64::from(!same) << (32 * half);
        }
        let mut map_changed = 0u64;
        for q in 0..16 {
            let a = _mm256_loadu_si256(old.threats.as_ptr().add(4 * q) as *const __m256i);
            let b = _mm256_loadu_si256(new.threats.as_ptr().add(4 * q) as *const __m256i);
            let same = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, b))) as u64;
            map_changed |= (!same & 0xf) << (4 * q);
        }
        (board_changed, map_changed)
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let mut board_changed = 0u64;
        let mut map_changed = 0u64;
        for sq in 0..64 {
            if old.board[sq] != new.board[sq] {
                board_changed |= 1 << sq;
            }
            if old.threats[sq] != new.threats[sq] {
                map_changed |= 1 << sq;
            }
        }
        (board_changed, map_changed)
    }
}

const MAX_PSQT_DELTA: usize = 32;
const MAX_AUX_DELTA: usize = 512;

/// Feature changes for one perspective.
struct Lists {
    psqt_add: [usize; MAX_PSQT_DELTA],
    psqt_sub: [usize; MAX_PSQT_DELTA],
    aux_add: [usize; MAX_AUX_DELTA],
    aux_sub: [usize; MAX_AUX_DELTA],
    n_psqt_add: usize,
    n_psqt_sub: usize,
    n_aux_add: usize,
    n_aux_sub: usize,
}

impl Lists {
    const fn new() -> Lists {
        Lists {
            psqt_add: [0; MAX_PSQT_DELTA],
            psqt_sub: [0; MAX_PSQT_DELTA],
            aux_add: [0; MAX_AUX_DELTA],
            aux_sub: [0; MAX_AUX_DELTA],
            n_psqt_add: 0,
            n_psqt_sub: 0,
            n_aux_add: 0,
            n_aux_sub: 0,
        }
    }

    fn clear(&mut self) {
        self.n_psqt_add = 0;
        self.n_psqt_sub = 0;
        self.n_aux_add = 0;
        self.n_aux_sub = 0;
    }

    #[inline(always)]
    fn psqt(&mut self, f: usize, add: bool) {
        if add {
            self.psqt_add[self.n_psqt_add] = f;
            self.n_psqt_add += 1;
        } else {
            self.psqt_sub[self.n_psqt_sub] = f;
            self.n_psqt_sub += 1;
        }
    }

    #[inline(always)]
    fn aux(&mut self, f: usize, add: bool) {
        // Start pulling the row in now; the rest of the enumeration hides the latency.
        #[cfg(target_feature = "avx2")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            // Pull in the first half of the row only. Requesting all sixteen lines of every
            // row floods the load buffers, and the hardware prefetcher streams the rest once
            // the update starts reading the row; measured 5% faster over 17 paired runs.
            let row = NET.aux_weights[f].as_ptr();
            let mut line = 0;
            while line < HL / 2 {
                _mm_prefetch(row.add(line), _MM_HINT_T0);
                line += 64;
            }
        }
        if add {
            self.aux_add[self.n_aux_add] = f;
            self.n_aux_add += 1;
        } else {
            self.aux_sub[self.n_aux_sub] = f;
            self.n_aux_sub += 1;
        }
    }
}

/// Perspective, king bucket and mirror flag a delta is mapped for.
#[derive(Clone, Copy)]
struct View {
    p: Color,
    bucket: usize,
    flip: u8,
}

/// Feature changes between two snapshots, enumerated once and mapped for up to two
/// perspectives at a time.
struct Delta {
    lists: [Lists; 2],
    views: [Option<View>; 2],
}

impl Delta {
    fn new() -> Box<Delta> {
        Box::new(Delta { lists: [Lists::new(), Lists::new()], views: [None; 2] })
    }

    #[inline(always)]
    fn psqt(&mut self, pc: Piece, sq: Square, add: bool) {
        for k in 0..2 {
            if let Some(v) = self.views[k] {
                self.lists[k].psqt(psqt_feature(v.p, v.bucket, v.flip, pc, sq), add);
            }
        }
    }

    #[inline(always)]
    fn threat(&mut self, attacker: Piece, src: Square, dest: Square, target: Piece, add: bool) {
        for k in 0..2 {
            if let Some(v) = self.views[k] {
                if let Some(f) = threat_feature(attacker, src, dest, target, v.p, v.flip) {
                    self.lists[k].aux(f, add);
                }
            }
        }
    }

    #[inline(always)]
    fn pp(&mut self, ca: Color, sa: Square, cb: Color, sb: Square, add: bool) {
        for k in 0..2 {
            if let Some(v) = self.views[k] {
                self.lists[k].aux(pp_feature(ca, sa, cb, sb, v.p, v.flip), add);
            }
        }
    }

    /// All threat features of the attacker on `s` in `snap`.
    fn threats_of(&mut self, snap: &Snapshot, s: Square, add: bool) {
        let pc = snap.board[s as usize];
        let mut b = snap.threats[s as usize];
        while b != 0 {
            let d = pop_lsb(&mut b);
            self.threat(pc, s, d, snap.board[d as usize], add);
        }
    }

    /// Fills the lists from a move's recorded threat changes plus the board and pawn
    /// differences between the snapshots; no attack sets are needed.
    fn compute_from_dirty(&mut self, old: &Snapshot, new: &Snapshot, dirty: &DirtyThreats) {
        self.lists[0].clear();
        self.lists[1].clear();
        let board_changed = board_diff(old, new);
        self.psqt_changes(old, new, board_changed);
        for d in dirty.as_slice() {
            self.threat(d.attacker(), d.from(), d.to(), d.target(), d.add());
        }
        self.pawn_pair_changes(old, new);
    }

    /// Piece-square features for the squares whose piece changed.
    #[inline]
    fn psqt_changes(&mut self, old: &Snapshot, new: &Snapshot, board_changed: Bitboard) {
        let mut b = board_changed;
        while b != 0 {
            let sq = pop_lsb(&mut b);
            let (o, n) = (old.board[sq as usize], new.board[sq as usize]);
            if !o.is_none() {
                self.psqt(o, sq, false);
            }
            if !n.is_none() {
                self.psqt(n, sq, true);
            }
        }
    }

    /// Fills the lists of every set view with what differs between `old` and `new`,
    /// from their attack sets.
    fn compute(&mut self, old: &Snapshot, new: &Snapshot) {
        self.lists[0].clear();
        self.lists[1].clear();

        let (board_changed, map_changed) = snapshot_diff(old, new);
        let changed = board_changed | map_changed;
        self.psqt_changes(old, new, board_changed);

        // Attackers whose piece changed: replace all their threats.
        let mut b = board_changed & old.attackers;
        while b != 0 {
            let s = pop_lsb(&mut b);
            self.threats_of(old, s, false);
        }
        let mut b = board_changed & new.attackers;
        while b != 0 {
            let s = pop_lsb(&mut b);
            self.threats_of(new, s, true);
        }

        // Same piece, different attack set (a line opened or closed): only the targets
        // that appeared or vanished, plus kept targets whose piece changed.
        let mut b = map_changed & !board_changed;
        while b != 0 {
            let s = pop_lsb(&mut b);
            let pc = old.board[s as usize];
            let (o, n) = (old.threats[s as usize], new.threats[s as usize]);
            let mut gone = o & !n;
            while gone != 0 {
                let d = pop_lsb(&mut gone);
                self.threat(pc, s, d, old.board[d as usize], false);
            }
            let mut fresh = n & !o;
            while fresh != 0 {
                let d = pop_lsb(&mut fresh);
                self.threat(pc, s, d, new.board[d as usize], true);
            }
            let mut swapped = o & n & board_changed;
            while swapped != 0 {
                let d = pop_lsb(&mut swapped);
                self.threat(pc, s, d, old.board[d as usize], false);
                self.threat(pc, s, d, new.board[d as usize], true);
            }
        }

        // Unchanged attackers looking at a square whose piece changed. Such a square holds
        // a target both before and after, else the attack set would have changed too.
        let stable = old.attackers & !changed;
        let mut b = board_changed & old.attackers;
        while b != 0 {
            let t = pop_lsb(&mut b);
            let mut a = stable & old.attackers_of(t);
            while a != 0 {
                let s = pop_lsb(&mut a);
                debug_assert!(old.threats[s as usize] & sq_bb(t) != 0);
                let pc = old.board[s as usize];
                self.threat(pc, s, t, old.board[t as usize], false);
                self.threat(pc, s, t, new.board[t as usize], true);
            }
        }

        self.pawn_pair_changes(old, new);
    }

    /// Pawn pairs: dissolve pairs of removed pawns, then form pairs of added pawns.
    fn pawn_pair_changes(&mut self, old: &Snapshot, new: &Snapshot) {
        let mut w = old.pawns;
        for c in [Color::White, Color::Black] {
            let mut removed = old.pawns[c.idx()] & !new.pawns[c.idx()];
            while removed != 0 {
                let r = pop_lsb(&mut removed);
                w[c.idx()] &= !sq_bb(r);
                for c2 in [Color::White, Color::Black] {
                    let mut x = w[c2.idx()] & pawn_band(r);
                    while x != 0 {
                        let s = pop_lsb(&mut x);
                        self.pp(c, r, c2, s, false);
                    }
                }
            }
        }
        for c in [Color::White, Color::Black] {
            let mut added = new.pawns[c.idx()] & !old.pawns[c.idx()];
            while added != 0 {
                let a = pop_lsb(&mut added);
                for c2 in [Color::White, Color::Black] {
                    let mut x = w[c2.idx()] & pawn_band(a);
                    while x != 0 {
                        let s = pop_lsb(&mut x);
                        self.pp(c, a, c2, s, true);
                    }
                }
                w[c.idx()] |= sq_bb(a);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct AccEntry {
    acc: [Accumulator; 2],
    computed: [bool; 2],
    snap: Snapshot,
    /// Threat changes of the move into this ply, when they were recorded in full.
    dirty: DirtyThreats,
    dirty_valid: bool,
}

#[derive(Clone, Copy)]
struct CacheEntry {
    acc: Accumulator,
    snap: Snapshot,
}

/// Per perspective, per (king bucket, mirror) accumulator cache.
#[derive(Clone)]
struct Cache {
    entries: [[CacheEntry; 2 * INPUT_BUCKETS]; 2],
}

impl Cache {
    fn new() -> Box<Cache> {
        Box::new(Cache { entries: [[CacheEntry { acc: NET.ft_bias, snap: Snapshot::EMPTY }; 2 * INPUT_BUCKETS]; 2] })
    }
}

/// Everything the network needs that lives alongside a `Position`.
pub struct NnueState {
    stack: Vec<AccEntry>,
    top: usize,
    cache: Box<Cache>,
    delta: Box<Delta>,
    /// Threat changes of the move being made, filled by the board primitives.
    pub dirty: DirtyThreats,
    /// True between `begin` and `push`: the primitives are making a move, not unmaking
    /// one or setting up a position.
    pub recording: bool,
}

impl Clone for NnueState {
    fn clone(&self) -> Self {
        NnueState {
            stack: self.stack.clone(),
            top: self.top,
            cache: self.cache.clone(),
            delta: Delta::new(),
            dirty: DirtyThreats::EMPTY,
            recording: false,
        }
    }
}

impl NnueState {
    pub fn new() -> NnueState {
        let root = AccEntry {
            acc: [NET.ft_bias; 2],
            computed: [false; 2],
            snap: Snapshot::EMPTY,
            dirty: DirtyThreats::EMPTY,
            dirty_valid: false,
        };
        let mut stack = Vec::with_capacity(64);
        stack.push(root);
        NnueState { stack, top: 0, cache: Cache::new(), delta: Delta::new(), dirty: DirtyThreats::EMPTY, recording: false }
    }

    /// Starts recording the threat changes of a move.
    #[inline(always)]
    pub fn begin(&mut self) {
        self.dirty.clear();
        self.recording = true;
    }

    /// Forgets everything: the board was set up from scratch.
    pub fn reset(&mut self, snap: Snapshot) {
        self.top = 0;
        self.stack[0].computed = [false; 2];
        self.stack[0].snap = snap;
        self.stack[0].dirty_valid = false;
        self.recording = false;
    }

    /// Opens the ply after a move, given the resulting position.
    #[inline]
    pub fn push(&mut self, snap: Snapshot) {
        let valid = self.recording && !self.dirty.overflow;
        self.recording = false;
        self.open(snap);
        let e = &mut self.stack[self.top];
        e.dirty_valid = valid;
        if valid {
            e.dirty.n = self.dirty.n;
            e.dirty.list[..self.dirty.n as usize].copy_from_slice(self.dirty.as_slice());
        }
    }

    /// Opens the ply after a null move: the board is unchanged, nothing is dirty.
    #[inline]
    pub fn push_same(&mut self) {
        let snap = self.stack[self.top].snap;
        self.open(snap);
        let e = &mut self.stack[self.top];
        e.dirty.n = 0;
        e.dirty_valid = true;
    }

    #[inline]
    fn open(&mut self, snap: Snapshot) {
        self.top += 1;
        if self.top == self.stack.len() {
            self.stack.push(AccEntry {
                acc: [NET.ft_bias; 2],
                computed: [false; 2],
                snap,
                dirty: DirtyThreats::EMPTY,
                dirty_valid: false,
            });
        } else {
            let e = &mut self.stack[self.top];
            e.computed = [false; 2];
            e.snap = snap;
        }
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        debug_assert!(self.top > 0);
        self.top -= 1;
    }

    /// Side-to-move relative evaluation in centipawns; `count` is the number of pieces.
    pub fn evaluate(&mut self, stm: Color, count: i32) -> i32 {
        let computed = self.stack[self.top].computed;
        match computed {
            [false, false] => self.update_both(),
            [false, true] => self.update_one(Color::White),
            [true, false] => self.update_one(Color::Black),
            [true, true] => {}
        }
        let e = &self.stack[self.top];
        let bucket = ((count - 2).max(0) as usize / (32 / OUTPUT_BUCKETS)).min(OUTPUT_BUCKETS - 1);
        let (us, them) = (&e.acc[stm.idx()], &e.acc[stm.flip().idx()]);
        count!(EVALS, 1);
        timed!(DOT, head(us, them, bucket))
    }

    /// How to bring `p`'s top accumulator up to date: rebuild it from the cache, or replay
    /// the plies from the given index (whose predecessor is computed).
    fn plan(&self, p: Color) -> Option<usize> {
        let mut i = self.top;
        loop {
            if i == 0 || self.needs_refresh(i, p) {
                return None;
            }
            if self.stack[i - 1].computed[p.idx()] {
                return Some(i);
            }
            i -= 1;
        }
    }

    fn update_one(&mut self, p: Color) {
        match self.plan(p) {
            None => self.refresh(p),
            Some(i) => self.replay(i, [p == Color::White, p == Color::Black]),
        }
    }

    /// Both perspectives share one pass over the changed features whenever they replay
    /// the same plies.
    fn update_both(&mut self) {
        match (self.plan(Color::White), self.plan(Color::Black)) {
            (Some(a), Some(b)) if a == b => self.replay(a, [true, true]),
            (w, b) => {
                match w {
                    None => self.refresh(Color::White),
                    Some(i) => self.replay(i, [true, false]),
                }
                match b {
                    None => self.refresh(Color::Black),
                    Some(i) => self.replay(i, [false, true]),
                }
            }
        }
    }

    fn view(&self, p: Color) -> View {
        let (bucket, flip) = king_context(p, self.stack[self.top].snap.kings[p.idx()]);
        View { p, bucket, flip }
    }

    /// Replays plies `i..=top` for the selected perspectives.
    fn replay(&mut self, i: usize, which: [bool; 2]) {
        let top = self.top;
        for k in 0..2 {
            let p = Color::from_idx(k);
            self.delta.views[k] = if which[k] { Some(self.view(p)) } else { None };
        }
        for j in i..=top {
            let (before, after) = self.stack.split_at_mut(j);
            let src = &mut before[j - 1];
            let e = &mut after[0];
            if e.dirty_valid {
                timed!(DELTA, self.delta.compute_from_dirty(&src.snap, &e.snap, &e.dirty));
            } else {
                src.snap.ensure_threats();
                e.snap.ensure_threats();
                timed!(DELTA, self.delta.compute(&src.snap, &e.snap));
            }
            for k in 0..2 {
                if which[k] {
                    count!(AUX_ROWS, self.delta.lists[k].n_aux_add + self.delta.lists[k].n_aux_sub);
                    timed!(UPDATE, simd::update(&src.acc[k], &mut e.acc[k], &self.delta.lists[k]));
                    e.computed[k] = true;
                }
            }
        }
    }

    /// Whether the move into ply `i` moved `p`'s king to another bucket or across the
    /// mirror line.
    #[inline]
    fn needs_refresh(&self, i: usize, p: Color) -> bool {
        let (from, to) = (self.stack[i - 1].snap.kings[p.idx()], self.stack[i].snap.kings[p.idx()]);
        from != to && king_context(p, from) != king_context(p, to)
    }

    /// Rebuilds the top accumulator for `p` from the cache entry of its king bucket.
    fn refresh(&mut self, p: Color) {
        self.stack[self.top].snap.ensure_threats();
        let k = p.idx();
        let view = self.view(p);
        self.delta.views = [None; 2];
        self.delta.views[k] = Some(view);
        let snap = &self.stack[self.top].snap;
        let entry = &mut self.cache.entries[k][2 * view.bucket + (view.flip != 0) as usize];
        count!(REFRESHES, 1);
        timed!(DELTA, self.delta.compute(&entry.snap, snap));
        count!(AUX_ROWS, self.delta.lists[k].n_aux_add + self.delta.lists[k].n_aux_sub);
        count!(AUX_ROWS_REFRESH, self.delta.lists[k].n_aux_add + self.delta.lists[k].n_aux_sub);
        count!(PSQT_ROWS_REFRESH, self.delta.lists[k].n_psqt_add + self.delta.lists[k].n_psqt_sub);
        timed!(UPDATE, simd::update_in_place(&mut entry.acc, &self.delta.lists[k]));
        entry.snap = *snap;
        let e = &mut self.stack[self.top];
        e.acc[k] = entry.acc;
        e.computed[k] = true;
    }
}

/// Active feature indices of `snap` for both perspectives, side to move first, as four
/// sorted lines (`stm_psqt`, `ntm_psqt`, `stm_aux`, `ntm_aux`) for cross-checking against
/// the trainer.
pub fn debug_features(mut snap: Snapshot, stm: Color) -> String {
    snap.ensure_threats();
    let mut d = Delta::new();
    let mut out = String::new();
    let mut lines = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (k, p) in [stm, stm.flip()].into_iter().enumerate() {
        let (bucket, flip) = king_context(p, snap.kings[p.idx()]);
        d.views = [Some(View { p, bucket, flip }), None];
        d.compute(&Snapshot::EMPTY, &snap);
        let l = &d.lists[0];
        lines[k] = l.psqt_add[..l.n_psqt_add].to_vec();
        lines[2 + k] = l.aux_add[..l.n_aux_add].to_vec();
    }
    for (name, v) in ["stm_psqt", "ntm_psqt", "stm_aux", "ntm_aux"].iter().zip(lines.iter_mut()) {
        v.sort_unstable();
        out.push_str(name);
        for f in v.iter() {
            out.push(' ');
            out.push_str(&f.to_string());
        }
        out.push('\n');
    }
    out
}

impl Default for NnueState {
    fn default() -> Self {
        NnueState::new()
    }
}

/// Bucket and file-mirror flag of `p`'s king on `ksq`.
#[inline(always)]
fn king_context(p: Color, ksq: Square) -> (usize, u8) {
    let rel = relative_square(p, ksq);
    (KING_BUCKET[rel as usize] as usize, if file_of(rel) > 3 { 7 } else { 0 })
}

/// Piece-square feature index of `pc` on `sq` from `p`'s perspective.
#[inline(always)]
fn psqt_feature(p: Color, bucket: usize, flip: u8, pc: Piece, sq: Square) -> usize {
    let side = if pc.color() == p { 0 } else { 384 };
    768 * bucket + side + 64 * pc.piece_type().idx() + (relative_square(p, sq) ^ flip) as usize
}

#[cfg(target_feature = "avx2")]
mod simd {
    use super::{Accumulator, CHUNKS, FT_SHIFT, HL, L2, L3, Lists, NET, PAIRED, QA};
    use std::arch::x86_64::*;

    const CHUNK: usize = 16;

    /// Pulls the piece-square rows of the delta towards the cache (the aux rows were
    /// prefetched as they were enumerated).
    #[inline(always)]
    fn prefetch(d: &Lists) {
        unsafe {
            for &r in d.psqt_add[..d.n_psqt_add].iter().chain(d.psqt_sub[..d.n_psqt_sub].iter()) {
                let row = NET.psqt_weights[r].v.as_ptr() as *const i8;
                for line in (0..2 * HL).step_by(64) {
                    _mm_prefetch(row.add(line), _MM_HINT_T0);
                }
            }
        }
    }

    /// Registers per tile: 16 ymm hold 256 accumulator lanes.
    const TILE_REGS: usize = 16;
    const TILE: usize = TILE_REGS * CHUNK;

    /// Applies every row of `d` to the tile starting at lane `j`, reading it from `src`
    /// and writing it to `dst`; the tile stays in registers throughout.
    #[inline(always)]
    unsafe fn apply_tile(src: *const i16, dst: *mut i16, j: usize, d: &Lists) {
        unsafe {
            let mut acc: [__m256i; TILE_REGS] = [_mm256_setzero_si256(); TILE_REGS];
            for k in 0..TILE_REGS {
                acc[k] = _mm256_load_si256(src.add(j + k * CHUNK) as *const __m256i);
            }
            for &r in &d.psqt_add[..d.n_psqt_add] {
                let row = NET.psqt_weights[r].v.as_ptr().add(j);
                for k in 0..TILE_REGS {
                    acc[k] = _mm256_add_epi16(acc[k], _mm256_load_si256(row.add(k * CHUNK) as *const __m256i));
                }
            }
            for &r in &d.psqt_sub[..d.n_psqt_sub] {
                let row = NET.psqt_weights[r].v.as_ptr().add(j);
                for k in 0..TILE_REGS {
                    acc[k] = _mm256_sub_epi16(acc[k], _mm256_load_si256(row.add(k * CHUNK) as *const __m256i));
                }
            }
            for &r in &d.aux_add[..d.n_aux_add] {
                let row = NET.aux_weights[r].as_ptr().add(j);
                for k in 0..TILE_REGS {
                    let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(k * CHUNK) as *const __m128i));
                    acc[k] = _mm256_add_epi16(acc[k], w);
                }
            }
            for &r in &d.aux_sub[..d.n_aux_sub] {
                let row = NET.aux_weights[r].as_ptr().add(j);
                for k in 0..TILE_REGS {
                    let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(k * CHUNK) as *const __m128i));
                    acc[k] = _mm256_sub_epi16(acc[k], w);
                }
            }
            for k in 0..TILE_REGS {
                _mm256_store_si256(dst.add(j + k * CHUNK) as *mut __m256i, acc[k]);
            }
        }
    }

    /// `dst = src + delta`.
    #[inline]
    pub fn update(src: &Accumulator, dst: &mut Accumulator, d: &Lists) {
        prefetch(d);
        unsafe {
            for j in (0..HL).step_by(TILE) {
                apply_tile(src.v.as_ptr(), dst.v.as_mut_ptr(), j, d);
            }
        }
    }

    /// `acc += delta`.
    #[inline]
    pub fn update_in_place(acc: &mut Accumulator, d: &Lists) {
        prefetch(d);
        unsafe {
            let p = acc.v.as_mut_ptr();
            for j in (0..HL).step_by(TILE) {
                apply_tile(p, p, j, d);
            }
        }
    }

    /// Clips both halves of `acc`, multiplies them together and writes the 512 bytes to
    /// `out`. `mulhi` on a pre-shifted operand does the shift for free.
    #[inline]
    unsafe fn pairwise(acc: &Accumulator, out: *mut u8) {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut j = 0;
        while j < PAIRED {
            let mut packed = [_mm256_setzero_si256(); 2];
            for (h, slot) in packed.iter_mut().enumerate() {
                let o = j + h * CHUNK;
                let lo = _mm256_load_si256(acc.v.as_ptr().add(o) as *const __m256i);
                let hi = _mm256_load_si256(acc.v.as_ptr().add(o + PAIRED) as *const __m256i);
                let lo = _mm256_min_epi16(_mm256_max_epi16(lo, zero), qa);
                let hi = _mm256_min_epi16(_mm256_max_epi16(hi, zero), qa);
                // (lo * hi) >> FT_SHIFT, keeping the product in 16 bits throughout.
                *slot = _mm256_mulhi_epu16(_mm256_slli_epi16(lo, 16 - FT_SHIFT as i32), hi);
            }
            // packus interleaves the two 128-bit lanes, so undo that.
            let bytes = _mm256_packus_epi16(packed[0], packed[1]);
            let bytes = _mm256_permute4x64_epi64(bytes, 0b11_01_10_00);
            _mm256_storeu_si256(out.add(j) as *mut __m256i, bytes);
            j += 2 * CHUNK;
        }
    }

    /// Pairwise transformer output, then the bucket's layer stack.
    ///
    /// The first layer visits only non-zero inputs, two at a time: with the activations
    /// capped at 127 a pair of products cannot saturate `maddubs`, so one instruction
    /// accumulates both inputs' contributions to sixteen outputs.
    pub unsafe fn head_avx2(us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
        let mut x = [0u8; HL];
        pairwise(us, x.as_mut_ptr());
        pairwise(them, x.as_mut_ptr().add(PAIRED));
        // Byte-granular rather than four-input chunks: at the ~9% density this net produces,
        // grouping inputs in fours makes a third of the chunks live, and the coarser skip
        // cancels out the cheaper iteration (measured at 1.02x, i.e. nothing). Chunking wins
        // from about 25% density upwards, so `l1_chunk4` is kept for a denser net.
        let a1 = l1_bytewise(&x, bucket);
        super::stack(&a1, bucket)
    }

    /// First affine layer over four inputs at a time, reading one contiguous 128-byte chunk
    /// of reordered weights per iteration. A chunk is live if any of its four bytes is, so
    /// this trades a coarser skip for much cheaper iterations.
    pub unsafe fn l1_chunk4(x: &[u8; HL], bucket: usize) -> [i32; L2] {
        let base = super::l1_chunked().w.as_ptr().add(bucket * CHUNKS);
        let x32 = x.as_ptr() as *const i32;
        let zero = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let mut acc = [zero; L2 / 8];
        for blk in 0..CHUNKS / 8 {
            let v = _mm256_loadu_si256(x32.add(blk * 8) as *const __m256i);
            let mut m = !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero))) as u32) & 0xFF;
            while m != 0 {
                let c = blk * 8 + m.trailing_zeros() as usize;
                m &= m - 1;
                let inv = _mm256_set1_epi32(*x32.add(c));
                let wp = (*base.add(c)).as_ptr();
                for (g, a) in acc.iter_mut().enumerate() {
                    let w = _mm256_loadu_si256(wp.add(g * 32) as *const __m256i);
                    *a = _mm256_add_epi32(*a, _mm256_madd_epi16(_mm256_maddubs_epi16(inv, w), ones));
                }
            }
        }
        let mut a1 = [0i32; L2];
        for (g, a) in acc.iter().enumerate() {
            _mm256_storeu_si256(a1.as_mut_ptr().add(g * 8) as *mut __m256i, *a);
        }
        a1
    }

    /// Byte-granular first layer, kept so the two can be timed against each other.
    pub unsafe fn l1_bytewise(x: &[u8; HL], bucket: usize) -> [i32; L2] {
        let zero = _mm256_setzero_si256();
        let mut nz = [0u16; HL];
        let mut cnt = 0usize;
        for c in (0..HL).step_by(32) {
            let v = _mm256_loadu_si256(x.as_ptr().add(c) as *const __m256i);
            let mut m = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, zero)) as u32);
            while m != 0 {
                *nz.get_unchecked_mut(cnt) = (c + m.trailing_zeros() as usize) as u16;
                cnt += 1;
                m &= m - 1;
            }
        }

        // maddubs pairs adjacent bytes, so the unpacked weights arrive as outputs
        // 0-7, 16-23 from the low half and 8-15, 24-31 from the high half.
        let (mut a, mut b, mut c2, mut d) = (zero, zero, zero, zero);
        let mut k = 0;
        while k < cnt {
            let i1 = *nz.get_unchecked(k) as usize;
            let (i2, x2) = if k + 1 < cnt {
                let i = *nz.get_unchecked(k + 1) as usize;
                (i, u32::from(*x.get_unchecked(i)))
            } else {
                (i1, 0)
            };
            let xv = _mm256_set1_epi16((u32::from(*x.get_unchecked(i1)) | (x2 << 8)) as i16);
            let w1 = _mm256_loadu_si256(NET.l1_weights[i1].as_ptr().add(bucket * L2) as *const __m256i);
            let w2 = _mm256_loadu_si256(NET.l1_weights[i2].as_ptr().add(bucket * L2) as *const __m256i);
            let lo = _mm256_maddubs_epi16(xv, _mm256_unpacklo_epi8(w1, w2));
            let hi = _mm256_maddubs_epi16(xv, _mm256_unpackhi_epi8(w1, w2));
            a = _mm256_add_epi32(a, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo)));
            b = _mm256_add_epi32(b, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo, 1)));
            c2 = _mm256_add_epi32(c2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi)));
            d = _mm256_add_epi32(d, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi, 1)));
            k += 2;
        }

        let mut a1 = [0i32; L2];
        _mm256_storeu_si256(a1.as_mut_ptr() as *mut __m256i, a);
        _mm256_storeu_si256(a1.as_mut_ptr().add(8) as *mut __m256i, c2);
        _mm256_storeu_si256(a1.as_mut_ptr().add(16) as *mut __m256i, b);
        _mm256_storeu_si256(a1.as_mut_ptr().add(24) as *mut __m256i, d);
        a1
    }

    /// Second affine layer. The weights are input-major, so a bucket's 32 outputs sit
    /// contiguously for each input and the whole layer is four running accumulators fed by
    /// one broadcast per input.
    pub unsafe fn l2_avx2(h2: &[f32; L2 * 2], b2: usize) -> [f32; L3] {
        let s = super::stack_f();
        let mut acc = [_mm256_setzero_ps(); 4];
        for (i, &h) in h2.iter().enumerate() {
            let hv = _mm256_set1_ps(h);
            let row = s.l2_weights.get_unchecked(i).as_ptr().add(b2);
            for (k, a) in acc.iter_mut().enumerate() {
                let w = _mm256_loadu_ps(row.add(k * 8));
                *a = _mm256_fmadd_ps(hv, w, *a);
            }
        }
        let mut out = [0f32; L3];
        for (k, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), *a);
        }
        out
    }
}

#[cfg(not(target_feature = "avx2"))]
mod simd {
    use super::{Accumulator, HL, Lists, NET};

    pub fn update(src: &Accumulator, dst: &mut Accumulator, d: &Lists) {
        dst.v = src.v;
        update_in_place(dst, d);
    }

    pub fn update_in_place(acc: &mut Accumulator, d: &Lists) {
        for &a in &d.psqt_add[..d.n_psqt_add] {
            for i in 0..HL {
                acc.v[i] += NET.psqt_weights[a].v[i];
            }
        }
        for &s in &d.psqt_sub[..d.n_psqt_sub] {
            for i in 0..HL {
                acc.v[i] -= NET.psqt_weights[s].v[i];
            }
        }
        for &a in &d.aux_add[..d.n_aux_add] {
            for i in 0..HL {
                acc.v[i] += i16::from(NET.aux_weights[a][i]);
            }
        }
        for &s in &d.aux_sub[..d.n_aux_sub] {
            for i in 0..HL {
                acc.v[i] -= i16::from(NET.aux_weights[s][i]);
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::evaluate;
    use crate::movegen::generate_legal;
    use crate::position::Position;
    use crate::types::MoveList;

    const FENS: [&str; 5] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/4P1b1/1nP2N2/PP1PQPPP/R4RKR w - - 0 1",
    ];

    /// The threat tables must match the trainer's: these are its attack-set sizes.
    #[test]
    fn threat_table_sizes() {
        assert_eq!(KNIGHT_THREATS.count, 336);
        assert_eq!(BISHOP_THREATS.count, 560);
        assert_eq!(ROOK_THREATS.count, 896);
        assert_eq!(QUEEN_THREATS.count, 1456);
        assert_eq!(QUEEN_THREATS.offset + 10 * QUEEN_THREATS.count, THREATS_PER_SIDE as u32);
        crate::init();
        for sq in 0..64u8 {
            assert_eq!(KNIGHT_THREATS.attacks[sq as usize], pseudo_attacks(PieceType::Knight, sq));
            assert_eq!(BISHOP_THREATS.attacks[sq as usize], pseudo_attacks(PieceType::Bishop, sq));
            assert_eq!(ROOK_THREATS.attacks[sq as usize], pseudo_attacks(PieceType::Rook, sq));
            assert_eq!(QUEEN_THREATS.attacks[sq as usize], pseudo_attacks(PieceType::Queen, sq));
        }
    }

    /// The lookup tables must agree with the arithmetic they replaced, for every attacker,
    /// target and reachable pair of squares, in both directions (neither may decline where
    /// the other produces a value).
    #[test]
    fn threat_lut_matches_reference() {
        // The original arithmetic, verbatim apart from returning the index rather than
        // asserting on it.
        fn reference(attacker: Piece, s: usize, d: usize, target: Piece, p: Color) -> Option<usize> {
            let tpt = target.piece_type().idx();
            let tclass = tpt + if target.color() == p { 0 } else { 6 };
            let pt = attacker.piece_type();
            let idx = if pt == PieceType::Pawn {
                let m = PAWN_TARGETS[tclass];
                if m == NO_TARGET {
                    return None;
                }
                let id = if d.abs_diff(s) == [9, 7][(d > s) as usize] { 0 } else { 1 };
                let attack = (2 * (s % 8) + id) as i32 - 1;
                m as usize * 84 + (s / 8 - 1) * 14 + attack as usize
            } else {
                let t = match pt {
                    PieceType::Knight => &KNIGHT_THREATS,
                    PieceType::Bishop => &BISHOP_THREATS,
                    PieceType::Rook => &ROOK_THREATS,
                    PieceType::Queen => &QUEEN_THREATS,
                    _ => return None,
                };
                let m = t.targets[tclass];
                if m == NO_TARGET || (d > s && tpt == pt.idx()) {
                    return None;
                }
                let within = (t.attacks[s] & ((1u64 << d) - 1)).count_ones();
                (t.offset + m as u32 * t.count + t.index[s] + within) as usize
            };
            let side = (attacker.color() != p) as usize;
            Some(PP_FEATURES + THREATS_PER_SIDE * side + idx)
        }

        // Squares `attacker` on `s` can actually attack, in p-relative coordinates: own
        // pawns capture upward, the opponent's downward.
        fn reachable(ra: usize, s: usize) -> Bitboard {
            let own = ra < 6;
            match ra % 6 {
                0 => {
                    let bb = sq_bb(s as Square);
                    if own {
                        ((bb << 7) & !file_bb_of(7)) | ((bb << 9) & !file_bb_of(0))
                    } else {
                        ((bb >> 7) & !file_bb_of(0)) | ((bb >> 9) & !file_bb_of(7))
                    }
                }
                1 => KNIGHT_THREATS.attacks[s],
                2 => BISHOP_THREATS.attacks[s],
                3 => ROOK_THREATS.attacks[s],
                4 => QUEEN_THREATS.attacks[s],
                _ => 0,
            }
        }

        crate::init();
        let mut checked = 0u64;
        for a in 0..12u8 {
            for t in 0..12u8 {
                let (attacker, target) = (Piece(a), Piece(t));
                for p in [Color::White, Color::Black] {
                    let ra = relative_piece(attacker, p);
                    for s in 0..64usize {
                        let mut bb = reachable(ra, s);
                        while bb != 0 {
                            let d = pop_lsb(&mut bb) as usize;
                            // Feed both the oriented squares, which is what the caller's
                            // relative_square/flip step produces.
                            let want = reference(attacker, s, d, target, p);
                            let got = threat_index(ra, relative_piece(target, p), s, d);
                            assert_eq!(
                                want, got,
                                "attacker {a} target {t} {s}->{d} perspective {p:?}: reference {want:?} lut {got:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 100_000, "only {checked} comparisons");
    }

    /// Every feature index of a full position lies inside its table, and no delta from
    /// the empty snapshot has duplicates (each feature is added once).
    #[test]
    fn features_in_range_and_unique() {
        crate::init();
        let mut d = Delta::new();
        for fen in FENS {
            let pos = Position::from_fen(fen).unwrap();
            let mut snap = pos.nnue_snapshot();
            snap.ensure_threats();
            for p in [Color::White, Color::Black] {
                let (bucket, flip) = king_context(p, snap.kings[p.idx()]);
                d.views = [Some(View { p, bucket, flip }), None];
                d.compute(&Snapshot::EMPTY, &snap);
                let l = &d.lists[0];
                assert_eq!(l.n_psqt_sub, 0);
                assert_eq!(l.n_aux_sub, 0);
                let mut aux: Vec<usize> = l.aux_add[..l.n_aux_add].to_vec();
                assert!(aux.iter().all(|&f| f < AUX_FEATURES));
                aux.sort_unstable();
                aux.dedup();
                assert_eq!(aux.len(), l.n_aux_add, "duplicate aux feature in {fen}");
                assert!(l.psqt_add[..l.n_psqt_add].iter().all(|&f| f < PSQT_FEATURES));
            }
        }
    }

    /// Prints the cost of each stage of an update; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn timings() {
        use std::time::Instant;
        crate::init();
        let fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8";
        let mut pos = Position::from_fen(fen).unwrap();
        let n = 200_000;

        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..n {
            let s = pos.nnue_snapshot();
            acc = acc.wrapping_add(s.attackers);
        }
        println!("snapshot build:   {:6.1} ns", t.elapsed().as_nanos() as f64 / n as f64);

        let t = Instant::now();
        for _ in 0..n {
            let mut s = pos.nnue_snapshot();
            s.ensure_threats();
            acc = acc.wrapping_add(s.threats[3]);
        }
        println!("build + threats:  {:6.1} ns", t.elapsed().as_nanos() as f64 / n as f64);

        let mut old = pos.nnue_snapshot();
        old.ensure_threats();
        let m = crate::types::Move::make(crate::types::make_square(2, 3), crate::types::make_square(3, 4), crate::types::MoveType::Normal, PieceType::Pawn);
        let gc = pos.gives_check(m);
        pos.make_move(m, gc);
        let mut new = pos.nnue_snapshot();
        new.ensure_threats();
        let mut d = Delta::new();
        let views = [Some(View { p: Color::White, bucket: 0, flip: 0 }), Some(View { p: Color::Black, bucket: 0, flip: 0 })];
        d.views = views;
        let t = Instant::now();
        for _ in 0..n {
            d.compute(&old, &new);
            acc = acc.wrapping_add(d.lists[0].n_aux_add as u64);
        }
        println!("delta (2 views):  {:6.1} ns  ({} psqt, {}+{} aux)", t.elapsed().as_nanos() as f64 / n as f64, d.lists[0].n_psqt_add + d.lists[0].n_psqt_sub, d.lists[0].n_aux_add, d.lists[0].n_aux_sub);
        d.views = [views[0], None];
        let t = Instant::now();
        for _ in 0..n {
            d.compute(&old, &new);
            acc = acc.wrapping_add(d.lists[0].n_aux_add as u64);
        }
        println!("delta (1 view):   {:6.1} ns", t.elapsed().as_nanos() as f64 / n as f64);

        let mut a = NET.ft_bias;
        let t = Instant::now();
        for _ in 0..n {
            simd::update_in_place(&mut a, &d.lists[0]);
            acc = acc.wrapping_add(a.v[7] as u64);
        }
        println!("acc update (warm):{:6.1} ns", t.elapsed().as_nanos() as f64 / n as f64);

        // Touch every row once: the network is demand-paged out of the executable, so
        // without this the first pattern measured pays all the page faults.
        {
            let mut warm = 0u64;
            for r in 0..AUX_FEATURES {
                warm = warm.wrapping_add(NET.aux_weights[r][0] as u64);
            }
            acc = acc.wrapping_add(warm);
        }

        // How much does the scattered layout of the aux table cost? Apply the same number
        // of rows with different spacing: adjacent rows share pages and stream; rows spread
        // across the whole 66 MB table miss the TLB and every cache level.
        for (label, stride) in [("adjacent", 1usize), ("same page", 4), ("near (64)", 64), ("scattered", 977)] {
            let mut l = Lists::new();
            let t = Instant::now();
            let mut row = 0usize;
            let reps = n / 10;
            for _ in 0..reps {
                l.clear();
                for _ in 0..20 {
                    l.aux(row % AUX_FEATURES, true);
                    row = row.wrapping_add(stride);
                }
                // start each batch somewhere new so nothing is already resident
                row = row.wrapping_add(7919);
                simd::update_in_place(&mut a, &l);
                acc = acc.wrapping_add(a.v[7] as u64);
            }
            println!("acc update, 20 rows {label:10}: {:6.0} ns", t.elapsed().as_nanos() as f64 / reps as f64);
        }

        // Perturb the accumulator and vary the bucket each iteration, or the whole call is
        // loop-invariant and gets hoisted out.
        let t = Instant::now();
        for i in 0..n {
            a.v[i % HL] = a.v[i % HL].wrapping_add(1);
            acc = acc.wrapping_add(head(&a, &a, i % OUTPUT_BUCKETS) as u64);
        }
        println!("head:             {:6.1} ns", t.elapsed().as_nanos() as f64 / n as f64);

        // Wall-clock times from separate runs are useless here: the same binary varies by
        // 15% between invocations. Timing both variants micro-seconds apart in one process
        // cancels that, because whatever the machine is doing affects them equally. The
        // ratio is the measurement; the absolute numbers are not.
        {
            let mut inputs = [[0i32; L2]; 64];
            let mut seed = 0x2545_F491_4F6C_DD1Du64;
            for row in inputs.iter_mut() {
                for v in row.iter_mut() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *v = (seed % (4 * S1 as u64)) as i32 - 2 * S1;
                }
            }
            let reps = 20000;
            let mut ratios = Vec::new();
            let (mut ta, mut tb) = (0u128, 0u128);
            for trial in 0..25 {
                let t = Instant::now();
                for i in 0..reps {
                    acc = acc.wrapping_add(stack_int(&inputs[i % 64], i % OUTPUT_BUCKETS) as u64);
                }
                let ea = t.elapsed().as_nanos();
                let t = Instant::now();
                for i in 0..reps {
                    acc = acc.wrapping_add(stack_flt(&inputs[i % 64], i % OUTPUT_BUCKETS) as u64);
                }
                let eb = t.elapsed().as_nanos();
                if trial >= 5 {
                    ta += ea;
                    tb += eb;
                    ratios.push(eb as f64 / ea as f64);
                }
            }
            ratios.sort_by(|x: &f64, y: &f64| x.partial_cmp(y).unwrap());
            let n2 = (ratios.len() * reps) as f64;
            println!(
                "stack int:        {:6.1} ns\nstack float:      {:6.1} ns\nfloat/int ratio:   {:.3} (median, {:.3}..{:.3} over {} paired trials)",
                ta as f64 / n2,
                tb as f64 / n2,
                ratios[ratios.len() / 2],
                ratios[0],
                ratios[ratios.len() - 1],
                ratios.len(),
            );
        }

        // Same paired method for the two first-layer shapes, on activations at the density
        // real positions produce (~9% of the 1024 inputs non-zero).
        #[cfg(target_feature = "avx2")]
        {
            for live in [96usize, 256, 512] {
            let mut xs = [[0u8; HL]; 16];
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            for x in xs.iter_mut() {
                for _ in 0..live {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    x[(seed % HL as u64) as usize] = (seed % 127 + 1) as u8;
                }
            }
            let reps = 20000;
            let mut ratios: Vec<f64> = Vec::new();
            let (mut ta, mut tb) = (0u128, 0u128);
            for trial in 0..25 {
                let t = Instant::now();
                for i in 0..reps {
                    acc = acc.wrapping_add(unsafe { simd::l1_bytewise(&xs[i % 16], i % OUTPUT_BUCKETS) }[0] as u64);
                }
                let ea = t.elapsed().as_nanos();
                let t = Instant::now();
                for i in 0..reps {
                    acc = acc.wrapping_add(unsafe { simd::l1_chunk4(&xs[i % 16], i % OUTPUT_BUCKETS) }[0] as u64);
                }
                let eb = t.elapsed().as_nanos();
                if trial >= 5 {
                    ta += ea;
                    tb += eb;
                    ratios.push(eb as f64 / ea as f64);
                }
            }
            ratios.sort_by(|x: &f64, y: &f64| x.partial_cmp(y).unwrap());
            let n2 = (ratios.len() * reps) as f64;
            println!(
                "density {:>4}/1024: bytewise {:6.1} ns, chunk4 {:6.1} ns, ratio {:.3} ({:.3}..{:.3})",
                live,
                ta as f64 / n2,
                tb as f64 / n2,
                ratios[ratios.len() / 2],
                ratios[0],
                ratios[ratios.len() - 1],
            );
            }
        }

        let t = Instant::now();
        for _ in 0..n {
            acc = acc.wrapping_add(evaluate(&mut pos) as u64);
        }
        println!("evaluate (cached):{:6.1} ns   [{acc}]", t.elapsed().as_nanos() as f64 / n as f64);
    }

    /// The vectorised head must agree with the reference on every bucket, including the
    /// saturating corners: accumulators pinned at the clip bounds and an odd number of
    /// non-zero inputs, which is the case the pairing has to special-case.
    #[cfg(target_feature = "avx2")]
    #[test]
    fn simd_head_matches_scalar() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..200 {
            let mut fill = |acc: &mut Accumulator| {
                for v in acc.v.iter_mut() {
                    *v = match case % 4 {
                        // mostly zero, so few inputs survive the clip
                        0 => (next() % 5) as i16 - 4,
                        // pinned at the bounds
                        1 => [0, QA as i16, -1, 3000][(next() % 4) as usize],
                        2 => (next() % 600) as i16 - 300,
                        _ => (next() % 64_000) as i16 - 32_000,
                    };
                }
            };
            let mut us = Accumulator { v: [0; HL] };
            let mut them = Accumulator { v: [0; HL] };
            fill(&mut us);
            fill(&mut them);
            for bucket in 0..OUTPUT_BUCKETS {
                let want = head_scalar(&us, &them, bucket);
                let got = unsafe { simd::head_avx2(&us, &them, bucket) };
                assert_eq!(want, got, "case {case}, bucket {bucket}");
            }
        }
    }

    /// The vectorised second layer must agree with the reference. This needs its own test:
    /// both heads route through the dispatching accumulator, so `simd_head_matches_scalar`
    /// would compare the vectorised path against itself.
    #[cfg(target_feature = "avx2")]
    #[test]
    fn simd_l2_matches_scalar() {
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..300 {
            let mut h2 = [0f32; L2 * 2];
            for v in h2.iter_mut() {
                *v = match case % 3 {
                    // the activation bounds, where the accumulation is largest
                    0 => 1.0,
                    1 => 0.0,
                    _ => (next() % 1_000_001) as f32 / 1.0e6,
                };
            }
            for bucket in 0..OUTPUT_BUCKETS {
                let b2 = bucket * L3;
                let want = l2_scalar(&h2, b2);
                let got = unsafe { simd::l2_avx2(&h2, b2) };
                for (k, (w, g)) in want.iter().zip(got.iter()).enumerate() {
                    // fused multiply-add rounds once where the reference rounds twice
                    assert!((w - g).abs() < 1e-3, "case {case}, bucket {bucket}, lane {k}: {w} vs {g}");
                }
            }
        }
    }

    /// Reordering the first layer's weights must not change what it computes.
    #[cfg(target_feature = "avx2")]
    #[test]
    fn chunked_l1_matches_bytewise() {
        let mut seed = 0xB5AD_4ECE_DA1C_E2A9u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..200 {
            let mut x = [0u8; HL];
            // Densities either side of the ~9% seen on real positions, plus the extremes,
            // since the two paths skip work at different granularities.
            let live = [0usize, 1, 20, 96, 400, HL][case % 6];
            for _ in 0..live {
                x[(next() % HL as u64) as usize] = (next() % 128) as u8;
            }
            for bucket in 0..OUTPUT_BUCKETS {
                let want = unsafe { simd::l1_bytewise(&x, bucket) };
                let got = unsafe { simd::l1_chunk4(&x, bucket) };
                assert_eq!(want, got, "case {case}, live {live}, bucket {bucket}");
            }
        }
    }

    /// Every move's recorded threat changes must equal the difference of the attack sets.
    #[test]
    fn dirty_threats_match_snapshots() {
        crate::init();
        let mut rng = 0x1234_5678_9ABC_DEF1u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut d1 = Delta::new();
        let mut d2 = Delta::new();
        let mut plies = 0;
        for fen in FENS {
            for _ in 0..20 {
                let mut pos = Position::from_fen(fen).unwrap();
                for _ in 0..200 {
                    let mut list = MoveList::default();
                    generate_legal(&pos, &mut list);
                    if list.is_empty() || pos.rule50_count() > 90 {
                        break;
                    }
                    let m = list.as_slice()[next() as usize % list.len()];
                    let before = pos.fen();
                    let uci = pos.move_to_uci(m);
                    let gives_check = pos.gives_check(m);
                    pos.make_move(m, gives_check);
                    let st = pos.nnue_mut();
                    let top = st.top;
                    assert!(st.stack[top].dirty_valid, "no dirty list after {uci} in {before}");
                    let (a, b) = st.stack.split_at_mut(top);
                    let (old, new) = (&mut a[top - 1], &mut b[0]);
                    old.snap.ensure_threats();
                    new.snap.ensure_threats();
                    for p in [Color::White, Color::Black] {
                        let (bucket, flip) = king_context(p, new.snap.kings[p.idx()]);
                        d1.views = [Some(View { p, bucket, flip }), None];
                        d2.views = d1.views;
                        d1.compute(&old.snap, &new.snap);
                        d2.compute_from_dirty(&old.snap, &new.snap, &new.dirty);
                        let net = |d: &Delta| {
                            let l = &d.lists[0];
                            let mut m = std::collections::BTreeMap::new();
                            for &f in &l.aux_add[..l.n_aux_add] {
                                *m.entry(f).or_insert(0i32) += 1;
                            }
                            for &f in &l.aux_sub[..l.n_aux_sub] {
                                *m.entry(f).or_insert(0i32) -= 1;
                            }
                            m.retain(|_, v| *v != 0);
                            m
                        };
                        let (n1, n2) = (net(&d1), net(&d2));
                        if n1 != n2 {
                            let only1: Vec<_> = n1.iter().filter(|(k, v)| n2.get(k) != Some(v)).collect();
                            let only2: Vec<_> = n2.iter().filter(|(k, v)| n1.get(k) != Some(v)).collect();
                            panic!(
                                "mismatch after {uci} in {before} (perspective {:?})
 snapshot-only {:?}
 dirty-only {:?}
 dirty list: {:?}",
                                p,
                                only1,
                                only2,
                                new.dirty.as_slice().iter().map(|d| (d.add(), d.attacker().0, d.from(), d.to(), d.target().0)).collect::<Vec<_>>()
                            );
                        }
                    }
                    plies += 1;
                }
            }
        }
        assert!(plies > 5000, "only {plies} plies");
    }

    /// Random playouts with evaluations at random plies, after unmade side branches and
    /// after null moves; every incremental value must equal a fresh evaluation of the same
    /// position, so this covers the snapshot diffs, king-bucket refreshes and the cache.
    #[test]
    fn incremental_matches_fresh() {
        crate::init();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut checks = 0;
        for fen in FENS {
            for _ in 0..12 {
                let mut pos = Position::from_fen(fen).unwrap();
                for _ in 0..160 {
                    let mut list = MoveList::default();
                    generate_legal(&pos, &mut list);
                    if list.is_empty() || pos.rule50_count() > 90 {
                        break;
                    }
                    let m = list.as_slice()[next() as usize % list.len()];

                    // A side branch that is unmade again, evaluated so its accumulators fill in.
                    if next() % 2 == 0 {
                        let side = list.as_slice()[next() as usize % list.len()];
                        let gives_check = pos.gives_check(side);
                        pos.make_move(side, gives_check);
                        let _ = evaluate(&mut pos);
                        pos.unmake_move(side);
                    }
                    if next() % 4 == 0 && !pos.in_check() {
                        pos.make_null_move();
                        let inc = evaluate(&mut pos);
                        let mut fresh = Position::from_fen(&pos.fen()).unwrap();
                        assert_eq!(inc, evaluate(&mut fresh), "after null move in {}", pos.fen());
                        pos.unmake_null_move();
                    }

                    let gives_check = pos.gives_check(m);
                    pos.make_move(m, gives_check);
                    if next() % 3 == 0 {
                        let inc = evaluate(&mut pos);
                        let mut fresh = Position::from_fen(&pos.fen()).unwrap();
                        assert_eq!(inc, evaluate(&mut fresh), "{}", pos.fen());
                        checks += 1;
                    }
                }
            }
        }
        assert!(checks > 500, "only {checks} comparisons");
    }
}
