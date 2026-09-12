//! NNUE evaluation.
//!
//! Architecture: `(768 x 10 king buckets, horizontally mirrored -> 1024)x2 -> 8 output buckets`
//! with SCReLU activation, trained with bullet. Weights are `i16` quantised by 255 (feature
//! transformer) and 64 (output layer) and embedded in the binary.
//!
//! Each perspective keeps an accumulator per ply. `make_move` only records which pieces
//! changed; the accumulators are brought up to date on the first evaluation that needs
//! them by replaying those changes from the nearest computed ply. When the king crosses
//! into another bucket (or over the mirror line) the accumulator is instead rebuilt from a
//! cache holding the last accumulator seen in that bucket, updating only the pieces that
//! differ from the cached board.

use crate::bitboard::*;
use crate::types::*;

pub const HL: usize = 1024;
const INPUT_BUCKETS: usize = 10;
const OUTPUT_BUCKETS: usize = 8;
const FEATURES: usize = 768 * INPUT_BUCKETS;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

/// King bucket by square (own perspective, a1 = 0), given for the a-d files only; the
/// e-h files mirror onto them.
#[rustfmt::skip]
const BUCKET_LAYOUT: [u8; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
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

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    v: [i16; HL],
}

#[repr(C)]
struct Network {
    ft_weights: [Accumulator; FEATURES],
    ft_bias: Accumulator,
    out_weights: [[i16; 2 * HL]; OUTPUT_BUCKETS],
    out_bias: [i16; OUTPUT_BUCKETS],
}

/// The network file is padded to a 64-byte multiple, which is exactly the struct's size.
static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("PERAS_NET"))) };

/// One board change: `from` and/or `to` may be `SQ_NONE`.
#[derive(Clone, Copy)]
pub struct DirtyPiece {
    pc: Piece,
    from: Square,
    to: Square,
}

const NO_DIRTY: DirtyPiece = DirtyPiece { pc: Piece::NONE, from: SQ_NONE, to: SQ_NONE };

/// A move touches at most four primitives (castling and capture promotions both take four).
pub const MAX_DIRTY: usize = 4;

#[derive(Clone, Copy)]
struct AccEntry {
    acc: [Accumulator; 2],
    computed: [bool; 2],
    dirty: [DirtyPiece; MAX_DIRTY],
    n_dirty: u8,
}

#[derive(Clone, Copy)]
struct CacheEntry {
    acc: Accumulator,
    pieces: [Bitboard; 12],
}

/// Per perspective, per (king bucket, mirror) accumulator cache.
#[derive(Clone)]
struct Cache {
    entries: [[CacheEntry; 2 * INPUT_BUCKETS]; 2],
}

impl Cache {
    fn new() -> Box<Cache> {
        Box::new(Cache { entries: [[CacheEntry { acc: NET.ft_bias, pieces: [0; 12] }; 2 * INPUT_BUCKETS]; 2] })
    }
}

/// Everything the network needs that lives alongside a `Position`.
#[derive(Clone)]
pub struct NnueState {
    stack: Vec<AccEntry>,
    top: usize,
    cache: Box<Cache>,
    dirty: [DirtyPiece; MAX_DIRTY],
    n_dirty: u8,
}

impl NnueState {
    pub fn new() -> NnueState {
        let root = AccEntry { acc: [NET.ft_bias; 2], computed: [false; 2], dirty: [NO_DIRTY; MAX_DIRTY], n_dirty: 0 };
        let mut stack = Vec::with_capacity(64);
        stack.push(root);
        NnueState { stack, top: 0, cache: Cache::new(), dirty: [NO_DIRTY; MAX_DIRTY], n_dirty: 0 }
    }

    /// Forgets everything: the board was set up from scratch.
    pub fn reset(&mut self) {
        self.top = 0;
        self.stack[0].computed = [false; 2];
        self.stack[0].n_dirty = 0;
        self.n_dirty = 0;
    }

    /// Starts recording the changes of a new move.
    #[inline(always)]
    pub fn begin(&mut self) {
        self.n_dirty = 0;
    }

    /// Records one board primitive.
    #[inline(always)]
    pub fn record(&mut self, pc: Piece, from: Square, to: Square) {
        let n = self.n_dirty as usize;
        if n < MAX_DIRTY {
            self.dirty[n] = DirtyPiece { pc, from, to };
        }
        self.n_dirty += 1;
    }

    /// Opens the ply after the recorded move.
    #[inline(always)]
    pub fn push(&mut self) {
        self.top += 1;
        if self.top == self.stack.len() {
            self.stack.push(AccEntry {
                acc: [NET.ft_bias; 2],
                computed: [false; 2],
                dirty: [NO_DIRTY; MAX_DIRTY],
                n_dirty: 0,
            });
        }
        let e = &mut self.stack[self.top];
        e.computed = [false; 2];
        e.dirty = self.dirty;
        e.n_dirty = self.n_dirty;
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        debug_assert!(self.top > 0);
        self.top -= 1;
    }

    /// Side-to-move relative evaluation in centipawns.
    ///
    /// `pieces` is indexed by `Piece::idx()`; `count` is the number of pieces on the board.
    pub fn evaluate(&mut self, stm: Color, kings: [Square; 2], pieces: &[Bitboard; 12], count: i32) -> i32 {
        for p in [Color::White, Color::Black] {
            if !self.stack[self.top].computed[p.idx()] {
                self.bring_up_to_date(p, kings[p.idx()], pieces);
            }
        }
        let e = &self.stack[self.top];
        let bucket = ((count - 2).max(0) as usize / (32 / OUTPUT_BUCKETS)).min(OUTPUT_BUCKETS - 1);
        let w = &NET.out_weights[bucket];
        let (us, them) = (&e.acc[stm.idx()], &e.acc[stm.flip().idx()]);
        let sum = simd::dot_screlu(us, &w[..HL]) + simd::dot_screlu(them, &w[HL..]);
        (sum / QA + i32::from(NET.out_bias[bucket])) * SCALE / (QA * QB)
    }

    fn bring_up_to_date(&mut self, p: Color, ksq: Square, pieces: &[Bitboard; 12]) {
        let top = self.top;
        let mut i = top;
        loop {
            if i == 0 || self.needs_refresh(i, p) {
                self.refresh(p, ksq, pieces);
                return;
            }
            if self.stack[i - 1].computed[p.idx()] {
                break;
            }
            i -= 1;
        }
        let (bucket, flip) = king_context(p, ksq);
        for j in i..=top {
            let (before, after) = self.stack.split_at_mut(j);
            let src = &before[j - 1].acc[p.idx()];
            let e = &mut after[0];
            let mut adds = [0usize; MAX_DIRTY];
            let mut subs = [0usize; MAX_DIRTY];
            let (mut na, mut ns) = (0, 0);
            for d in &e.dirty[..e.n_dirty as usize] {
                if d.from != SQ_NONE {
                    subs[ns] = feature(p, bucket, flip, d.pc, d.from);
                    ns += 1;
                }
                if d.to != SQ_NONE {
                    adds[na] = feature(p, bucket, flip, d.pc, d.to);
                    na += 1;
                }
            }
            simd::update(src, &mut e.acc[p.idx()], &adds[..na], &subs[..ns]);
            e.computed[p.idx()] = true;
        }
    }

    /// Whether the move into ply `i` moved `p`'s king to another bucket or across the
    /// mirror line (or recorded more changes than fit, which only happens on setup).
    fn needs_refresh(&self, i: usize, p: Color) -> bool {
        let e = &self.stack[i];
        if e.n_dirty as usize > MAX_DIRTY {
            return true;
        }
        let king = Piece::make(p, PieceType::King);
        let (mut from, mut to) = (SQ_NONE, SQ_NONE);
        for d in &e.dirty[..e.n_dirty as usize] {
            if d.pc == king {
                if d.from != SQ_NONE {
                    from = d.from;
                }
                if d.to != SQ_NONE {
                    to = d.to;
                }
            }
        }
        if from == SQ_NONE {
            return false;
        }
        to == SQ_NONE || king_context(p, from) != king_context(p, to)
    }

    /// Rebuilds the top accumulator for `p` from the cache entry of its king bucket.
    fn refresh(&mut self, p: Color, ksq: Square, pieces: &[Bitboard; 12]) {
        let (bucket, flip) = king_context(p, ksq);
        let entry = &mut self.cache.entries[p.idx()][2 * bucket + (flip != 0) as usize];
        let mut adds = [0usize; 32];
        let mut subs = [0usize; 32];
        let (mut na, mut ns) = (0, 0);
        for pc in 0..12 {
            let piece = Piece(pc as u8);
            let mut added = pieces[pc] & !entry.pieces[pc];
            while added != 0 {
                adds[na] = feature(p, bucket, flip, piece, pop_lsb(&mut added));
                na += 1;
            }
            let mut removed = entry.pieces[pc] & !pieces[pc];
            while removed != 0 {
                subs[ns] = feature(p, bucket, flip, piece, pop_lsb(&mut removed));
                ns += 1;
            }
        }
        simd::update_in_place(&mut entry.acc, &adds[..na], &subs[..ns]);
        entry.pieces = *pieces;
        let e = &mut self.stack[self.top];
        e.acc[p.idx()] = entry.acc;
        e.computed[p.idx()] = true;
    }
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

/// Feature index of `pc` on `sq` from `p`'s perspective.
#[inline(always)]
fn feature(p: Color, bucket: usize, flip: u8, pc: Piece, sq: Square) -> usize {
    let side = if pc.color() == p { 0 } else { 384 };
    768 * bucket + side + 64 * pc.piece_type().idx() + (relative_square(p, sq) ^ flip) as usize
}

#[cfg(target_feature = "avx2")]
mod simd {
    use super::{Accumulator, HL, NET, QA};
    use std::arch::x86_64::*;

    const CHUNK: usize = 16;

    /// `dst = src + sum(adds) - sum(subs)` in one pass over the accumulator.
    #[inline]
    pub fn update(src: &Accumulator, dst: &mut Accumulator, adds: &[usize], subs: &[usize]) {
        unsafe {
            for c in (0..HL).step_by(CHUNK) {
                let mut v = _mm256_load_si256(src.v.as_ptr().add(c) as *const __m256i);
                for &a in adds {
                    v = _mm256_add_epi16(v, _mm256_load_si256(NET.ft_weights[a].v.as_ptr().add(c) as *const __m256i));
                }
                for &s in subs {
                    v = _mm256_sub_epi16(v, _mm256_load_si256(NET.ft_weights[s].v.as_ptr().add(c) as *const __m256i));
                }
                _mm256_store_si256(dst.v.as_mut_ptr().add(c) as *mut __m256i, v);
            }
        }
    }

    #[inline]
    pub fn update_in_place(acc: &mut Accumulator, adds: &[usize], subs: &[usize]) {
        unsafe {
            for c in (0..HL).step_by(CHUNK) {
                let p = acc.v.as_mut_ptr().add(c) as *mut __m256i;
                let mut v = _mm256_load_si256(p);
                for &a in adds {
                    v = _mm256_add_epi16(v, _mm256_load_si256(NET.ft_weights[a].v.as_ptr().add(c) as *const __m256i));
                }
                for &s in subs {
                    v = _mm256_sub_epi16(v, _mm256_load_si256(NET.ft_weights[s].v.as_ptr().add(c) as *const __m256i));
                }
                _mm256_store_si256(p, v);
            }
        }
    }

    /// `sum(screlu(acc[i]) * w[i])`. The clipped value (at most 255) times the weight (at
    /// most 127 in magnitude) fits an `i16`, so the square goes through `madd`.
    #[inline]
    pub fn dot_screlu(acc: &Accumulator, w: &[i16]) -> i32 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);
            let mut sum = _mm256_setzero_si256();
            for c in (0..HL).step_by(CHUNK) {
                let x = _mm256_load_si256(acc.v.as_ptr().add(c) as *const __m256i);
                let x = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);
                let wv = _mm256_loadu_si256(w.as_ptr().add(c) as *const __m256i);
                let xw = _mm256_mullo_epi16(x, wv);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(xw, x));
            }
            let hi = _mm256_extracti128_si256(sum, 1);
            let lo = _mm256_castsi256_si128(sum);
            let s = _mm_add_epi32(lo, hi);
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
            _mm_cvtsi128_si32(s)
        }
    }
}

#[cfg(not(target_feature = "avx2"))]
mod simd {
    use super::{Accumulator, HL, NET, QA};

    pub fn update(src: &Accumulator, dst: &mut Accumulator, adds: &[usize], subs: &[usize]) {
        dst.v = src.v;
        update_in_place(dst, adds, subs);
    }

    pub fn update_in_place(acc: &mut Accumulator, adds: &[usize], subs: &[usize]) {
        for &a in adds {
            for i in 0..HL {
                acc.v[i] += NET.ft_weights[a].v[i];
            }
        }
        for &s in subs {
            for i in 0..HL {
                acc.v[i] -= NET.ft_weights[s].v[i];
            }
        }
    }

    pub fn dot_screlu(acc: &Accumulator, w: &[i16]) -> i32 {
        let mut sum = 0i32;
        for i in 0..HL {
            let x = i32::from(acc.v[i]).clamp(0, QA);
            sum += x * x * i32::from(w[i]);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
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

    /// Random playouts with evaluations at random plies, after unmade side branches and
    /// after null moves; every incremental value must equal a fresh evaluation of the same
    /// position, so this covers the dirty-piece replay, king-bucket refreshes and the cache.
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
