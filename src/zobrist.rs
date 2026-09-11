//! Compile-time Zobrist keys (SplitMix64) and the cuckoo tables used to detect an
//! upcoming repetition without making a move.

use crate::bitboard::{attacks_bb, init as init_bitboards};
use crate::types::*;
use std::cell::UnsafeCell;
use std::sync::Once;

const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub struct ZobristKeys {
    pub psq: [[u64; 64]; PIECE_NB],
    pub en_passant: [u64; 8],
    pub castling: [u64; 16],
    pub side: u64,
    pub no_pawns: u64,
    /// Three-check: checks delivered so far by [color][count].
    #[cfg(feature = "variants")]
    pub checks: [[u64; 4]; 2],
    /// Crazyhouse pockets by [color][piece type][count].
    #[cfg(feature = "variants")]
    pub hand: [[[u64; 17]; 5]; 2],
    /// Crazyhouse: squares holding a promoted piece.
    #[cfg(feature = "variants")]
    pub promoted: [u64; 64],
}

pub static ZOBRIST: ZobristKeys = {
    let mut seed = 0x1070372u64;
    let mut psq = [[0u64; 64]; PIECE_NB];
    let mut p = 0;
    while p < PIECE_NB {
        let mut s = 0;
        while s < 64 {
            seed = splitmix64(seed);
            psq[p][s] = seed;
            s += 1;
        }
        p += 1;
    }
    let mut en_passant = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        seed = splitmix64(seed);
        en_passant[f] = seed;
        f += 1;
    }
    let mut castling = [0u64; 16];
    let mut c = 1;
    while c < 16 {
        seed = splitmix64(seed);
        castling[c] = seed;
        c += 1;
    }
    seed = splitmix64(seed);
    let side = seed;
    seed = splitmix64(seed);
    let no_pawns = seed;
    // Variant keys come last so the standard keys are identical in either build.
    #[cfg(feature = "variants")]
    let mut checks = [[0u64; 4]; 2];
    #[cfg(feature = "variants")]
    {
        let mut c = 0;
        while c < 2 {
            let mut n = 1;
            while n < 4 {
                seed = splitmix64(seed);
                checks[c][n] = seed;
                n += 1;
            }
            c += 1;
        }
    }
    #[cfg(feature = "variants")]
    let mut hand = [[[0u64; 17]; 5]; 2];
    #[cfg(feature = "variants")]
    {
        let mut c = 0;
        while c < 2 {
            let mut p = 0;
            while p < 5 {
                let mut n = 1;
                while n < 17 {
                    seed = splitmix64(seed);
                    hand[c][p][n] = seed;
                    n += 1;
                }
                p += 1;
            }
            c += 1;
        }
    }
    #[cfg(feature = "variants")]
    let mut promoted = [0u64; 64];
    #[cfg(feature = "variants")]
    {
        let mut s = 0;
        while s < 64 {
            seed = splitmix64(seed);
            promoted[s] = seed;
            s += 1;
        }
    }
    ZobristKeys {
        psq,
        en_passant,
        castling,
        side,
        no_pawns,
        #[cfg(feature = "variants")]
        checks,
        #[cfg(feature = "variants")]
        hand,
        #[cfg(feature = "variants")]
        promoted,
    }
};

#[inline(always)]
pub fn psq_key(pc: Piece, sq: Square) -> u64 {
    ZOBRIST.psq[pc.idx()][sq as usize]
}

// Cuckoo hashing of every reversible move's key delta, so `upcoming_repetition`
// can ask "is there a move that turns the current key into a previous one?".
const CUCKOO_SIZE: usize = 8192;

struct SyncCell<T>(UnsafeCell<T>);
unsafe impl<T> Sync for SyncCell<T> {}

pub struct Cuckoo {
    pub keys: [u64; CUCKOO_SIZE],
    pub moves: [Move; CUCKOO_SIZE],
}

static CUCKOO: SyncCell<Cuckoo> = SyncCell(UnsafeCell::new(Cuckoo {
    keys: [0; CUCKOO_SIZE],
    moves: [Move::NONE; CUCKOO_SIZE],
}));
static INIT: Once = Once::new();

#[inline(always)]
pub const fn h1(key: u64) -> usize {
    (key & 0x1fff) as usize
}

#[inline(always)]
pub const fn h2(key: u64) -> usize {
    ((key >> 16) & 0x1fff) as usize
}

#[inline(always)]
pub fn cuckoo() -> &'static Cuckoo {
    unsafe { &*CUCKOO.0.get() }
}

pub fn init() {
    INIT.call_once(|| {
        init_bitboards();
        let c = unsafe { &mut *CUCKOO.0.get() };
        let mut count = 0;
        for pc in 0..PIECE_NB as u8 {
            let piece = Piece(pc);
            let pt = piece.piece_type();
            if pt == PieceType::Pawn {
                continue;
            }
            for s1 in 0..64u8 {
                for s2 in (s1 + 1)..64u8 {
                    if attacks_bb(pt, s1, 0) & (1u64 << s2) == 0 {
                        continue;
                    }
                    let mut mv = Move::new(s1, s2);
                    let mut key = psq_key(piece, s1) ^ psq_key(piece, s2) ^ ZOBRIST.side;
                    let mut i = h1(key);
                    loop {
                        std::mem::swap(&mut c.keys[i], &mut key);
                        std::mem::swap(&mut c.moves[i], &mut mv);
                        if mv.is_none() {
                            break;
                        }
                        i = if i == h1(key) { h2(key) } else { h1(key) };
                    }
                    count += 1;
                }
            }
        }
        debug_assert_eq!(count, 3668);
    });
}
