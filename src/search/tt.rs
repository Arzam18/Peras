//! Shared transposition table: 64-byte buckets of four 16-byte entries, each split
//! into two relaxed atomic words so every thread can read and write without locks.
//! Torn reads (data from one store, score from another) are tolerated as in every
//! lazy-SMP engine; the 32-bit key check filters almost all of them.

use crate::types::*;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

const ENTRIES_PER_BUCKET: usize = 4;
const GENERATION_BITS: u8 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
const GENERATION_MASK: u8 = 0xF8;
const PV_BIT: u8 = 0x04;
const REL: Ordering = Ordering::Relaxed;

/// Word 0: key32 | depth8 << 32 | gen_bound8 << 40 | move16 << 48.
/// Word 1: score16 | eval16 << 16.
#[repr(C)]
struct Entry {
    data: AtomicU64,
    extra: AtomicU64,
}

#[repr(C, align(64))]
struct Bucket {
    entries: [Entry; ENTRIES_PER_BUCKET],
}

const _: () = assert!(std::mem::size_of::<Bucket>() == 64);

#[derive(Clone, Copy, Debug)]
pub struct TTData {
    /// Ply-relative score already converted back by the caller via `value_from_tt`.
    pub score: Value,
    pub eval: Value,
    pub depth: i32,
    pub bound: Bound,
    pub is_pv: bool,
    pub mv: Move,
}

#[inline(always)]
fn pack_data(key32: u32, depth: u8, gen_bound: u8, mv: Move) -> u64 {
    key32 as u64 | (depth as u64) << 32 | (gen_bound as u64) << 40 | (mv.0 as u64) << 48
}

#[inline(always)]
fn pack_extra(score: i16, eval: i16) -> u64 {
    (score as u16 as u64) | (eval as u16 as u64) << 16
}

#[inline(always)]
fn data_key(d: u64) -> u32 {
    d as u32
}
#[inline(always)]
fn data_depth(d: u64) -> u8 {
    (d >> 32) as u8
}
#[inline(always)]
fn data_gen_bound(d: u64) -> u8 {
    (d >> 40) as u8
}
#[inline(always)]
fn data_move(d: u64) -> Move {
    Move((d >> 48) as u16)
}
#[inline(always)]
fn extra_score(e: u64) -> Value {
    (e as u16 as i16) as Value
}
#[inline(always)]
fn extra_eval(e: u64) -> Value {
    ((e >> 16) as u16 as i16) as Value
}

#[inline(always)]
fn pack_gen_bound(generation: u8, is_pv: bool, bound: Bound) -> u8 {
    (generation & GENERATION_MASK) | if is_pv { PV_BIT } else { 0 } | (bound as u8 & 3)
}

#[inline(always)]
fn relative_age(gen_bound: u8, current: u8) -> u8 {
    current.wrapping_sub(gen_bound & GENERATION_MASK) & GENERATION_MASK
}

/// Rebases a mate score from root-relative plies to node-relative for storage.
#[inline(always)]
pub fn value_to_tt(v: Value, ply: usize) -> Value {
    if is_win(v) {
        v + ply as Value
    } else if is_loss(v) {
        v - ply as Value
    } else {
        v
    }
}

/// Inverse of `value_to_tt`; a mate that the fifty-move rule would interrupt is
/// downgraded to the largest non-mate score so it cannot be trusted blindly.
#[inline(always)]
pub fn value_from_tt(v: Value, ply: usize, rule50: i32) -> Value {
    if v == VALUE_NONE {
        return VALUE_NONE;
    }
    if is_win(v) {
        if VALUE_MATE - v > 100 - rule50 {
            return VALUE_MATE_IN_MAX_PLY - 1;
        }
        return v - ply as Value;
    }
    if is_loss(v) {
        if VALUE_MATE + v > 100 - rule50 {
            return VALUE_MATED_IN_MAX_PLY + 1;
        }
        return v + ply as Value;
    }
    v
}

pub struct TranspositionTable {
    buckets: Box<[Bucket]>,
    mask: usize,
    generation: AtomicU8,
}

unsafe impl Sync for TranspositionTable {}
unsafe impl Send for TranspositionTable {}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> TranspositionTable {
        let bytes = size_mb.max(1) * 1024 * 1024;
        let count = (bytes / std::mem::size_of::<Bucket>()).max(1);
        // Power of two so the index is a mask.
        let mut cap = 1usize;
        while cap * 2 <= count {
            cap *= 2;
        }
        // All-zero bytes are the empty entry, so zeroed allocation needs no init pass.
        let buckets: Box<[Bucket]> = unsafe {
            let layout = std::alloc::Layout::array::<Bucket>(cap).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Bucket;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, cap))
        };
        TranspositionTable {
            buckets,
            mask: cap - 1,
            generation: AtomicU8::new(GENERATION_DELTA),
        }
    }

    pub fn size_mb(&self) -> usize {
        (self.mask + 1) * std::mem::size_of::<Bucket>() / (1024 * 1024)
    }

    #[inline(always)]
    fn key32(key: u64) -> u32 {
        (key >> 32) as u32
    }

    #[inline(always)]
    fn bucket(&self, key: u64) -> &Bucket {
        unsafe { self.buckets.get_unchecked((key as usize) & self.mask) }
    }

    #[inline(always)]
    pub fn generation(&self) -> u8 {
        self.generation.load(REL)
    }

    pub fn increment_age(&self) {
        self.generation.fetch_add(GENERATION_DELTA, REL);
    }

    pub fn clear(&self) {
        for b in self.buckets.iter() {
            for e in &b.entries {
                e.data.store(0, REL);
                e.extra.store(0, REL);
            }
        }
        self.generation.store(GENERATION_DELTA, REL);
    }

    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let ptr = self.buckets.as_ptr().add((key as usize) & self.mask) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }

    /// Looks the position up; the returned score is still node-relative.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTData> {
        let key32 = Self::key32(key);
        let bucket = self.bucket(key);
        let generation = self.generation();
        for e in &bucket.entries {
            let d = e.data.load(REL);
            if data_key(d) != key32 || d == 0 {
                continue;
            }
            // Refresh the generation so entries the search still uses win replacement
            // fights. A failed exchange means someone else rewrote the slot; fine.
            let refreshed = (d & !(0xFFu64 << 40)) | (((generation & GENERATION_MASK) | (data_gen_bound(d) & 0x07)) as u64) << 40;
            if refreshed != d {
                let _ = e.data.compare_exchange(d, refreshed, REL, REL);
            }
            let x = e.extra.load(REL);
            let gb = data_gen_bound(d);
            return Some(TTData {
                score: extra_score(x),
                eval: extra_eval(x),
                depth: data_depth(d) as i32,
                bound: Bound::from_u8(gb),
                is_pv: gb & PV_BIT != 0,
                mv: data_move(d),
            });
        }
        None
    }

    /// Cheap move-only lookup used to extend printed PVs.
    pub fn probe_move(&self, key: u64) -> Move {
        let key32 = Self::key32(key);
        for e in &self.bucket(key).entries {
            let d = e.data.load(REL);
            if data_key(d) == key32 && d != 0 {
                return data_move(d);
            }
        }
        Move::NONE
    }

    /// Shaves plies off an entry that was deep enough to cut but held the wrong bound,
    /// so a real search replaces it instead of re-probing a useless entry.
    pub fn penalize(&self, key: u64, penalty: u8) {
        let key32 = Self::key32(key);
        for e in &self.bucket(key).entries {
            let d = e.data.load(REL);
            if data_key(d) == key32 && d != 0 {
                let depth = data_depth(d).saturating_sub(penalty);
                let nd = (d & !(0xFFu64 << 32)) | (depth as u64) << 32;
                let _ = e.data.compare_exchange(d, nd, REL, REL);
                return;
            }
        }
    }

    /// Stores an entry. `score` must already be node-relative (`value_to_tt`).
    /// `eval == VALUE_NONE` keeps the previously stored static eval.
    #[allow(clippy::too_many_arguments)]
    pub fn store(&self, key: u64, depth: i32, bound: Bound, is_pv: bool, score: Value, eval: Value, mv: Move) {
        let key32 = Self::key32(key);
        let generation = self.generation();
        let bucket = self.bucket(key);
        let depth8 = depth.clamp(0, 255) as u8;
        let score16 = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

        let mut replace_idx = 0;
        let mut worst = i32::MAX;
        for (i, e) in bucket.entries.iter().enumerate() {
            let d = e.data.load(REL);
            if d != 0 && data_key(d) == key32 {
                let old_x = e.extra.load(REL);
                let old_depth = data_depth(d) as i32;
                let old_gb = data_gen_bound(d);
                let pv_bonus = if bound == Bound::Exact || is_pv { 2 } else { 0 };
                if bound == Bound::Exact || depth + pv_bonus > old_depth - 4 || relative_age(old_gb, generation) != 0 {
                    let new_mv = if mv.is_some() { mv } else { data_move(d) };
                    let new_eval = if eval != VALUE_NONE { eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16 } else { extra_eval(old_x) as i16 };
                    // Payload before key/data so a reader matching the new data sees it.
                    e.extra.store(pack_extra(score16, new_eval), REL);
                    e.data.store(pack_data(key32, depth8, pack_gen_bound(generation, is_pv, bound), new_mv), REL);
                } else if old_depth >= 5 && Bound::from_u8(old_gb) != Bound::Exact && is_decisive(extra_score(old_x)) {
                    // Only stale decisive bounds decay; aging ordinary deep bounds costs
                    // cutoffs table-wide.
                    let nd = (d & !(0xFFu64 << 32)) | ((old_depth - 1) as u64) << 32;
                    let _ = e.data.compare_exchange(d, nd, REL, REL);
                }
                return;
            }
            // Replacement priority: depth (+PV bonus), penalised by age.
            let priority = if d == 0 {
                i32::MIN
            } else {
                data_depth(d) as i32 + 3 + if data_gen_bound(d) & PV_BIT != 0 { 2 } else { 0 } - relative_age(data_gen_bound(d), generation) as i32
            };
            if priority < worst {
                worst = priority;
                replace_idx = i;
            }
        }

        let e = &bucket.entries[replace_idx];
        let eval16 = if eval == VALUE_NONE { VALUE_NONE as i16 } else { eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16 };
        e.extra.store(pack_extra(score16, eval16), REL);
        e.data.store(pack_data(key32, depth8, pack_gen_bound(generation, is_pv, bound), mv), REL);
    }

    /// Approximate fill in permille, sampling the first buckets for current-generation entries.
    pub fn hashfull(&self) -> u32 {
        let sample = (self.mask + 1).min(1000);
        let generation = self.generation();
        let mut occ = 0u32;
        for b in self.buckets.iter().take(sample) {
            for e in &b.entries {
                let d = e.data.load(REL);
                if d != 0 && relative_age(data_gen_bound(d), generation) == 0 {
                    occ += 1;
                }
            }
        }
        occ * 1000 / (sample * ENTRIES_PER_BUCKET) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let tt = TranspositionTable::new(1);
        let key = 0x1234_5678_9ABC_DEF0u64;
        let mv = Move::new(12, 28);
        tt.store(key, 7, Bound::Exact, true, 123, -45, mv);
        let d = tt.probe(key).unwrap();
        assert_eq!(d.score, 123);
        assert_eq!(d.eval, -45);
        assert_eq!(d.depth, 7);
        assert_eq!(d.bound, Bound::Exact);
        assert!(d.is_pv);
        assert_eq!(d.mv, mv);
        assert!(tt.probe(key ^ 0xFFFF_FFFF_0000_0000).is_none());
    }

    #[test]
    fn shallow_store_keeps_deep_entry_and_move() {
        let tt = TranspositionTable::new(1);
        let key = 0xABCD_EF01_2345_6789u64;
        let mv = Move::new(8, 16);
        tt.store(key, 10, Bound::Lower, false, 100, 90, mv);
        tt.store(key, 0, Bound::Upper, false, -500, -500, Move::NONE);
        let d = tt.probe(key).unwrap();
        assert_eq!(d.depth, 10);
        assert_eq!(d.score, 100);
        assert_eq!(d.mv, mv);
    }

    #[test]
    fn none_eval_sentinel_survives() {
        let tt = TranspositionTable::new(1);
        let key = 0x0F0F_0F0F_F0F0_F0F0u64;
        tt.store(key, 3, Bound::Lower, false, 50, VALUE_NONE, Move::NONE);
        assert_eq!(tt.probe(key).unwrap().eval, VALUE_NONE);
    }

    #[test]
    fn mate_scores_survive_tt() {
        for ply in 0..MAX_PLY {
            let v = mate_in(ply + 3);
            let stored = value_to_tt(v, ply);
            assert_eq!(value_from_tt(stored, ply, 0), v);
            let v = mated_in(ply + 3);
            let stored = value_to_tt(v, ply);
            assert_eq!(value_from_tt(stored, ply, 0), v);
        }
    }
}
