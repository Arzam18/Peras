//! Time allocation for base+increment and moves-to-go controls, plus a fixed-per-move
//! mode where the whole budget is a soft target.

use crate::types::Color;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub time: [u64; 2],
    pub inc: [u64; 2],
    pub movestogo: u64,
    pub movetime: u64,
    pub depth: i32,
    pub nodes: u64,
    pub mate: i32,
    pub infinite: bool,
    pub ponder: bool,
    /// Root moves to restrict the search to (empty = all).
    pub searchmoves: Vec<crate::types::Move>,
}

impl Limits {
    pub fn use_time_management(&self) -> bool {
        self.time[0] != 0 || self.time[1] != 0
    }
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    pub start: Instant,
    /// Target thinking time (ms); dynamic factors scale it.
    pub optimum: u64,
    /// Hard ceiling (ms).
    pub maximum: u64,
    /// True for movetime/infinite/depth-limited searches: no flag risk, so the budget
    /// is a soft target rather than something to stay well under.
    pub soft: bool,
    /// Whether any wall-clock limit applies at all.
    pub timed: bool,
}

impl TimeManager {
    pub fn untimed() -> TimeManager {
        TimeManager {
            start: Instant::now(),
            optimum: u64::MAX / 4,
            maximum: u64::MAX / 4,
            soft: true,
            timed: false,
        }
    }

    /// `original_time_adjust` persists across the game (reset to a negative value on a
    /// new game) so the per-game scaling stays consistent.
    pub fn init(limits: &Limits, us: Color, ply: i32, move_overhead: u64, original_time_adjust: &mut f64) -> TimeManager {
        let start = Instant::now();

        if limits.infinite {
            return TimeManager::untimed();
        }
        if limits.movetime > 0 {
            let ms = limits.movetime.saturating_sub(move_overhead.min(limits.movetime / 2));
            return TimeManager {
                start,
                optimum: ms,
                maximum: ms,
                soft: true,
                timed: true,
            };
        }
        let time = limits.time[us.idx()];
        if time == 0 {
            // Depth/nodes-limited or nothing at all.
            return TimeManager::untimed();
        }
        let inc = limits.inc[us.idx()] as f64;
        let time_f = time as f64;
        let overhead = move_overhead as f64;

        let scaled_time = time_f.max(1.0);
        let mut mtg: f64 = if limits.movestogo > 0 { limits.movestogo.min(50) as f64 } else { 50.0 };
        // Below one second gradually reduce the horizon.
        if scaled_time < 1000.0 && limits.movestogo == 0 {
            mtg = (scaled_time * 0.05).max(1.0);
        }

        let time_left = (time_f + inc * (mtg - 1.0) - overhead * (2.0 + mtg)).max(1.0);

        let (opt_scale, max_scale);
        if limits.movestogo == 0 {
            if *original_time_adjust < 0.0 {
                *original_time_adjust = 0.3272 * time_left.log10() - 0.4141;
            }
            let log_time_sec = (scaled_time / 1000.0).log10();
            let opt_constant = (0.0029869 + 0.00033554 * log_time_sec).min(0.004905);
            let max_constant = (3.3744 + 3.0608 * log_time_sec).max(3.1441);
            opt_scale = (0.012112 + (ply as f64 + 3.22713).powf(0.46866) * opt_constant).min(0.19404 * time_f / time_left)
                * *original_time_adjust;
            max_scale = (max_constant + ply as f64 / 12.352).min(6.873);
        } else {
            opt_scale = ((0.88 + ply as f64 / 116.4) / mtg).min(0.88 * time_f / time_left);
            max_scale = 1.3 + 0.11 * mtg;
        }

        let optimum = (opt_scale * time_left).max(1.0);
        let maximum = optimum.max((0.8097 * time_f - overhead).min(max_scale * optimum));

        let mut optimum = optimum as u64;
        let mut maximum = maximum.max(1.0) as u64;
        // Never plan to spend more than the clock allows.
        let cap = (time_f * 0.9 - overhead).max(1.0) as u64;
        optimum = optimum.min(cap);
        maximum = maximum.min(cap).max(optimum);

        TimeManager {
            start,
            optimum,
            maximum,
            soft: false,
            timed: true,
        }
    }

    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}
