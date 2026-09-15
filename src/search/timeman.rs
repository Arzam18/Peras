//! Time allocation for base+increment and moves-to-go controls, plus a fixed-per-move
//! mode where the whole budget is a soft target.

use crate::types::{Color, Value};
use std::time::Instant;

/// Position context the allocator needs beyond the clock itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct Situation {
    pub ply: i32,
    pub rule50: i32,
    pub last_score: Option<Value>,
}

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

    /// Allocates this move's budget: increment spent in full each move, the rest of the
    /// clock amortised over the weighted moves remaining, scaled by the absolute clock.
    pub fn init(limits: &Limits, us: Color, sit: Situation, move_overhead: u64) -> TimeManager {
        let start = Instant::now();

        if limits.infinite {
            return TimeManager::untimed();
        }
        if limits.movetime > 0 {
            let ms = limits.movetime.saturating_sub(move_overhead.min(limits.movetime / 2));
            return TimeManager { start, optimum: ms, maximum: ms, soft: true, timed: true };
        }
        let time = limits.time[us.idx()];
        if time == 0 {
            // Depth/nodes-limited or nothing at all.
            return TimeManager::untimed();
        }
        let time_f = time as f64;
        let inc = limits.inc[us.idx()] as f64;
        let overhead = move_overhead as f64;

        let weight = phase_weight(sit.ply) * shuffle_discount(sit.rule50, sit.last_score);
        let cyclic = limits.movestogo > 0;

        // A repeating control only has this cycle's moves to weigh, all of which will
        // be played; a normal control weighs the whole rest of the game.
        let (rest, share) = if cyclic {
            let mut rest = 0.0;
            for i in 1..limits.movestogo {
                rest += phase_weight(sit.ply + 2 * i as i32);
            }
            let share = (CYCLE_AMORTISE * weight / (weight + rest)).min(CYCLE_MAX_SHARE);
            (rest, share)
        } else {
            let rest = weighted_moves_left(sit.ply);
            (rest, AMORTISE * weight / (weight + rest))
        };

        let bank = (time_f - overhead * 4.0 - 50.0).max(0.0);

        let budget_s = (time_f + inc * rest.min(HORIZON_CAP as f64)) / 1000.0;
        let tc = TC_FLOOR + (TC_CEIL - TC_FLOOR) * budget_s / (budget_s + TC_HALF);

        // Behind on the clock spends less; ahead does not spend more.
        let theirs = limits.time[us.flip().idx()] as f64;
        let edge = (time_f - theirs) / (1.0 + time_f + theirs);
        let disadvantage = 1.0 + 0.9 * edge.min(0.0);

        let move_cap = if cyclic {
            let mtg = limits.movestogo as f64;
            0.30 + 0.5 / (mtg * mtg)
        } else {
            MOVE_CAP
        };
        // Run the budget out on the last move of a period, if the clock is long enough.
        let spend_all = cyclic && limits.movestogo <= 1 && time_f > SPEND_ALL_MIN_MS;
        let hard = (time_f * if spend_all { 0.72 } else { 0.81 } - overhead * 2.0).max(1.0);

        let optimum = ((inc * INC_USE + bank * share) * tc * disadvantage)
            .min(time_f * move_cap)
            .min(hard)
            .max(1.0);

        // Allowed overrun when the search is unsettled, capped at the hard ceiling.
        let burst = (BURST_BASE + sit.ply as f64 * BURST_PER_PLY).min(BURST_CAP);
        let maximum = (optimum * burst).min(hard).max(optimum);

        let optimum = (optimum as u64).max(1);
        let maximum = (maximum as u64).max(optimum);

        TimeManager { start, optimum, maximum, soft: spend_all, timed: true }
    }

    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

// Move weight: a bell over the middlegame, floor PHASE_FLOOR elsewhere.
const PEAK_PLY: f64 = 28.0;
const WIDTH_PLY: f64 = 32.0;
const PHASE_FLOOR: f64 = 0.38;

// Survival curve fitted to 660 games (median ply 128); weighting future moves by it
// is what avoids reserving time for shuffles that usually don't happen.
const MEDIAN_PLY: f64 = 128.0;
const SURVIVE_WIDTH: f64 = 34.0;

// Horizon for the weighted-moves-left sum.
const HORIZON_CAP: i32 = 60;
const MAX_PLY_AHEAD: i32 = 260;

// Increment is spent in full each move; AMORTISE controls how hard the rest of the
// clock is drawn down (1.0 = evenly over the weighted moves remaining).
const INC_USE: f64 = 0.95;
const AMORTISE: f64 = 1.3;

// Share scales up with the absolute clock (TC_HALF = committable seconds at halfway).
const TC_FLOOR: f64 = 0.95;
const TC_CEIL: f64 = 1.55;
const TC_HALF: f64 = 90.0;

// Halfmove-clock shuffle discount, gated on a quiet score (QUIET_CP).
const SHUFFLE_FROM: f64 = 14.0;
const SHUFFLE_MIN: f64 = 0.45;
const QUIET_CP: i32 = 120;

// Moves-per-period control: the budget is spread over that period's own moves only.
const CYCLE_AMORTISE: f64 = 1.7;
const CYCLE_MAX_SHARE: f64 = 0.80;
const SPEND_ALL_MIN_MS: f64 = 5000.0;

const MOVE_CAP: f64 = 0.35;
const BURST_BASE: f64 = 3.0;
const BURST_PER_PLY: f64 = 1.0 / 24.0;
const BURST_CAP: f64 = 5.0;

fn phase_weight(ply: i32) -> f64 {
    let m = (ply as f64 - PEAK_PLY) / WIDTH_PLY;
    PHASE_FLOOR + (1.0 - PHASE_FLOOR) * (-0.5 * m * m).exp()
}

fn survival(ply: i32) -> f64 {
    1.0 / (1.0 + ((ply as f64 - MEDIAN_PLY) / SURVIVE_WIDTH).exp())
}

fn weighted_moves_left(ply: i32) -> f64 {
    let here = survival(ply).max(1e-6);
    let mut total = 0.0;
    for i in 1..=HORIZON_CAP {
        let p = ply + 2 * i;
        if p > MAX_PLY_AHEAD {
            break;
        }
        total += survival(p) / here * phase_weight(p);
    }
    total.max(1.0)
}

fn shuffle_discount(rule50: i32, last_score: Option<Value>) -> f64 {
    let quiet = matches!(last_score, Some(s) if s.abs() <= QUIET_CP);
    if !quiet || rule50 <= SHUFFLE_FROM as i32 {
        return 1.0;
    }
    let t = ((rule50 as f64 - SHUFFLE_FROM) / (100.0 - SHUFFLE_FROM)).min(1.0);
    1.0 - (1.0 - SHUFFLE_MIN) * t
}
