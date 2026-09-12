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

    /// Allocates this move's budget from the clock and the measured horizon.
    pub fn init(limits: &Limits, us: Color, ply: i32, move_overhead: u64) -> TimeManager {
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

        // Horizon: how many moves this engine actually has left, not a fixed guess.
        let mr = if limits.movestogo > 0 {
            (limits.movestogo as f64).min(moves_remaining(ply))
        } else {
            moves_remaining(ply)
        };

        // Clock we can commit: what is on it, plus the increments we will earn over the
        // remaining moves, less the latency reserved for each of them.
        let budget = (time_f + inc * (mr - 1.0) - overhead * (mr + 1.0)).max(1.0);

        // An even share of that budget, nudged by URGENCY, and never more than a fixed
        // slice of the clock in one move.
        let optimum = (budget / mr * URGENCY).max(1.0).min(time_f * 0.35);
        // The hard ceiling allows one move to run long when the search is unsettled.
        let maximum = optimum.max((time_f * 0.8 - overhead).min(optimum * BURST));

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

/// Expected remaining moves at ply 0, 16, 32 ... 240, measured over 660 of this engine's
/// own games (median 128 plies). The curve falls to a minimum near ply 115 and then rises,
/// because the games still running that late are the long drawish ones; no decaying
/// formula expresses that, so the measured curve is tabulated and interpolated.
const MOVES_REMAINING: [f64; 16] = [
    71.6, 63.6, 55.7, 48.0,
    41.6, 35.9, 31.4, 29.4,
    29.8, 35.5, 39.9, 46.2,
    51.1, 53.4, 52.6, 48.8,
];

/// Scales the even share. Above 1.0 spends earlier, below 1.0 holds time back.
const URGENCY: f64 = 1.65;
/// How far one move may exceed its share before the hard ceiling stops it.
const BURST: f64 = 4.0;

/// Expected remaining moves at `ply`, interpolating the measured table.
fn moves_remaining(ply: i32) -> f64 {
    let x = (ply.max(0) as f64) / 16.0;
    let i = x.floor() as usize;
    if i + 1 >= MOVES_REMAINING.len() {
        return MOVES_REMAINING[MOVES_REMAINING.len() - 1];
    }
    let f = x - i as f64;
    MOVES_REMAINING[i] * (1.0 - f) + MOVES_REMAINING[i + 1] * f
}
