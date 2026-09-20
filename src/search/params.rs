//! Search tuning constants. Values are ported from Apeiron (tuned against a 100cp
//! pawn) except where a term was infinite-board specific; those were rescaled to the
//! 8x8 board and are marked as such.

use crate::types::Value;

// Razoring margin, quadratic in depth (8x8 rescaled).
pub const RAZORING_QUAD: Value = 232;
pub const RAZORING_MAX_DEPTH: i32 = 8;

// Null move pruning. Apeiron's margin form was an infinite-board adaptation; the 8x8
// form is static_eval >= beta - mult*depth - improving_bonus + base.
pub const NMP_MIN_DEPTH: i32 = 3;
pub const NMP_BASE: Value = 175;
pub const NMP_DEPTH_MULT: Value = 6;
pub const NMP_IMPROVING: Value = 22;
pub const NMP_REDUCTION_BASE: i32 = 7;
pub const NMP_REDUCTION_DIV: i32 = 3;
pub const NMP_EVAL_DIV: Value = 123;
pub const NMP_VERIFY_MIN_DEPTH: i32 = 16;

// Late move reductions
pub const LMR_MIN_DEPTH: i32 = 3;
pub const LMR_MIN_MOVES: usize = 4;
pub const LMR_DIVISOR: f64 = 2.0;
pub const LMR_CUTOFF_THRESH: u8 = 2;
pub const LMR_TT_HISTORY_THRESH: i32 = -1000;

// History leaf pruning
pub const HLP_MAX_DEPTH: i32 = 3;
pub const HLP_MIN_MOVES: usize = 4;
pub const HLP_HISTORY_REDUCE: i32 = 300;
pub const HLP_HISTORY_LEAF: i32 = 0;

// Aspiration windows
pub const ASPIRATION_WINDOW: Value = 60;
pub const ASPIRATION_FAIL_MULT: Value = 4;
pub const ASPIRATION_MAX_WINDOW: Value = 1000;
pub const ASPIRATION_MAX_RETRIES: u32 = 4;

// Reverse futility pruning
pub const RFP_MAX_DEPTH: i32 = 14;
pub const RFP_SEEK_MATE_DEPTH: i32 = 6;
pub const RFP_MULT_TT: Value = 101;
pub const RFP_MULT_NO_TT: Value = 70;
pub const RFP_IMPROVING_MULT: Value = 2474;
pub const RFP_WORSENING_MULT: Value = 331;

// ProbCut
pub const PROBCUT_MARGIN: Value = 235;
pub const PROBCUT_IMPROVING: Value = 63;
pub const PROBCUT_MIN_DEPTH: i32 = 5;
pub const PROBCUT_DEPTH_SUB: i32 = 5;
pub const PROBCUT_DIVISOR: Value = 315;
pub const LOW_DEPTH_PROBCUT_MARGIN: Value = 800;

// Internal iterative reductions
pub const IIR_MIN_DEPTH: i32 = 3;
pub const IIR_REDUCTION: i32 = 2;

// SEE pruning
pub const SEE_CAPTURE_LINEAR: Value = 166;
pub const SEE_CAPTURE_HIST_DIV: i32 = 29;
// Quiet SEE pruning threshold, quadratic in the reduced depth (8x8 rescaled).
pub const SEE_QUIET_QUAD: Value = 11;
pub const SEE_WINNING_THRESHOLD: Value = 0;
/// Captures failing this SEE bound are deferred to the bad-capture stage.
pub const GOOD_CAPTURE_SEE: Value = -18;

// Quiet-move pruning inside the move loop
pub const HISTORY_PRUNE_MULT: i32 = 4083;
pub const QUIET_FUTILITY_MAX_LMR_DEPTH: i32 = 13;
pub const QUIET_FUTILITY_BASE: Value = 42;
pub const QUIET_FUTILITY_NO_BEST: Value = 161;
pub const QUIET_FUTILITY_PER_DEPTH: Value = 127;
pub const QUIET_HISTORY_DEPTH_DIV: i32 = 3208;

// Move ordering scores
pub const SORT_HASH: i32 = 6_000_000;
pub const SORT_WINNING_CAPTURE: i32 = 1_000_000;
pub const SORT_LOSING_CAPTURE: i32 = 0;
pub const SORT_QUIET: i32 = 0;
pub const SORT_KILLER1: i32 = 900_000;
pub const SORT_KILLER2: i32 = 800_000;
pub const SORT_COUNTERMOVE: i32 = 600_000;
/// Quiet checks that do not lose material are boosted by this much.
pub const SORT_CHECK_BONUS: i32 = 16384;
pub const SORT_CHECK_SEE: Value = -75;
/// Quiets scoring below this are tried after the bad captures.
pub const GOOD_QUIET_THRESHOLD: i32 = -14000;
/// Quiets below `-QUIET_SORT_LIMIT_PER_DEPTH * depth` are not sorted at all.
pub const QUIET_SORT_LIMIT_PER_DEPTH: i32 = 3560;
/// Depth from which the (more expensive) threat/escape ordering terms are used.
pub const THREAT_ORDERING_MIN_DEPTH: i32 = 4;

// History updates (gravity formula with a fixed maximum)
pub const HISTORY_BONUS_BASE: i32 = 300;
pub const HISTORY_BONUS_SUB: i32 = 250;
pub const HISTORY_BONUS_CAP: i32 = 1536;
pub const HISTORY_MAX: i32 = 16384;
pub const PAWN_HISTORY_BONUS_SCALE: i32 = 2;
pub const PAWN_HISTORY_MALUS_SCALE: i32 = 1;
/// Continuation-history weights for the move 1, 2 and 4 plies ago (out of 1024).
pub const CONT_WEIGHTS: [i32; 3] = [1024, 712, 410];
pub const CONT_OFFSETS: [usize; 3] = [1, 2, 4];

// Quiescence
pub const DELTA_MARGIN: Value = 200;
pub const MAX_QSEARCH_DEPTH: usize = 16;

// Singular extensions
pub const SE_MIN_DEPTH: i32 = 6;
pub const SE_TT_DEPTH_SUB: i32 = 3;
pub const SE_BETA_DEPTH_MULT: Value = 3;
pub const SE_TT_HISTORY_DIV: i32 = 150;

// Hindsight depth adjustment
pub const HINDSIGHT_EXTEND_REDUCTION: i32 = 3;
pub const HINDSIGHT_REDUCE_REDUCTION: i32 = 2;
pub const HINDSIGHT_REDUCE_EVAL: Value = 173;

// Correction history
pub const CORRHIST_SIZE: usize = 16384;
pub const CORRHIST_MASK: u64 = (CORRHIST_SIZE - 1) as u64;
pub const LASTMOVE_CORRHIST_SIZE: usize = 4096;
pub const CORRHIST_GRAIN: i32 = 256;
pub const CORRHIST_LIMIT: i32 = 1024 * 32;
pub const CORRHIST_WEIGHT_SCALE: i32 = 256;
pub const CORR_W_NONPAWN: i32 = 34;
pub const CORR_W_OPP_NONPAWN: i32 = 8;
pub const CORR_W_MINOR: i32 = 23;
pub const CORR_W_MATERIAL: i32 = 17;
pub const CORR_W_LASTMOVE: i32 = 18;

// LMR correction adjustment: how much the correction-history delta can trim a reduction.
pub const LMR_CORR_DIVISOR: i32 = 15185;

// Win-rate model, used only to report scores: the chance of winning from an internal
// score `v` is `1 / (1 + exp((a - v) / b))`, where `a` is the score that wins half the
// time and `b` sets how quickly that changes. Both move with the material on the board,
// because the same score converts far more often with a full board than in a sparse
// endgame. The constants describe the network and are refitted when it changes: by
// maximum likelihood over 684k eval/result pairs from 12000 of this engine's own games,
// binned into nine material bands. `a` runs from 366 in the sparsest to 274 with
// everything on, and a line in `m` reproduces all nine to within 13cp.

// Scaled by 1000 to keep this integer. `m` is the material count over 58, clamped to the
// range the fit covers.
pub const WDL_A_INTERCEPT: i32 = 411_111;
pub const WDL_A_SLOPE: i32 = -103_382;
pub const WDL_B_INTERCEPT: i32 = 74_879;
pub const WDL_B_SLOPE: i32 = 41_764;
pub const WDL_MATERIAL_ANCHOR: i32 = 58;
pub const WDL_MATERIAL_MIN: i32 = 17;
pub const WDL_MATERIAL_MAX: i32 = 78;

// Table sizes
pub const LOW_PLY_HISTORY_SIZE: usize = 4;
pub const PAWN_HISTORY_SIZE: usize = 2048;
pub const PAWN_HISTORY_MASK: u64 = (PAWN_HISTORY_SIZE - 1) as u64;

/// Standard depth-and-move-count reduction: 1 + ln(m) * ln(d) / divisor.
pub fn lmr_reduction(depth: i32, moves: usize) -> i32 {
    if depth <= 0 || moves == 0 {
        return 0;
    }
    (1.0 + (moves as f64).ln() * (depth as f64).ln() / LMR_DIVISOR) as i32
}
