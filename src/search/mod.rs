//! Alpha-beta search: a port of Apeiron's negamax (TT cutoffs, razoring, RFP, null
//! move with verification, ProbCut, IIR, staged move picking, LMP/SEE/futility/history
//! pruning, singular and check extensions, LMR with hindsight adjustment, killer/
//! counter/continuation/pawn/low-ply/capture histories, correction history) driving a
//! root search with aspiration windows, MultiPV, stability-based time management and
//! Lazy SMP with vote-based move selection.

pub mod movepick;
pub mod params;
pub mod timeman;
pub mod tt;

use crate::eval::evaluate;
use crate::movegen;
use crate::position::{Position, piece_value};
use crate::types::*;
use movepick::{ContKey, MovePicker, sort_captures};
use params::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
pub use timeman::{Limits, TimeManager};
pub use tt::{TranspositionTable, value_from_tt, value_to_tt};

pub fn init() {
    crate::zobrist::init();
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    PV,
    Cut,
    All,
}

/// Allocates a zero-initialised box without touching the memory (calloc semantics).
fn zeroed_box<T>() -> Box<T> {
    unsafe {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    }
}

/// Continuation history sub-table: indexed by [piece][to].
pub type PieceToHistory = [[i16; 64]; PIECE_NB];

pub struct Histories {
    /// Butterfly history [color][from*64+to].
    pub main: Box<[[i32; 4096]; 2]>,
    /// Capture history [piece][to][captured type].
    pub capture: Box<[[[i32; PIECE_TYPE_NB]; 64]; PIECE_NB]>,
    /// Continuation history [in_check][capture][prev piece][prev to] -> PieceTo table.
    pub cont: Box<[[[[PieceToHistory; 64]; PIECE_NB]; 2]; 2]>,
    /// Pawn-structure history [pawn key][piece][to].
    pub pawn: Box<[[[i16; 64]; PIECE_NB]; PAWN_HISTORY_SIZE]>,
    /// Low-ply history [ply][from*64+to].
    pub low_ply: Box<[[i32; 4096]; LOW_PLY_HISTORY_SIZE]>,
    /// Refutation of the previous move, [prev piece][prev to].
    pub countermoves: Box<[[Move; 64]; PIECE_NB]>,
    pub nonpawn_corr: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub minor_corr: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub material_corr: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub lastmove_corr: Box<[i32; LASTMOVE_CORRHIST_SIZE]>,
    /// Running reliability of TT moves (positive: they tend to be best).
    pub tt_move_history: i32,
}

impl Histories {
    pub fn new() -> Histories {
        Histories {
            main: zeroed_box(),
            capture: zeroed_box(),
            cont: zeroed_box(),
            pawn: zeroed_box(),
            low_ply: zeroed_box(),
            countermoves: zeroed_box(),
            nonpawn_corr: zeroed_box(),
            minor_corr: zeroed_box(),
            material_corr: zeroed_box(),
            lastmove_corr: zeroed_box(),
            tt_move_history: 0,
        }
    }

    pub fn clear(&mut self) {
        *self = Histories::new();
    }

    #[inline(always)]
    pub fn cont(&self, k: ContKey) -> &PieceToHistory {
        &self.cont[k.0][k.1][k.2][k.3]
    }

    #[inline(always)]
    pub fn cont_mut(&mut self, k: ContKey) -> &mut PieceToHistory {
        &mut self.cont[k.0][k.1][k.2][k.3]
    }
}

impl Default for Histories {
    fn default() -> Self {
        Self::new()
    }
}

/// Gravity update: pulls the entry toward the bonus, saturating at HISTORY_MAX.
#[inline(always)]
fn gravity_i32(entry: &mut i32, bonus: i32) {
    let b = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
    *entry += b - (*entry * b.abs()) / HISTORY_MAX;
}

#[inline(always)]
fn gravity_i16(entry: &mut i16, bonus: i32) {
    let b = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
    let cur = *entry as i32;
    *entry = (cur + b - (cur * b.abs()) / HISTORY_MAX) as i16;
}

#[derive(Clone, Copy)]
pub struct StackEntry {
    /// Move made from this ply (NONE before any, NULL for a null move).
    pub current_move: Move,
    pub moved_piece: Piece,
    /// Whether the side to move at this ply was in check.
    pub in_check: bool,
    /// Whether `current_move` captured.
    pub is_capture: bool,
    pub static_eval: Value,
    pub stat_score: i32,
    pub cutoff_cnt: u8,
    pub tt_pv: bool,
    pub reduction: i32,
    pub killers: [Move; 2],
}

impl Default for StackEntry {
    fn default() -> Self {
        StackEntry {
            current_move: Move::NONE,
            moved_piece: Piece::NONE,
            in_check: false,
            is_capture: false,
            static_eval: 0,
            stat_score: 0,
            cutoff_cnt: 0,
            tt_pv: false,
            reduction: 0,
            killers: [Move::NONE; 2],
        }
    }
}

#[derive(Clone, Debug)]
pub struct RootMove {
    pub mv: Move,
    pub score: Value,
    pub prev_score: Value,
    pub pv: Vec<Move>,
    pub nodes: u64,
}

/// Per-thread search state. Histories persist across moves of a game.
pub struct Searcher {
    pub thread_id: usize,
    pub tt: Arc<TranspositionTable>,
    pub stop: Arc<AtomicBool>,
    /// Slot `thread_id` is refreshed with this thread's node count for aggregated NPS.
    pub node_counters: Arc<Vec<AtomicU64>>,
    pub hist: Box<Histories>,
    pub stack: Vec<StackEntry>,
    pv_table: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    pv_length: [usize; MAX_PLY + 1],

    pub nodes: u64,
    pub qnodes: u64,
    pub seldepth: usize,
    pub stopped: bool,
    pub limits: Limits,
    pub tm: TimeManager,
    pub silent: bool,
    pub multipv: usize,
    pub contempt: Value,
    pub chess960: bool,
    /// Skill level 0..=20; below 20 the final move is picked from several lines.
    pub skill_level: i32,
    /// Set by the GUI on `ponderhit`; converts a ponder search into a timed one.
    pub ponderhit: Arc<AtomicBool>,
    /// Time manager to install on ponderhit.
    pub ponder_tm: TimeManager,

    pub root_moves: Vec<RootMove>,
    pv_idx: usize,
    pub completed_depth: i32,
    pub best_move_root: Move,
    pub prev_score: Value,
    nmp_min_ply: usize,
    seek_mate: bool,
    /// Depth 1 must always complete before any time-based stop.
    min_depth_required: bool,

    // Dynamic time management state
    tot_best_move_changes: f64,
    best_move_changes: f64,
    best_move_nodes: u64,
    best_previous_average_score: Value,
    iter_values: [Value; 4],
    iter_idx: usize,
    prev_time_reduction: f64,
    last_best_move_depth: i32,
    /// Effective budget for this move after the dynamic factors, in ms.
    total_time_ms: f64,
    iter_start_ms: f64,
}

#[inline(always)]
fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 2) as Value
}

/// Scores are side-to-move relative and the root side moves at even ply, so the
/// sign flips with parity to make a draw cost us either way.
#[inline(always)]
fn draw_contempt(contempt: Value, ply: usize) -> Value {
    if ply.is_multiple_of(2) { -contempt } else { contempt }
}

/// Score of a node with no legal moves.
#[cfg(not(feature = "variants"))]
#[inline(always)]
fn terminal_value(_pos: &Position, in_check: bool, ply: usize) -> Value {
    if in_check { mated_in(ply) } else { VALUE_DRAW }
}
#[cfg(feature = "variants")]
#[inline(always)]
fn terminal_value(pos: &Position, in_check: bool, ply: usize) -> Value {
    match pos.variant() {
        // Being stalemated (or having nothing left to move) wins.
        Variant::Antichess => mate_in(ply),
        // No checkmate exists; a blocked side simply draws.
        Variant::RacingKings => VALUE_DRAW,
        _ => {
            if in_check { mated_in(ply) } else { VALUE_DRAW }
        }
    }
}

/// Variant game-over tests that do not depend on move generation.
#[cfg(not(feature = "variants"))]
#[inline(always)]
fn variant_terminal(_pos: &Position, _ply: usize) -> Option<Value> {
    None
}
#[cfg(feature = "variants")]
#[inline(always)]
fn variant_terminal(pos: &Position, ply: usize) -> Option<Value> {
    match pos.variant() {
        Variant::ThreeCheck if pos.checks_given(pos.side_to_move().flip()) >= 3 => Some(mated_in(ply)),
        Variant::RacingKings => pos.racing_kings_result(ply),
        Variant::KingOfTheHill if pos.pieces_cp(pos.side_to_move().flip(), PieceType::King) & crate::bitboard::CENTER != 0 => {
            Some(mated_in(ply))
        }
        _ => None,
    }
}

#[inline(always)]
fn history_bonus(depth: i32) -> i32 {
    (HISTORY_BONUS_BASE * depth - HISTORY_BONUS_SUB).min(HISTORY_BONUS_CAP)
}

/// Whether reported scores are normalised. Off reports the network's own units, which is
/// what the search works in.
pub static NORMALIZE_SCORE: AtomicBool = AtomicBool::new(true);
/// Whether to append `wdl` to the reported score.
pub static SHOW_WDL: AtomicBool = AtomicBool::new(false);

/// The win-rate model's parameters for this position, both times 1000.
fn win_rate_params(pos: &Position) -> (i32, i32) {
    let m = pos.material_count().clamp(WDL_MATERIAL_MIN, WDL_MATERIAL_MAX);
    let a = WDL_A_INTERCEPT + WDL_A_SLOPE * m / WDL_MATERIAL_ANCHOR;
    let b = WDL_B_INTERCEPT + WDL_B_SLOPE * m / WDL_MATERIAL_ANCHOR;
    (a.max(1000), b.max(1000))
}

/// Chance in a thousand of winning from `v`, per the fitted model.
fn win_permille(v: Value, a: i32, b: i32) -> i32 {
    let x = (a as f64 - 1000.0 * v as f64) / b as f64;
    (1000.0 / (1.0 + x.exp())).round() as i32
}

pub fn format_score(v: Value, pos: &Position) -> String {
    let score = if is_win(v) {
        format!("mate {}", (VALUE_MATE - v + 1) / 2)
    } else if is_loss(v) {
        format!("mate -{}", (VALUE_MATE + v + 1) / 2)
    } else if NORMALIZE_SCORE.load(Ordering::Relaxed) {
        // A shown +1.00 is the score this engine wins half the time from.
        let (a, _) = win_rate_params(pos);
        format!("cp {}", 100_000 * v / a)
    } else {
        format!("cp {}", v)
    };
    if !SHOW_WDL.load(Ordering::Relaxed) {
        return score;
    }
    let (a, b) = win_rate_params(pos);
    let (w, l) = if is_win(v) {
        (1000, 0)
    } else if is_loss(v) {
        (0, 1000)
    } else {
        (win_permille(v, a, b), win_permille(-v, a, b))
    };
    format!("{} wdl {} {} {}", score, w, 1000 - w - l, l)
}

impl Searcher {
    pub fn new(thread_id: usize, tt: Arc<TranspositionTable>, stop: Arc<AtomicBool>, node_counters: Arc<Vec<AtomicU64>>) -> Searcher {
        Searcher {
            thread_id,
            tt,
            stop,
            node_counters,
            hist: Box::new(Histories::new()),
            stack: vec![StackEntry::default(); MAX_PLY + 8],
            pv_table: zeroed_box(),
            pv_length: [0; MAX_PLY + 1],
            nodes: 0,
            qnodes: 0,
            seldepth: 0,
            stopped: false,
            limits: Limits::default(),
            tm: TimeManager::untimed(),
            silent: false,
            multipv: 1,
            contempt: DEFAULT_CONTEMPT,
            chess960: false,
            skill_level: 20,
            ponderhit: Arc::new(AtomicBool::new(false)),
            ponder_tm: TimeManager::untimed(),
            root_moves: Vec::new(),
            pv_idx: 0,
            completed_depth: 0,
            best_move_root: Move::NONE,
            prev_score: 0,
            nmp_min_ply: 0,
            seek_mate: false,
            min_depth_required: true,
            tot_best_move_changes: 0.0,
            best_move_changes: 0.0,
            best_move_nodes: 0,
            best_previous_average_score: 0,
            iter_values: [0; 4],
            iter_idx: 0,
            prev_time_reduction: 1.0,
            last_best_move_depth: 0,
            total_time_ms: 0.0,
            iter_start_ms: 0.0,
        }
    }

    /// Forgets everything learned (new game).
    pub fn clear(&mut self) {
        self.hist.clear();
        for e in self.stack.iter_mut() {
            *e = StackEntry::default();
        }
    }

    /// Per-search reset. Killers are position specific and start fresh; histories persist.
    pub fn new_search(&mut self, limits: Limits, tm: TimeManager) {
        self.limits = limits;
        // Pondering runs unlimited until the GUI confirms the predicted move.
        if self.limits.ponder {
            self.ponder_tm = tm;
            self.tm = TimeManager::untimed();
        } else {
            self.tm = tm;
        }
        self.nodes = 0;
        self.qnodes = 0;
        self.seldepth = 0;
        self.stopped = false;
        self.min_depth_required = true;
        self.tot_best_move_changes = 0.0;
        self.best_move_changes = 0.0;
        self.best_move_nodes = 0;
        self.best_previous_average_score = 0;
        self.seek_mate = false;
        self.iter_values = [0; 4];
        self.iter_idx = 0;
        self.prev_time_reduction = 1.0;
        self.last_best_move_depth = 0;
        self.total_time_ms = 0.0;
        self.iter_start_ms = 0.0;
        self.prev_score = 0;
        self.completed_depth = 0;
        self.best_move_root = Move::NONE;
        self.nmp_min_ply = 0;
        self.pv_idx = 0;
        for e in self.stack.iter_mut() {
            *e = StackEntry::default();
        }
        self.hist.tt_move_history = 0;
        // A small positive bias for moves not yet seen near the root.
        for row in self.hist.low_ply.iter_mut() {
            row.fill(97);
        }
    }

    // History helpers

    /// Continuation-history keys for the moves 1, 2 and 4 plies before `ply`.
    #[inline]
    pub fn cont_keys(&self, ply: usize) -> [Option<ContKey>; 3] {
        let mut keys = [None; 3];
        for (i, &off) in CONT_OFFSETS.iter().enumerate() {
            if ply >= off {
                let e = &self.stack[ply - off];
                if e.current_move.is_ok() && e.moved_piece.is_some() {
                    keys[i] = Some((e.in_check as usize, e.is_capture as usize, e.moved_piece.idx(), e.current_move.to_sq() as usize));
                }
            }
        }
        keys
    }

    #[inline(always)]
    pub fn pawn_hist(&self, pawn_key: u64, pc: Piece, to: usize) -> i32 {
        self.hist.pawn[(pawn_key & PAWN_HISTORY_MASK) as usize][pc.idx()][to] as i32
    }

    #[inline]
    fn update_pawn_history(&mut self, pawn_key: u64, pc: Piece, to: usize, bonus: i32) {
        gravity_i16(&mut self.hist.pawn[(pawn_key & PAWN_HISTORY_MASK) as usize][pc.idx()][to], bonus);
    }

    #[inline]
    fn update_main_history(&mut self, c: Color, m: Move, bonus: i32) {
        gravity_i32(&mut self.hist.main[c.idx()][m.from_to()], bonus);
    }

    #[inline]
    fn update_capture_history(&mut self, pc: Piece, to: usize, captured: PieceType, bonus: i32) {
        gravity_i32(&mut self.hist.capture[pc.idx()][to][captured.idx()], bonus);
    }

    #[inline]
    fn update_low_ply_history(&mut self, ply: usize, m: Move, bonus: i32) {
        if ply < LOW_PLY_HISTORY_SIZE {
            gravity_i32(&mut self.hist.low_ply[ply][m.from_to()], bonus);
        }
    }

    /// Updates the continuation histories of `pc`/`to` for the ancestors of `ply`.
    #[inline]
    fn update_continuation_histories(&mut self, ply: usize, in_check: bool, pc: Piece, to: usize, bonus: i32) {
        let keys = self.cont_keys(ply);
        for (i, key) in keys.iter().enumerate() {
            if in_check && CONT_OFFSETS[i] > 2 {
                break;
            }
            if let Some(k) = key {
                let adj = bonus * CONT_WEIGHTS[i] / 1024;
                gravity_i16(&mut self.hist.cont_mut(*k)[pc.idx()][to], adj);
            }
        }
    }

    #[inline]
    fn cont_history_sum(&self, keys: &[Option<ContKey>; 3], pc: Piece, to: usize, max_offset_idx: usize) -> i32 {
        let mut sum = 0;
        for k in keys.iter().take(max_offset_idx).flatten() {
            sum += self.hist.cont(*k)[pc.idx()][to] as i32;
        }
        sum
    }

    /// Butterfly index of the previous move for the last-move correction table.
    #[inline(always)]
    fn prev_move_idx(&self, ply: usize) -> usize {
        if ply == 0 {
            return 0;
        }
        let m = self.stack[ply - 1].current_move;
        if m.is_ok() { m.from_to() } else { 0 }
    }

    /// Static evaluation corrected by the learned search-vs-eval error.
    #[inline]
    pub fn adjusted_eval(&self, pos: &Position, raw: Value, prev_move_idx: usize) -> Value {
        let us = pos.side_to_move();
        let them = us.flip();
        let h = &self.hist;
        let nonpawn = h.nonpawn_corr[us.idx()][(pos.nonpawn_key(us) & CORRHIST_MASK) as usize];
        // The opponent's slot holds corrections learned with them to move: negate.
        let opp = -h.nonpawn_corr[them.idx()][(pos.nonpawn_key(them) & CORRHIST_MASK) as usize];
        let minor = h.minor_corr[us.idx()][(pos.minor_key() & CORRHIST_MASK) as usize];
        let mat = h.material_corr[us.idx()][(pos.material_key() & CORRHIST_MASK) as usize];
        let lm = h.lastmove_corr[prev_move_idx & (LASTMOVE_CORRHIST_SIZE - 1)];
        let total = (nonpawn * CORR_W_NONPAWN + opp * CORR_W_OPP_NONPAWN + minor * CORR_W_MINOR + mat * CORR_W_MATERIAL + lm * CORR_W_LASTMOVE)
            / (CORRHIST_GRAIN * 100);
        (raw + total).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX)
    }

    fn update_correction_history(&mut self, pos: &Position, depth: i32, static_eval: Value, score: Value, prev_move_idx: usize) {
        let us = pos.side_to_move().idx();
        let diff = score - static_eval;
        let weight = (depth * depth + 2 * depth + 1).clamp(1, 128) as i64;
        let scaled = (diff * CORRHIST_GRAIN) as i64;
        let blend = |entry: &mut i32, w: i64| {
            let v = (*entry as i64 * (CORRHIST_WEIGHT_SCALE as i64 - w) + scaled * w) / CORRHIST_WEIGHT_SCALE as i64;
            *entry = (v as i32).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
        };
        let np_idx = (pos.nonpawn_key(pos.side_to_move()) & CORRHIST_MASK) as usize;
        blend(&mut self.hist.nonpawn_corr[us][np_idx], weight);
        let mat_idx = (pos.material_key() & CORRHIST_MASK) as usize;
        blend(&mut self.hist.material_corr[us][mat_idx], weight);
        let minor_idx = (pos.minor_key() & CORRHIST_MASK) as usize;
        blend(&mut self.hist.minor_corr[us][minor_idx], weight);
        let lm_idx = prev_move_idx & (LASTMOVE_CORRHIST_SIZE - 1);
        blend(&mut self.hist.lastmove_corr[lm_idx], weight.min(64));
    }

    // Stack context

    #[inline(always)]
    fn set_ctx(&mut self, ply: usize, m: Move, pc: Piece, in_check: bool, is_capture: bool) {
        let e = &mut self.stack[ply];
        e.current_move = m;
        e.moved_piece = pc;
        e.in_check = in_check;
        e.is_capture = is_capture;
    }

    #[inline(always)]
    fn set_null_ctx(&mut self, ply: usize) {
        let e = &mut self.stack[ply];
        e.current_move = Move::NULL;
        e.moved_piece = Piece::NONE;
        e.in_check = false;
        e.is_capture = false;
    }

    #[inline]
    fn update_pv(&mut self, ply: usize, m: Move) {
        let child_len = self.pv_length[ply + 1];
        let mut tmp = [Move::NONE; MAX_PLY + 1];
        tmp[..child_len].copy_from_slice(&self.pv_table[ply + 1][..child_len]);
        self.pv_table[ply][0] = m;
        self.pv_table[ply][1..=child_len].copy_from_slice(&tmp[..child_len]);
        self.pv_length[ply] = child_len + 1;
    }

    /// Geometric shuffle detection: this move undoes the ply-2 move which undid ply-4.
    fn is_shuffling(&self, pos: &Position, m: Move, ply: usize, is_capture: bool) -> bool {
        if pos.moved_piece(m).piece_type() == PieceType::Pawn || is_capture || pos.rule50_count() < 10 {
            return false;
        }
        if pos.plies_from_null() <= 6 || ply < 20 {
            return false;
        }
        let m2 = self.stack[ply - 2].current_move;
        let m4 = self.stack[ply - 4].current_move;
        m2.is_ok() && m4.is_ok() && m.from_sq() == m2.to_sq() && m2.from_sq() == m4.to_sq()
    }

    /// Polls the stop flag and clock every 4096 nodes. Returns true once stopped.
    #[inline]
    pub fn check_time(&mut self) -> bool {
        if self.nodes & 4095 != 0 {
            return self.stopped;
        }
        self.node_counters[self.thread_id].store(self.nodes, Ordering::Relaxed);
        if self.stop.load(Ordering::Relaxed) {
            self.stopped = true;
            return true;
        }
        if self.thread_id != 0 {
            return self.stopped;
        }
        self.poll_ponderhit();
        if self.limits.nodes > 0 && self.total_nodes() >= self.limits.nodes {
            self.stopped = true;
            return true;
        }
        if !self.tm.timed || self.min_depth_required {
            return self.stopped;
        }
        let elapsed = self.tm.elapsed_ms() as f64;
        let hard = self.tm.maximum as f64;
        if elapsed >= hard {
            self.stopped = true;
            return true;
        }
        // Last-resort: stop if the next poll would land past the hard limit.
        if self.nodes > 4096 {
            let per_poll = 4096.0 * elapsed / self.nodes as f64;
            if elapsed + per_poll > hard {
                self.stopped = true;
                return true;
            }
        }
        // A hard-limited iteration that alone has eaten half the budget won't finish.
        if !self.tm.soft && self.total_time_ms > 0.0 && elapsed - self.iter_start_ms > self.total_time_ms * 0.5 {
            self.stopped = true;
            return true;
        }
        self.stopped
    }

    /// Switches from pondering to the real clock once the GUI sends `ponderhit`.
    #[inline]
    fn poll_ponderhit(&mut self) {
        if self.limits.ponder && self.ponderhit.load(Ordering::Relaxed) {
            self.limits.ponder = false;
            self.tm = self.ponder_tm.clone();
            self.tm.start = std::time::Instant::now();
            self.iter_start_ms = 0.0;
            self.total_time_ms = 0.0;
        }
    }

    pub fn total_nodes(&self) -> u64 {
        self.node_counters.iter().map(|c| c.load(Ordering::Relaxed)).sum::<u64>().max(self.nodes)
    }

    // Iterative deepening

    /// Runs the whole search for this thread. Results are left in `root_moves`,
    /// `best_move_root`, `prev_score` and `completed_depth`.
    pub fn iterative_deepening(&mut self, pos: &mut Position) {
        self.root_moves.clear();
        let mut legal = MoveList::new();
        movegen::generate_legal(pos, &mut legal);
        for &m in legal.iter() {
            if !self.limits.searchmoves.is_empty() && !self.limits.searchmoves.contains(&m) {
                continue;
            }
            self.root_moves.push(RootMove {
                mv: m,
                score: -VALUE_INFINITE,
                prev_score: -VALUE_INFINITE,
                pv: vec![m],
                nodes: 0,
            });
        }
        if self.root_moves.is_empty() {
            return;
        }
        self.best_move_root = self.root_moves[0].mv;

        let max_depth = if self.limits.depth > 0 { self.limits.depth.min(MAX_PLY as i32 - 1) } else { MAX_PLY as i32 - 1 };
        // Reduced strength picks among several lines, so search at least four.
        let want = if self.skill_level < 20 { self.multipv.max(4) } else { self.multipv };
        let multipv = want.clamp(1, self.root_moves.len());

        // A forced move under a clock needs no deliberation.
        if self.root_moves.len() == 1 && self.thread_id == 0 && self.tm.timed && !self.limits.infinite && self.limits.depth == 0 {
            let m = self.root_moves[0].mv;
            let raw = evaluate(pos);
            let score = self.adjusted_eval(pos, raw, 0);
            self.root_moves[0].score = score;
            self.prev_score = score;
            self.completed_depth = 1;
            self.print_info(pos, 1);
            self.stop.store(true, Ordering::Relaxed);
            let _ = m;
            return;
        }

        // Helpers with odd ids run one depth ahead so the threads diverge.
        let odd_helper = self.thread_id % 2 == 1;
        let start_depth = if odd_helper { 2.min(max_depth) } else { 1 };
        let mut prev_best = Move::NONE;

        for base_depth in start_depth..=max_depth {
            let depth = if odd_helper { (base_depth + 1).min(max_depth) } else { base_depth };
            self.seldepth = 0;
            self.pv_length = [0; MAX_PLY + 1];
            self.poll_ponderhit();
            self.iter_start_ms = self.tm.elapsed_ms() as f64;
            self.tot_best_move_changes /= 2.0;

            if self.thread_id == 0 && self.tm.timed && !self.min_depth_required {
                let elapsed = self.tm.elapsed_ms() as f64;
                if elapsed >= self.tm.maximum as f64 {
                    self.stopped = true;
                    break;
                }
                // Don't start a depth that cannot finish: hard limits are conservative,
                // fixed budgets are pushed much closer to the wall.
                let threshold = if self.tm.soft { 0.90 } else { 0.50 };
                if self.total_time_ms > 0.0 && elapsed > self.total_time_ms * threshold {
                    break;
                }
            }

            for rm in self.root_moves.iter_mut() {
                rm.prev_score = rm.score;
            }

            for pv_idx in 0..multipv {
                self.pv_idx = pv_idx;
                let prev = self.root_moves[pv_idx].prev_score;
                let mut score;
                if depth <= 1 || prev == -VALUE_INFINITE {
                    score = self.search_root(pos, depth, -VALUE_INFINITE, VALUE_INFINITE);
                } else {
                    let mut window = ASPIRATION_WINDOW;
                    let mut alpha = (prev - window).max(-VALUE_INFINITE);
                    let mut beta = (prev + window).min(VALUE_INFINITE);
                    let mut retries = 0;
                    loop {
                        score = self.search_root(pos, depth, alpha, beta);
                        retries += 1;
                        if self.stopped {
                            break;
                        }
                        if score <= alpha {
                            window *= ASPIRATION_FAIL_MULT;
                            alpha = (prev - window).max(-VALUE_INFINITE);
                        } else if score >= beta {
                            window *= ASPIRATION_FAIL_MULT;
                            beta = (prev + window).min(VALUE_INFINITE);
                        } else {
                            break;
                        }
                        if window > ASPIRATION_MAX_WINDOW || retries >= ASPIRATION_MAX_RETRIES {
                            score = self.search_root(pos, depth, -VALUE_INFINITE, VALUE_INFINITE);
                            break;
                        }
                    }
                }
                if self.stopped {
                    break;
                }
                // Completed lines are kept in score order; a later line may outscore an
                // earlier one because the earlier pass only bounded it.
                self.root_moves[..=pv_idx].sort_by(|a, b| b.score.cmp(&a.score));
                let _ = score;
            }

            if base_depth == start_depth {
                self.min_depth_required = false;
            }

            if self.stopped {
                break;
            }

            self.completed_depth = depth;
            self.prev_score = self.root_moves[0].score;
            let best = self.root_moves[0].mv;
            self.best_move_root = best;
            if prev_best.is_some() && prev_best != best {
                self.best_move_changes += 1.0;
                self.last_best_move_depth = depth;
            }
            prev_best = best;

            if self.thread_id == 0 && !self.silent {
                self.print_info(pos, depth);
            }

            if self.stop.load(Ordering::Relaxed) {
                self.stopped = true;
                break;
            }

            let best_score = self.root_moves[0].score;
            if self.limits.mate > 0 && is_win(best_score) && (VALUE_MATE - best_score + 1) / 2 <= self.limits.mate {
                break;
            }

            // Mate hunting is left to the search unless we are under a clock.
            if self.thread_id == 0 && self.tm.timed && !self.limits.infinite {
                let mate_shortcut = best_score >= mate_in(3) || best_score == mated_in(2);
                if mate_shortcut {
                    break;
                }
                self.seek_mate = base_depth >= 16 && best_score.abs() >= 4000;
                if self.dynamic_time_check(depth, best_score) {
                    break;
                }
            }
        }
    }

    /// Stability factors scaling the optimum time. Returns true when
    /// the search should not continue to the next depth.
    fn dynamic_time_check(&mut self, depth: i32, best_score: Value) -> bool {
        let elapsed = self.tm.elapsed_ms() as f64;

        let nodes_effort = if self.nodes > 0 { self.best_move_nodes as f64 * 100000.0 / self.nodes as f64 } else { 0.0 };
        let high_best_move_effort = if nodes_effort >= 93340.0 { 0.76 } else { 1.0 };

        self.tot_best_move_changes += self.best_move_changes;
        self.best_move_changes = 0.0;

        let iter_val = self.iter_values[self.iter_idx];
        let prev_avg = self.best_previous_average_score;
        let falling_eval = ((11.85 + 2.24 * (prev_avg - best_score) as f64 + 0.93 * (iter_val - best_score) as f64) / 100.0).clamp(0.57, 1.70);

        let k = 0.51;
        let center = self.last_best_move_depth as f64 + 12.15;
        let time_reduction = 0.66 + 0.85 / (0.98 + (-k * (depth as f64 - center)).exp());
        let reduction = (1.43 + self.prev_time_reduction) / (2.28 * time_reduction);
        let instability = (1.02 + 2.14 * self.tot_best_move_changes).min(2.5);

        let mut factors = (falling_eval * reduction * instability * high_best_move_effort).clamp(0.5, 2.5);
        if self.tm.soft {
            factors = factors.max(0.98);
        }
        let total = self.tm.optimum as f64 * factors;
        let effective = total.min(self.tm.maximum as f64);
        self.total_time_ms = effective;

        let stop = elapsed > effective;

        self.iter_values[self.iter_idx] = best_score;
        self.iter_idx = (self.iter_idx + 1) & 3;
        self.best_previous_average_score =
            if self.best_previous_average_score == 0 { best_score } else { (best_score + self.best_previous_average_score) / 2 };
        self.prev_time_reduction = time_reduction;
        stop
    }

    /// Validates a PV (truncating at the first illegal move) and extends it with TT
    /// moves toward `target_len`. Display only.
    fn extend_pv_with_tt(&self, pos: &Position, pv: &mut Vec<Move>, target_len: usize) {
        let mut p = pos.clone();
        let mut seen: Vec<u64> = Vec::with_capacity(target_len.max(pv.len()));
        let mut valid = 0;
        for &m in pv.iter() {
            if !p.pseudo_legal(m) || !p.legal(m) {
                break;
            }
            seen.push(p.key());
            let gc = p.gives_check(m);
            p.make_move(m, gc);
            valid += 1;
        }
        pv.truncate(valid);
        while pv.len() < target_len {
            let key = p.key();
            if seen.contains(&key) {
                break;
            }
            seen.push(key);
            let m = self.tt.probe_move(key);
            if m.is_none() || !p.pseudo_legal(m) || !p.legal(m) {
                break;
            }
            let gc = p.gives_check(m);
            p.make_move(m, gc);
            pv.push(m);
        }
    }

    fn print_info(&self, pos: &Position, depth: i32) {
        let elapsed = self.tm.elapsed_ms().max(1);
        let nodes = self.total_nodes();
        let nps = nodes * 1000 / elapsed;
        let hashfull = self.tt.hashfull();
        let want = if self.skill_level < 20 { self.multipv.max(4) } else { self.multipv };
        let multipv = want.clamp(1, self.root_moves.len());
        for i in 0..multipv {
            let rm = &self.root_moves[i];
            if rm.score == -VALUE_INFINITE {
                continue;
            }
            let mut pv = rm.pv.clone();
            self.extend_pv_with_tt(pos, &mut pv, depth as usize);
            let pv_str: Vec<String> = pv.iter().map(|m| pos.move_to_uci(*m)).collect();
            println!(
                "info depth {} seldepth {} multipv {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                depth,
                self.seldepth.max(depth as usize),
                i + 1,
                format_score(rm.score, pos),
                nodes,
                nps,
                hashfull,
                elapsed,
                pv_str.join(" ")
            );
        }
    }

    // Root search

    /// PVS over `root_moves[pv_idx..]`; updates their scores and PVs.
    fn search_root(&mut self, pos: &mut Position, depth: i32, mut alpha: Value, beta: Value) -> Value {
        let alpha_orig = alpha;
        self.pv_length[0] = 0;
        self.stack[2].cutoff_cnt = 0;
        self.stack[2].stat_score = 0;
        self.stack[4].stat_score = 0;
        self.stack[0].reduction = 0;

        let key = pos.key();
        let tt_move = self.tt.probe(key).map_or(Move::NONE, |d| d.mv);
        let in_check = pos.in_check();
        self.stack[0].static_eval = if in_check {
            0
        } else {
            let raw = evaluate(pos);
            self.adjusted_eval(pos, raw, 0)
        };
        self.order_root_moves(pos, tt_move);

        let mut best_score = -VALUE_INFINITE;
        let mut best_move = Move::NONE;
        let mut legal = 0usize;
        let pv_idx = self.pv_idx;

        for idx in pv_idx..self.root_moves.len() {
            let m = self.root_moves[idx].mv;
            let nodes_before = self.nodes;
            if self.thread_id == 0 && !self.silent && self.tm.elapsed_ms() > 3000 {
                println!("info depth {} currmove {} currmovenumber {}", depth, pos.move_to_uci(m), idx + 1);
            }
            let pc = pos.moved_piece(m);
            let is_capture = pos.is_capture(m);
            let gives_check = pos.gives_check(m);
            self.set_ctx(0, m, pc, in_check, is_capture);
            self.stack[0].stat_score = self.hist.main[pos.side_to_move().idx()][m.from_to()];
            pos.make_move(m, gives_check);
            legal += 1;

            let score = if legal == 1 {
                -self.negamax(pos, depth - 1, 1, -beta, -alpha, NodeType::PV, true, Move::NONE)
            } else {
                let mut s = -self.negamax(pos, depth - 1, 1, -alpha - 1, -alpha, NodeType::Cut, true, Move::NONE);
                if s > alpha && s < beta && !self.stopped {
                    s = -self.negamax(pos, depth - 1, 1, -beta, -alpha, NodeType::PV, true, Move::NONE);
                }
                s
            };
            pos.unmake_move(m);
            self.root_moves[idx].nodes += self.nodes - nodes_before;

            if self.stopped {
                return best_score;
            }

            if legal == 1 || score > alpha {
                self.root_moves[idx].score = score;
                self.update_pv(0, m);
                let len = self.pv_length[0];
                self.root_moves[idx].pv = self.pv_table[0][..len].to_vec();
                if idx > pv_idx {
                    // A new best line: move it to the front of the unsearched block.
                    self.root_moves[pv_idx..=idx].rotate_right(1);
                }
                if score > alpha {
                    alpha = score;
                }
                if self.pv_idx == 0 {
                    self.best_move_root = self.root_moves[pv_idx].mv;
                }
            } else {
                // Fail-low bound: keep the previous exact score for ordering.
                self.root_moves[idx].score = -VALUE_INFINITE;
            }
            if score > best_score {
                best_score = score;
                best_move = m;
            }
            if idx == pv_idx {
                self.best_move_nodes += self.nodes - nodes_before;
            }
            if alpha >= beta {
                break;
            }
        }

        if legal == 0 {
            return if in_check { mated_in(0) } else { VALUE_DRAW };
        }

        let bound = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        if self.pv_idx == 0 {
            let tt_best = if self.pv_length[0] > 0 { self.pv_table[0][0] } else { best_move };
            self.tt.store(key, depth, bound, true, value_to_tt(best_score, 0), VALUE_NONE, tt_best);
        }
        best_score
    }

    /// Orders `root_moves[pv_idx..]`: the previous PV/TT move first, captures by
    /// MVV-LVA+SEE, quiets by history (previous-iteration scores are not used: only the
    /// PV move has a real score, the rest are noisy fail-low bounds).
    fn order_root_moves(&mut self, pos: &Position, tt_move: Move) {
        let pv_idx = self.pv_idx;
        let prev_pv = if self.completed_depth > 0 || self.root_moves[pv_idx].prev_score != -VALUE_INFINITE {
            self.root_moves[pv_idx].mv
        } else {
            Move::NONE
        };
        let mut scored: Vec<(i32, usize)> = Vec::with_capacity(self.root_moves.len());
        for (i, rm) in self.root_moves.iter().enumerate().skip(pv_idx) {
            let m = rm.mv;
            let s = if m == prev_pv {
                SORT_HASH + 1
            } else if m == tt_move {
                SORT_HASH
            } else if pos.is_capture_stage(m) {
                let victim = pos.captured_type(m).map_or(0, piece_value);
                let attacker = piece_value(pos.moved_piece(m).piece_type());
                let promo = if m.is_promotion() { piece_value(m.promotion()) - piece_value(PieceType::Pawn) } else { 0 };
                let base = (victim + promo) * 10 - attacker;
                if pos.see_ge(m, SEE_WINNING_THRESHOLD) { base + SORT_WINNING_CAPTURE } else { base + SORT_LOSING_CAPTURE }
            } else {
                let pc = pos.moved_piece(m);
                2 * self.hist.main[pc.color().idx()][m.from_to()] + 2 * self.pawn_hist(pos.pawn_key(), pc, m.to_sq() as usize)
                    + 8 * self.hist.low_ply[0][m.from_to()]
            };
            scored.push((s, i));
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let reordered: Vec<RootMove> = scored.iter().map(|&(_, i)| self.root_moves[i].clone()).collect();
        for (k, rm) in reordered.into_iter().enumerate() {
            self.root_moves[pv_idx + k] = rm;
        }
    }

    // Main search

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &mut Position,
        depth: i32,
        ply: usize,
        alpha: Value,
        beta: Value,
        node_type: NodeType,
        allow_null: bool,
        excluded: Move,
    ) -> Value {
        let mut alpha = alpha;
        let mut beta = beta;
        let mut depth = depth;
        let is_pv = node_type == NodeType::PV;
        let cut_node = node_type == NodeType::Cut;
        let all_node = node_type == NodeType::All;

        // Cleared before any early return: a parent copies pv_table[ply] after the
        // child returns, and a stale length would splice in a line from another node.
        self.pv_length[ply] = 0;

        if depth <= 0 {
            return self.qsearch(pos, ply, 0, alpha, beta, node_type);
        }
        depth = depth.min(MAX_PLY as i32 - 1);

        if ply >= MAX_PLY - 1 {
            let idx = self.prev_move_idx(ply);
            let raw = evaluate(pos);
            return self.adjusted_eval(pos, raw, idx);
        }

        // A reversible move that repeats a position is always available as a draw.
        if alpha < VALUE_DRAW && pos.upcoming_repetition(ply) {
            let draw_val = value_draw(self.nodes) + draw_contempt(self.contempt, ply);
            if draw_val >= beta {
                return draw_val;
            }
            alpha = alpha.max(draw_val);
        }

        let in_check = pos.in_check();
        self.nodes += 1;
        if ply + 2 < MAX_PLY {
            self.stack[ply + 2].cutoff_cnt = 0;
            self.stack[ply + 2].stat_score = 0;
        }
        if ply + 4 < MAX_PLY {
            self.stack[ply + 4].stat_score = 0;
        }
        if ply + 1 < self.stack.len() {
            self.stack[ply + 1].killers = [Move::NONE; 2];
        }

        if self.check_time() {
            return 0;
        }
        if is_pv && ply > self.seldepth {
            self.seldepth = ply;
        }

        if pos.is_draw(ply) {
            return value_draw(self.nodes) + draw_contempt(self.contempt, ply);
        }
        if let Some(v) = variant_terminal(pos, ply) {
            return v;
        }
        // Mate distance pruning
        alpha = alpha.max(mated_in(ply));
        beta = beta.min(mate_in(ply + 1));
        if alpha >= beta {
            return alpha;
        }

        let alpha_orig = alpha;
        let beta_orig = beta;
        let us = pos.side_to_move();

        let prior_reduction = self.stack[ply - 1].reduction;
        self.stack[ply - 1].reduction = 0;

        // Transposition table
        let key = pos.key();
        let rule50 = pos.rule50_count();
        let tt_entry = self.tt.probe(key);
        let tt_hit = tt_entry.is_some();
        let (tt_move, tt_value, tt_eval, tt_depth, tt_was_pv, tt_bound) = match tt_entry {
            Some(d) => (d.mv, value_from_tt(d.score, ply, rule50), d.eval, d.depth, d.is_pv, d.bound),
            None => (Move::NONE, VALUE_NONE, VALUE_NONE, 0, false, Bound::None),
        };
        let tt_move = if tt_move.is_some() && pos.pseudo_legal(tt_move) { tt_move } else { Move::NONE };
        let tt_capture = tt_move.is_some() && pos.is_capture_stage(tt_move);

        let prev_move_idx = self.prev_move_idx(ply);

        // Static evaluation
        let (mut static_eval, raw_eval) = if in_check {
            let prev = if ply >= 2 { self.stack[ply - 2].static_eval } else { 0 };
            (prev, prev)
        } else {
            let mut raw = tt_eval;
            if raw == VALUE_NONE {
                raw = evaluate(pos);
                self.tt.store(key, 0, Bound::None, tt_was_pv, VALUE_NONE, raw, tt_move);
            }
            (self.adjusted_eval(pos, raw, prev_move_idx), raw)
        };

        // Evaluation smoothing with the parent move's history score.
        static_eval += -self.stack[ply - 1].stat_score / 512;
        self.stack[ply].static_eval = static_eval;

        let mut improving = if in_check {
            false
        } else if ply >= 2 {
            static_eval > self.stack[ply - 2].static_eval
        } else {
            true
        };
        let opponent_worsening = !in_check && static_eval > -self.stack[ply - 1].static_eval;

        // Refine the eval with a TT bound in the right direction.
        let mut eval = static_eval;
        if excluded.is_none() && tt_hit && tt_value != VALUE_NONE {
            let better = if tt_value > eval { tt_bound.has_lower() } else { tt_bound.has_upper() };
            if better {
                eval = tt_value;
            }
        }

        // Hindsight depth adjustment from the parent's reduction.
        if !in_check {
            let prev_eval = self.stack[ply - 1].static_eval;
            if prior_reduction >= HINDSIGHT_EXTEND_REDUCTION && !opponent_worsening {
                depth += 1;
            }
            if prior_reduction >= HINDSIGHT_REDUCE_REDUCTION && depth >= 2 && static_eval + prev_eval > HINDSIGHT_REDUCE_EVAL {
                depth -= 1;
            }
        }

        // TT cutoff (non-PV)
        if !is_pv && excluded.is_none() && tt_hit && tt_value != VALUE_NONE {
            let depth_threshold = if tt_value <= beta { depth - 1 } else { depth };
            let depth_ok = tt_depth > depth_threshold;
            let fails_high = tt_value >= beta;
            let bound_ok = if fails_high { tt_bound.has_lower() } else { tt_bound.has_upper() };
            let node_ok = (cut_node == fails_high) || depth > 5;
            let rule50_ok = rule50 < 96;
            // is_repetition(ply) is not re-checked here: pos.is_draw(ply) above already
            // tested it with this same ply and would have returned if true.
            if depth_ok && bound_ok && node_ok && rule50_ok {
                return tt_value;
            }
            // Deep enough but holding the opposite bound: shave a ply so a real search
            // replaces it.
            let opposite = if fails_high { tt_bound.has_upper() } else { tt_bound.has_lower() };
            if depth > 5 && depth_ok && tt_bound != Bound::Exact && opposite {
                self.tt.penalize(key, 1);
            }
        }

        let mut tt_pv = is_pv || (tt_hit && tt_was_pv);
        self.stack[ply].tt_pv = tt_pv;

        if !in_check {
            // Razoring
            if !is_pv && depth <= RAZORING_MAX_DEPTH && eval < alpha - RAZORING_QUAD * depth * depth {
                return self.qsearch(pos, ply, 0, alpha, beta, node_type);
            }

            // Reverse futility pruning
            let rfp_cap = if self.seek_mate { RFP_SEEK_MATE_DEPTH } else { RFP_MAX_DEPTH };
            if !tt_pv && depth < rfp_cap && (tt_move.is_none() || tt_capture) && !is_loss(beta) && !is_win(eval) {
                let mult = if tt_hit { RFP_MULT_TT } else { RFP_MULT_NO_TT };
                let mut bonus = 0;
                if improving {
                    bonus += RFP_IMPROVING_MULT * mult / 1024;
                }
                if opponent_worsening {
                    bonus += RFP_WORSENING_MULT * mult / 1024;
                }
                let margin = mult * depth - bonus;
                if eval - margin >= beta && eval >= beta {
                    return (2 * beta + eval) / 3;
                }
            }

            // Null move pruning with verification at high depth
            if !is_pv
                && allow_null
                && excluded.is_none()
                && pos.variant().allows_null_move()
                && depth >= NMP_MIN_DEPTH
                && !is_loss(beta)
                && ply >= self.nmp_min_ply
                && static_eval >= beta - NMP_DEPTH_MULT * depth - NMP_IMPROVING * improving as Value + NMP_BASE
                && pos.has_non_pawn_material(us)
            {
                let r = (NMP_REDUCTION_BASE + depth / NMP_REDUCTION_DIV + ((static_eval - beta) / NMP_EVAL_DIV).max(0)).min(depth);
                self.set_null_ctx(ply);
                self.stack[ply].reduction = 0;
                self.stack[ply].stat_score = 0;
                pos.make_null_move();
                let null_score = -self.negamax(pos, depth - r, ply + 1, -beta, -beta + 1, NodeType::Cut, false, Move::NONE);
                pos.unmake_null_move();
                if self.stopped {
                    return 0;
                }
                if null_score >= beta && !is_win(null_score) {
                    if self.nmp_min_ply != 0 || depth < NMP_VERIFY_MIN_DEPTH {
                        return null_score;
                    }
                    // Verification: disable null moves for a few plies and re-search.
                    self.nmp_min_ply = ply + (3 * (depth - r) / 4) as usize;
                    let v = self.negamax(pos, depth - r, ply, beta - 1, beta, NodeType::All, false, Move::NONE);
                    self.nmp_min_ply = 0;
                    if self.stopped {
                        return 0;
                    }
                    if v >= beta {
                        return null_score;
                    }
                }
            }

            improving = improving || static_eval >= beta;

            // Internal iterative reductions
            if depth >= IIR_MIN_DEPTH && tt_move.is_none() {
                depth -= IIR_REDUCTION;
            }
        }

        // ProbCut: a good capture whose reduced search stays well above beta.
        let prob_cut_beta = beta + PROBCUT_MARGIN - if improving { PROBCUT_IMPROVING } else { 0 };
        if !is_pv
            && !in_check
            && excluded.is_none()
            && depth >= PROBCUT_MIN_DEPTH
            && !is_decisive(beta)
            && (tt_value == VALUE_NONE || tt_value >= prob_cut_beta)
        {
            let prob_cut_depth = (depth - PROBCUT_DEPTH_SUB - (static_eval - beta) / PROBCUT_DIVISOR).clamp(0, depth);
            let threshold = prob_cut_beta - static_eval;
            let mut picker = MovePicker::new_probcut(tt_move, threshold, self, pos);
            while let Some(m) = picker.next(pos, self) {
                if m == excluded || !pos.legal(m) {
                    continue;
                }
                let pc = pos.moved_piece(m);
                let is_capture = pos.is_capture(m);
                let gives_check = pos.gives_check(m);
                self.set_ctx(ply, m, pc, in_check, is_capture);
                self.stack[ply].reduction = 0;
                self.stack[ply].stat_score = self.hist.main[us.idx()][m.from_to()];
                pos.make_move(m, gives_check);

                let mut val = -self.qsearch(pos, ply + 1, 0, -prob_cut_beta, -prob_cut_beta + 1, NodeType::Cut);
                if val >= prob_cut_beta && prob_cut_depth > 0 {
                    val = -self.negamax(pos, prob_cut_depth, ply + 1, -prob_cut_beta, -prob_cut_beta + 1, NodeType::Cut, true, Move::NONE);
                }
                pos.unmake_move(m);
                if self.stopped {
                    return 0;
                }
                if val >= prob_cut_beta {
                    self.tt.store(key, prob_cut_depth + 1, Bound::Lower, false, value_to_tt(val, ply), raw_eval, m);
                    if !is_decisive(val) {
                        return val - (prob_cut_beta - beta);
                    }
                }
            }
        }

        // Small ProbCut: a stored lower bound far above beta is trusted at once.
        if tt_hit
            && tt_value != VALUE_NONE
            && tt_bound.has_lower()
            && tt_depth >= depth - 4
            && tt_value >= beta + LOW_DEPTH_PROBCUT_MARGIN
            && !is_decisive(beta)
            && !is_decisive(tt_value)
        {
            return beta + LOW_DEPTH_PROBCUT_MARGIN;
        }

        // Singular-extension eligibility, evaluated when the TT move is reached.
        let se_conditions: Option<(Value, i32)> = if depth >= SE_MIN_DEPTH
            && !in_check
            && tt_move.is_some()
            && !self.seek_mate
            && tt_hit
            && tt_bound.has_lower()
            && tt_depth >= depth - SE_TT_DEPTH_SUB
            && tt_value != VALUE_NONE
            && !is_decisive(tt_value)
        {
            Some((tt_value, (depth - 1) / 2))
        } else {
            None
        };

        let mut picker = MovePicker::new(tt_move, ply, depth, self, pos);
        let cont_keys = picker.cont_keys;
        let pawn_key = pos.pawn_key();
        let has_non_pawn = pos.has_non_pawn_material(us);

        let mut best_score = -VALUE_INFINITE;
        let mut best_move = Move::NONE;
        // Only a move that raised alpha, stored in the TT: best_move can be the least-bad
        // of a set of fail-low bounds, which is not a claim the position is this good.
        let mut tt_best_move = Move::NONE;
        let mut legal_moves = 0usize;
        let mut quiets_searched = MoveList::new();
        let mut captures_searched: [(Move, PieceType); 32] = [(Move::NONE, PieceType::Pawn); 32];
        let mut captures_count = 0usize;

        while let Some(m) = picker.next(pos, self) {
            if m == excluded || !pos.legal(m) {
                continue;
            }
            // Counted before pruning: a node whose every move is
            // pruned must not read as mate or stalemate.
            legal_moves += 1;
            let pc = pos.moved_piece(m);
            let p_type = pc.piece_type();
            let captured_type = pos.captured_type(m);
            let is_capture = captured_type.is_some();
            let is_promotion = m.is_promotion();
            let gives_check = pos.gives_check(m);
            let to = m.to_sq() as usize;

            // Shallow-depth pruning
            if !is_pv && has_non_pawn && !is_loss(best_score) {
                let lmp_count = (3 + depth * depth) as usize / (2 - improving as usize);
                if legal_moves > lmp_count {
                    picker.skip_quiet_moves();
                }
                let lmr_depth = depth - 1;

                if is_capture || gives_check {
                    if let Some(cap) = captured_type
                        && !gives_check
                    {
                        let capt_hist = self.hist.capture[pc.idx()][to][cap.idx()];
                        let see_margin = (SEE_CAPTURE_LINEAR * depth + capt_hist / SEE_CAPTURE_HIST_DIV).max(0);
                        if !pos.see_ge(m, -see_margin) {
                            continue;
                        }
                    }
                } else {
                    let history = self.hist.main[us.idx()][m.from_to()];
                    if history < -HISTORY_PRUNE_MULT * depth {
                        continue;
                    }
                    let adj_lmr_depth = (lmr_depth + history / QUIET_HISTORY_DEPTH_DIV).max(0);

                    if !in_check && adj_lmr_depth < QUIET_FUTILITY_MAX_LMR_DEPTH {
                        let no_best = if best_move.is_none() { QUIET_FUTILITY_NO_BEST } else { 0 };
                        let futility = static_eval + QUIET_FUTILITY_BASE + no_best + QUIET_FUTILITY_PER_DEPTH * adj_lmr_depth;
                        if futility <= alpha {
                            if best_score <= futility && !is_decisive(best_score) {
                                best_score = futility;
                            }
                            continue;
                        }
                    }

                    if !pos.see_ge(m, -SEE_QUIET_QUAD * adj_lmr_depth * adj_lmr_depth) {
                        continue;
                    }
                }
            }

            self.tt.prefetch(pos.key_after(m));

            let mut extension: i32 = 0;
            let is_tt_move = m == tt_move;

            // Singular extension / multi-cut / negative extension on the TT move.
            if let Some((tt_s, singular_depth)) = se_conditions
                && is_tt_move
                && !is_pv
                && excluded.is_none()
                && depth >= SE_MIN_DEPTH + tt_pv as i32
                && !self.is_shuffling(pos, m, ply, is_capture)
            {
                let singular_beta = tt_s - SE_BETA_DEPTH_MULT * depth + self.hist.tt_move_history / SE_TT_HISTORY_DIV;
                let se_value = self.negamax(
                    pos,
                    singular_depth,
                    ply,
                    singular_beta - 1,
                    singular_beta,
                    if cut_node { NodeType::Cut } else { NodeType::All },
                    false,
                    m,
                );
                if self.stopped {
                    return 0;
                }
                if se_value < singular_beta {
                    let pv_bonus = if is_pv { depth * 2 } else { 0 };
                    let double_margin = depth * 2 - tt_capture as i32 * 5 + pv_bonus;
                    let triple_margin = depth * 4 - tt_capture as i32 * 10 + pv_bonus * 2;
                    extension = 1;
                    if se_value < singular_beta - double_margin {
                        extension = 2;
                    }
                    if se_value < singular_beta - triple_margin {
                        extension = 3;
                    }
                    depth += 1;
                } else if se_value >= beta && !is_pv && !is_decisive(se_value) {
                    // Multi-cut: other moves also beat beta, so prune the whole node.
                    let penalty = (-400 - 100 * depth).max(-4000);
                    self.hist.tt_move_history += penalty - ((self.hist.tt_move_history * penalty.abs()) >> 13);
                    return se_value;
                } else if tt_value != VALUE_NONE && tt_value >= beta {
                    extension = -3;
                } else if cut_node {
                    extension = -2;
                }
            }

            // A check at the horizon deserves one real reply rather than the qsearch boundary.
            if extension == 0 && depth <= 1 && gives_check && !in_check {
                extension = 1;
            }

            self.set_ctx(ply, m, pc, in_check, is_capture);
            self.stack[ply].stat_score = self.hist.main[us.idx()][m.from_to()];
            pos.make_move(m, gives_check);

            if !is_capture && !is_promotion {
                quiets_searched.push(m);
            } else if let Some(cap) = captured_type
                && captures_count < 32
            {
                captures_searched[captures_count] = (m, cap);
                captures_count += 1;
            }

            let new_depth = depth - 1 + extension;
            let score;
            if legal_moves == 1 {
                let child_type = if is_pv {
                    NodeType::PV
                } else if cut_node {
                    NodeType::All
                } else {
                    NodeType::Cut
                };
                self.stack[ply].reduction = 0;
                score = -self.negamax(pos, new_depth.max(0), ply + 1, -beta, -alpha, child_type, true, Move::NONE);
            } else {
                // Late move reductions
                let mut reduction = 0;
                if depth >= LMR_MIN_DEPTH
                    && legal_moves >= LMR_MIN_MOVES
                    && !in_check
                    && !is_capture
                    && !(gives_check && p_type == PieceType::Queen)
                {
                    reduction = lmr_reduction(depth, legal_moves);
                    if !improving {
                        reduction += 1;
                    }
                    if tt_pv {
                        reduction -= 1;
                    }
                    if cut_node {
                        reduction += 1;
                    }
                    let hist_score = self.hist.main[us.idx()][m.from_to()];
                    let pawn_score = self.pawn_hist(pawn_key, pc, to);
                    let cont_score = self.cont_history_sum(&cont_keys, pc, to, 2);
                    reduction -= (hist_score + pawn_score) / 4096 + cont_score / 6144;

                    let correction = (static_eval - raw_eval) * CORRHIST_GRAIN;
                    reduction -= (correction.abs() / LMR_CORR_DIVISOR).clamp(0, 2);

                    if self.is_shuffling(pos, m, ply, is_capture) {
                        reduction += 1;
                    }
                    if self.stack[ply + 1].cutoff_cnt > LMR_CUTOFF_THRESH {
                        reduction += 1;
                        if all_node {
                            reduction += 1;
                        }
                    }
                    if self.hist.tt_move_history < LMR_TT_HISTORY_THRESH && reduction > 0 {
                        reduction -= 1;
                    }
                    reduction = reduction.clamp(0, (depth - 2).max(0));
                }

                let mut child_depth = new_depth - reduction;

                // History leaf pruning
                if !in_check
                    && !is_pv
                    && !is_capture
                    && !is_promotion
                    && !gives_check
                    && depth <= HLP_MAX_DEPTH
                    && legal_moves >= HLP_MIN_MOVES
                    && !is_loss(best_score)
                {
                    let value = self.hist.main[us.idx()][m.from_to()] + self.pawn_hist(pawn_key, pc, to);
                    if value < HLP_HISTORY_REDUCE {
                        child_depth -= 1;
                        if child_depth <= 0 && value < HLP_HISTORY_LEAF {
                            pos.unmake_move(m);
                            continue;
                        }
                    }
                }

                let search_depth = child_depth.max(0);
                let child_type = if cut_node { NodeType::Cut } else { NodeType::All };
                // Flip: the child of a cut node is expected to be an all node.
                let child_type = if child_type == NodeType::Cut { NodeType::All } else { NodeType::Cut };
                self.stack[ply].reduction = reduction;

                let mut s =
                    -self.negamax(pos, search_depth, ply + 1, -alpha - 1, -alpha, child_type, true, Move::NONE);

                if s > alpha && (reduction > 0 || s < beta) && !self.stopped {
                    let research_type = if is_pv { NodeType::PV } else { child_type };
                    let base_depth = new_depth;
                    let do_deeper = search_depth < base_depth && s > best_score + 43 + 2 * base_depth;
                    let do_shallower = s < best_score + 9;
                    let mut adjusted = (base_depth + do_deeper as i32 - do_shallower as i32).max(0);
                    if is_pv && is_tt_move && adjusted == 0 {
                        let has_decisive = tt_value != VALUE_NONE && is_decisive(tt_value) && tt_depth > 0;
                        if has_decisive || tt_depth > 1 {
                            adjusted = 1;
                        }
                    }
                    self.stack[ply].reduction = 0;
                    s = -self.negamax(pos, adjusted, ply + 1, -beta, -alpha, research_type, true, Move::NONE);

                    // A reduced search that forced a re-search proved the quiet good.
                    if reduction > 0 && !is_capture && !is_promotion {
                        self.update_continuation_histories(ply, in_check, pc, to, 100 * depth);
                    }
                }
                score = s;
            }

            pos.unmake_move(m);

            if self.stopped {
                return best_score;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    tt_best_move = m;
                    self.update_pv(ply, m);
                }
            }

            if alpha >= beta {
                if extension < 2 || is_pv {
                    self.stack[ply].cutoff_cnt = self.stack[ply].cutoff_cnt.saturating_add(1);
                }
                let bonus = history_bonus(depth);
                if !is_capture {
                    // Credit the cutoff quiet, penalise the quiets tried before it.
                    self.update_main_history(us, m, bonus);
                    self.update_pawn_history(pawn_key, pc, to, bonus * PAWN_HISTORY_BONUS_SCALE);
                    self.update_low_ply_history(ply, m, bonus);
                    for i in 0..quiets_searched.len() {
                        let q = quiets_searched[i];
                        if q == m {
                            continue;
                        }
                        let qpc = pos.moved_piece(q);
                        self.update_main_history(us, q, -bonus);
                        self.update_pawn_history(pawn_key, qpc, q.to_sq() as usize, -bonus * PAWN_HISTORY_MALUS_SCALE);
                        self.update_low_ply_history(ply, q, -bonus);
                        self.update_continuation_histories(ply, in_check, qpc, q.to_sq() as usize, -bonus);
                    }
                    self.update_continuation_histories(ply, in_check, pc, to, bonus);

                    // Killers: don't shift a duplicate into the second slot.
                    let k = &mut self.stack[ply].killers;
                    if k[0] != m {
                        k[1] = k[0];
                        k[0] = m;
                    }
                    // Countermove
                    let prev = self.stack[ply - 1];
                    if prev.current_move.is_ok() && prev.moved_piece.is_some() {
                        self.hist.countermoves[prev.moved_piece.idx()][prev.current_move.to_sq() as usize] = m;
                    }
                } else if let Some(cap) = captured_type {
                    self.update_capture_history(pc, to, cap, bonus);
                }
                // Captures tried before the cutoff did not produce one.
                for &(cm, cap) in captures_searched.iter().take(captures_count) {
                    if cm == m {
                        continue;
                    }
                    let cpc = pos.moved_piece(cm);
                    self.update_capture_history(cpc, cm.to_sq() as usize, cap, -bonus);
                }
                break;
            }
        }

        if legal_moves == 0 {
            if excluded.is_some() {
                return alpha;
            }
            return terminal_value(pos, in_check, ply);
        }
        // Every legal move was pruned: the node fails low by construction.
        if best_score == -VALUE_INFINITE {
            best_score = alpha_orig;
        }

        // Soften fail-high scores from reduced searches.
        if best_score >= beta && !is_decisive(best_score) && !is_decisive(alpha) {
            best_score = (best_score * depth + beta) / (depth + 1);
        }

        // A fail-low under a ttPv parent keeps the PV flag alive on this path.
        if best_score <= alpha_orig && self.stack[ply - 1].tt_pv {
            tt_pv = true;
        }

        let bound = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta_orig {
            Bound::Lower
        } else {
            Bound::Exact
        };

        if excluded.is_none() {
            self.tt.store(key, depth, bound, tt_pv, value_to_tt(best_score, ply), raw_eval, tt_best_move);
        }

        // TT move reliability (non-PV only for clean statistics).
        if !is_pv && best_move.is_some() {
            let delta: i32 = if best_move == tt_move { 809 } else { -865 };
            self.hist.tt_move_history += delta - ((self.hist.tt_move_history * delta.abs()) >> 13);
        }

        // Nothing raised alpha: the opponent's previous quiet was good.
        if best_score <= alpha_orig {
            let prev = self.stack[ply - 1];
            if !prev.is_capture && prev.current_move.is_ok() && prev.moved_piece.is_some() {
                let bonus = history_bonus(depth) / 2;
                let prev_to = prev.current_move.to_sq() as usize;
                self.update_continuation_histories(ply - 1, in_check, prev.moved_piece, prev_to, bonus);
                self.update_main_history(us.flip(), prev.current_move, bonus);
                if prev.moved_piece.piece_type() != PieceType::Pawn && !prev.current_move.is_promotion() {
                    self.update_pawn_history(pawn_key, prev.moved_piece, prev_to, bonus * PAWN_HISTORY_BONUS_SCALE);
                }
            }
        }

        // Correction history learns from quiet, out-of-check nodes whose score respects
        // the bound relative to the static eval.
        if !in_check && excluded.is_none() {
            let best_is_quiet = best_move.is_none() || (!pos.is_capture(best_move) && !best_move.is_promotion());
            let should_update = match bound {
                Bound::Lower => best_score >= raw_eval,
                Bound::Upper => best_score <= raw_eval,
                Bound::Exact => true,
                Bound::None => false,
            };
            if best_is_quiet && should_update {
                self.update_correction_history(pos, depth, raw_eval, best_score, prev_move_idx);
            }
        }

        best_score
    }

    // Quiescence

    fn qsearch(&mut self, pos: &mut Position, ply: usize, qs_ply: usize, alpha: Value, beta: Value, node_type: NodeType) -> Value {
        let mut alpha = alpha;
        let is_pv = node_type == NodeType::PV;
        if ply >= MAX_PLY - 1 {
            return evaluate(pos);
        }

        if alpha < VALUE_DRAW && pos.upcoming_repetition(ply) {
            let draw_val = value_draw(self.nodes) + draw_contempt(self.contempt, ply);
            if draw_val >= beta {
                return draw_val;
            }
            alpha = alpha.max(draw_val);
        }

        self.nodes += 1;
        self.qnodes += 1;
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        let in_check = pos.in_check();

        if pos.is_draw(ply) {
            return VALUE_DRAW + draw_contempt(self.contempt, ply);
        }
        if let Some(v) = variant_terminal(pos, ply) {
            return v;
        }
        if self.check_time() {
            return 0;
        }
        // Antichess: with a capture available standing pat is not an option.
        let forced = pos.captures_forced();

        let key = pos.key();
        let alpha_orig = alpha;
        let rule50 = pos.rule50_count();
        let tt_entry = self.tt.probe(key);
        let tt_hit = tt_entry.is_some();
        let (tt_move, tt_value, tt_eval, tt_bound, pv_hit) = match tt_entry {
            Some(d) => (d.mv, value_from_tt(d.score, ply, rule50), d.eval, d.bound, d.is_pv),
            None => (Move::NONE, VALUE_NONE, VALUE_NONE, Bound::None, false),
        };

        if !is_pv && tt_hit && tt_value != VALUE_NONE {
            let fails_high = tt_value >= beta;
            let ok = if fails_high { tt_bound.has_lower() } else { tt_bound.has_upper() };
            if ok {
                return tt_value;
            }
        }

        let prev_move_idx = self.prev_move_idx(ply);
        let mut unadjusted = VALUE_NONE;
        let mut best_value;
        let mut best_move = Move::NONE;

        if in_check || forced {
            best_value = -VALUE_INFINITE;
        } else {
            let raw = if tt_hit && tt_eval != VALUE_NONE { tt_eval } else { evaluate(pos) };
            unadjusted = raw;
            best_value = self.adjusted_eval(pos, raw, prev_move_idx);
            if tt_hit && tt_value != VALUE_NONE && !is_decisive(tt_value) {
                let ok = if tt_value > best_value { tt_bound.has_lower() } else { tt_bound.has_upper() };
                if ok {
                    best_value = tt_value;
                }
            }
            // Stand pat
            if best_value >= beta {
                if !is_decisive(best_value) {
                    best_value = (best_value + beta) / 2;
                }
                if !tt_hit {
                    self.tt.store(key, 0, Bound::Lower, false, value_to_tt(best_value, ply), unadjusted, Move::NONE);
                }
                return best_value;
            }
            if best_value > alpha {
                alpha = best_value;
            }
            if !tt_hit {
                self.tt.store(key, 0, Bound::None, false, VALUE_NONE, unadjusted, Move::NONE);
            }
        }

        // Cap check/evasion chains.
        if qs_ply >= MAX_QSEARCH_DEPTH {
            return if in_check || forced { alpha } else { best_value };
        }

        let mut list = MoveList::new();
        if in_check {
            movegen::generate_evasions(pos, &mut list);
        } else {
            movegen::generate_captures(pos, &mut list);
        }
        sort_captures(pos, &mut list, tt_move);

        let prev_sq = {
            let pm = self.stack[ply - 1].current_move;
            if pm.is_ok() { pm.to_sq() } else { SQ_NONE }
        };

        let mut legal = 0usize;
        for i in 0..list.len() {
            let m = list[i];
            let is_recapture = prev_sq == m.to_sq();

            if !in_check && !forced && !is_loss(best_value) && !is_recapture {
                // Losing captures never help; nor do ones that cannot reach alpha.
                if !pos.see_ge(m, 0) {
                    continue;
                }
                let gain = pos.captured_type(m).map_or(0, piece_value)
                    + if m.is_promotion() { piece_value(m.promotion()) - piece_value(PieceType::Pawn) } else { 0 };
                if best_value + gain + DELTA_MARGIN < alpha {
                    continue;
                }
            }

            if !pos.legal(m) {
                continue;
            }
            self.tt.prefetch(pos.key_after(m));
            let pc = pos.moved_piece(m);
            let is_capture = pos.is_capture(m);
            let gives_check = pos.gives_check(m);
            self.set_ctx(ply, m, pc, in_check, is_capture);
            pos.make_move(m, gives_check);
            legal += 1;

            let score = -self.qsearch(pos, ply + 1, qs_ply + 1, -beta, -alpha, node_type);
            pos.unmake_move(m);

            if self.stopped {
                return best_value;
            }
            if score > best_value {
                best_value = score;
                if score > alpha {
                    alpha = score;
                    best_move = m;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        if legal == 0 {
            if in_check {
                return mated_in(ply);
            }
            if forced {
                return terminal_value(pos, false, ply);
            }
        }

        if !is_decisive(best_value) && best_value > beta {
            best_value = (best_value + beta) / 2;
        }

        let bound = if best_value >= beta {
            Bound::Lower
        } else if best_value <= alpha_orig {
            Bound::Upper
        } else {
            Bound::Exact
        };
        self.tt.store(key, 0, bound, pv_hit, value_to_tt(best_value, ply), unadjusted, best_move);
        best_value
    }
}

// Multi-threaded driver

#[derive(Clone, Debug)]
pub struct ThreadResult {
    pub best_move: Move,
    pub score: Value,
    pub completed_depth: i32,
    pub pv_length: usize,
    pub nodes: u64,
}

/// Thread voting: each move accrues `(score - min + 14) * depth` votes;
/// proven wins win outright and a proven loss is never switched to.
pub fn select_best_thread(results: &[ThreadResult]) -> usize {
    let min_score = results.iter().map(|r| r.score).min().unwrap_or(0);
    let vote_value = |r: &ThreadResult| (r.score - min_score + 14) as i64 * r.completed_depth as i64;
    let votes_for = |m: Move| -> i64 { results.iter().filter(|r| r.best_move == m).map(vote_value).sum() };

    let mut best_idx = 0;
    for (i, r) in results.iter().enumerate() {
        let best = &results[best_idx];
        let best_vote = votes_for(best.best_move);
        let new_vote = votes_for(r.best_move);
        let best_proven_win = is_win(best.score);
        let best_proven_loss = best.score != -VALUE_INFINITE && is_loss(best.score);
        let better_pv = vote_value(r) * (r.pv_length > 2) as i64 > vote_value(best) * (best.pv_length > 2) as i64;
        if best_proven_win || best_proven_loss {
            if r.score > best.score {
                best_idx = i;
            }
        } else if is_win(r.score) || (!is_loss(r.score) && (new_vote > best_vote || (new_vote == best_vote && better_pv))) {
            best_idx = i;
        }
    }
    best_idx
}

/// Reduced-strength move choice: weaker levels accept larger score deficits, with a
/// pseudo-random push so play varies between games.
fn skill_pick(root_moves: &[RootMove], level: i32, seed: u64) -> Move {
    let candidates: Vec<&RootMove> = root_moves.iter().take(4).filter(|rm| rm.score != -VALUE_INFINITE).collect();
    if candidates.is_empty() {
        return root_moves[0].mv;
    }
    let top = candidates[0].score;
    let lowest = candidates.last().unwrap().score;
    let delta = (top - lowest).min(piece_value(PieceType::Pawn));
    let weakness = (120 - 2 * level).max(1) as i64;
    let mut rng = seed | 1;
    let mut best = candidates[0].mv;
    let mut max_score = -VALUE_INFINITE as i64;
    for rm in candidates {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let push = (weakness * (top - rm.score) as i64 + delta as i64 * (rng % weakness as u64) as i64) / 128;
        let s = rm.score as i64 + push;
        if s >= max_score {
            max_score = s;
            best = rm.mv;
        }
    }
    best
}

pub struct SearchOutcome {
    pub best_move: Move,
    pub ponder: Move,
    pub score: Value,
    pub depth: i32,
    pub nodes: u64,
}

/// Runs a full search with all searchers (searcher 0 on the calling thread). Returns
/// the voted result; the caller prints `bestmove`.
pub fn think(pos: &Position, limits: &Limits, tm: TimeManager, searchers: &mut [Box<Searcher>], stop: &Arc<AtomicBool>) -> SearchOutcome {
    stop.store(false, Ordering::Relaxed);
    for c in searchers[0].node_counters.iter() {
        c.store(0, Ordering::Relaxed);
    }
    searchers[0].tt.increment_age();

    let (main, helpers) = searchers.split_first_mut().expect("at least one searcher");
    main.new_search(limits.clone(), tm.clone());

    std::thread::scope(|s| {
        for h in helpers.iter_mut() {
            let mut p = pos.clone();
            let limits = limits.clone();
            h.silent = true;
            h.new_search(limits, TimeManager::untimed());
            s.spawn(move || h.iterative_deepening(&mut p));
        }
        let mut p = pos.clone();
        main.iterative_deepening(&mut p);
        // In infinite mode (or an unconfirmed ponder) the GUI ends the search.
        if limits.infinite || main.limits.ponder {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        stop.store(true, Ordering::Relaxed);
    });

    if searchers[0].skill_level < 20 && !searchers[0].root_moves.is_empty() {
        let s = &mut searchers[0];
        let pick = skill_pick(&s.root_moves, s.skill_level, s.nodes ^ pos.key());
        s.best_move_root = pick;
        let total_nodes: u64 = searchers.iter().map(|s| s.nodes).sum();
        let s = &searchers[0];
        let rm = s.root_moves.iter().find(|rm| rm.mv == pick);
        return SearchOutcome {
            best_move: pick,
            ponder: rm.and_then(|rm| rm.pv.get(1).copied()).unwrap_or(Move::NONE),
            score: rm.map_or(s.prev_score, |rm| rm.score),
            depth: s.completed_depth,
            nodes: total_nodes,
        };
    }

    let results: Vec<ThreadResult> = searchers
        .iter()
        .filter(|s| !s.root_moves.is_empty() && s.best_move_root.is_some())
        .map(|s| ThreadResult {
            best_move: s.best_move_root,
            score: s.prev_score,
            completed_depth: s.completed_depth.max(1),
            pv_length: s.root_moves[0].pv.len(),
            nodes: s.nodes,
        })
        .collect();
    let total_nodes: u64 = searchers.iter().map(|s| s.nodes).sum();

    if results.is_empty() {
        return SearchOutcome {
            best_move: Move::NONE,
            ponder: Move::NONE,
            score: VALUE_DRAW,
            depth: 0,
            nodes: total_nodes,
        };
    }
    let best_idx = select_best_thread(&results);
    let winner = searchers.iter().filter(|s| !s.root_moves.is_empty() && s.best_move_root.is_some()).nth(best_idx).unwrap();
    let best_move = winner.best_move_root;
    let ponder = winner.root_moves.iter().find(|rm| rm.mv == best_move).and_then(|rm| rm.pv.get(1).copied()).unwrap_or(Move::NONE);
    SearchOutcome {
        best_move,
        ponder,
        score: winner.prev_score,
        depth: winner.completed_depth,
        nodes: total_nodes,
    }
}
