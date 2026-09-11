//! Staged move generation: hash move, good captures, killers, quiets (history
//! ordered, good before bad), bad captures. Evasions and ProbCut have their own
//! sequences. Later stages generate nothing if the search cuts off first.

use super::Searcher;
use super::params::*;
use crate::bitboard::*;
use crate::movegen;
use crate::position::{Position, piece_value, piece_value_of};
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    MainTT,
    CaptureInit,
    GoodCapture,
    Killer1,
    Killer2,
    QuietInit,
    GoodQuiet,
    BadCapture,
    BadQuiet,
    EvasionTT,
    EvasionInit,
    Evasion,
    ProbCutTT,
    ProbCutInit,
    ProbCut,
    Done,
}

#[derive(Clone, Copy)]
struct ScoredMove {
    m: Move,
    score: i32,
}

/// Continuation-history key of an ancestor move: (in_check, capture, piece, to).
pub type ContKey = (usize, usize, usize, usize);

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    moves: [ScoredMove; MAX_MOVES],
    len: usize,
    cur: usize,
    end_bad_captures: usize,
    end_captures: usize,
    end_generated: usize,
    ply: usize,
    depth: i32,
    threshold: Value,
    killers: [Move; 2],
    countermove: Move,
    skip_quiets: bool,
    /// Ancestor keys for the 1, 2 and 4 ply offsets (None where no real move exists).
    pub cont_keys: [Option<ContKey>; 3],
}

/// Sorts entries with `score >= limit` to the front in descending order.
fn partial_sort(moves: &mut [ScoredMove], limit: i32) {
    if limit == i32::MIN {
        moves.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        return;
    }
    let mut left = 0;
    for i in 0..moves.len() {
        if moves[i].score >= limit {
            moves.swap(i, left);
            left += 1;
        }
    }
    moves[..left].sort_unstable_by(|a, b| b.score.cmp(&a.score));
}

impl MovePicker {
    /// Main-search and evasion picker. An invalid TT move is dropped entirely so the
    /// later stages do not skip a real move they think was already tried.
    pub fn new(tt_move: Move, ply: usize, depth: i32, searcher: &Searcher, pos: &Position) -> MovePicker {
        let tt_valid = tt_move.is_some() && pos.pseudo_legal(tt_move);
        let tt_move = if tt_valid { tt_move } else { Move::NONE };
        let stage = if pos.in_check() {
            if tt_valid { Stage::EvasionTT } else { Stage::EvasionInit }
        } else if tt_valid {
            Stage::MainTT
        } else {
            Stage::CaptureInit
        };
        Self::init(tt_move, ply, depth, 0, searcher, stage)
    }

    /// ProbCut picker: captures whose SEE reaches `threshold`.
    pub fn new_probcut(tt_move: Move, threshold: Value, searcher: &Searcher, pos: &Position) -> MovePicker {
        debug_assert!(!pos.in_check());
        let tt_valid = tt_move.is_some() && pos.is_capture_stage(tt_move) && pos.pseudo_legal(tt_move);
        let tt_move = if tt_valid { tt_move } else { Move::NONE };
        let stage = if tt_valid { Stage::ProbCutTT } else { Stage::ProbCutInit };
        Self::init(tt_move, 0, 0, threshold, searcher, stage)
    }

    fn init(tt_move: Move, ply: usize, depth: i32, threshold: Value, searcher: &Searcher, stage: Stage) -> MovePicker {
        let killers = searcher.stack[ply].killers;
        let countermove = if ply > 0 {
            let prev = &searcher.stack[ply - 1];
            if prev.current_move.is_ok() && prev.moved_piece.is_some() {
                searcher.hist.countermoves[prev.moved_piece.idx()][prev.current_move.to_sq() as usize]
            } else {
                Move::NONE
            }
        } else {
            Move::NONE
        };
        MovePicker {
            stage,
            tt_move,
            moves: [ScoredMove { m: Move::NONE, score: 0 }; MAX_MOVES],
            len: 0,
            cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            ply,
            depth,
            threshold,
            killers,
            countermove,
            skip_quiets: false,
            cont_keys: searcher.cont_keys(ply),
        }
    }

    #[inline]
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    #[inline]
    fn push(&mut self, m: Move, score: i32) {
        self.moves[self.len] = ScoredMove { m, score };
        self.len += 1;
    }

    /// 10 * victim - attacker, plus capture and main history.
    #[inline]
    fn score_capture(pos: &Position, searcher: &Searcher, m: Move) -> i32 {
        let pc = pos.moved_piece(m);
        let attacker_val = piece_value_of(pc);
        let victim = pos.captured_type(m);
        let promo_gain = if m.is_promotion() { piece_value(m.promotion()) - piece_value(PieceType::Pawn) } else { 0 };
        let victim_val = victim.map_or(0, piece_value);
        let cap_hist = match victim {
            Some(v) => searcher.hist.capture[pc.idx()][m.to_sq() as usize][v.idx()],
            None => 0,
        };
        let main_hist = searcher.hist.main[pc.color().idx()][m.from_to()];
        10 * (victim_val + promo_gain) - attacker_val + cap_hist / 8 + main_hist / 8
    }

    /// History-based quiet score with killer/countermove boosts and threat terms.
    fn score_quiet(&self, pos: &Position, searcher: &Searcher, m: Move) -> i32 {
        if m == self.killers[0] {
            return SORT_KILLER1;
        }
        if m == self.killers[1] {
            return SORT_KILLER2;
        }
        let mut score = SORT_QUIET;
        if m == self.countermove {
            score += SORT_COUNTERMOVE;
        }

        let pc = pos.moved_piece(m);
        let us = pc.color();
        let to = m.to_sq() as usize;

        score += 2 * searcher.hist.main[us.idx()][m.from_to()];
        score += 2 * searcher.pawn_hist(pos.pawn_key(), pc, to);

        for (i, key) in self.cont_keys.iter().enumerate() {
            if let Some(k) = key {
                let v = searcher.hist.cont(*k)[pc.idx()][to] as i32;
                score += v * CONT_WEIGHTS[i] / 1024;
            }
        }

        if pos.gives_check(m) && pos.see_ge(m, SORT_CHECK_SEE) {
            score += SORT_CHECK_BONUS;
        }

        if self.ply < LOW_PLY_HISTORY_SIZE {
            score += 8 * searcher.hist.low_ply[self.ply][m.from_to()] / (1 + self.ply as i32);
        }

        // A quiet that attacks an enemy piece is played far more often than its
        // history suggests, and escaping an attacked square saves material outright.
        if self.depth >= THREAT_ORDERING_MIN_DEPTH {
            let them = us.flip();
            let from_bb = sq_bb(m.from_sq());
            let occ_after = (pos.pieces() ^ from_bb) | sq_bb(m.to_sq());
            let mover_type = if m.is_promotion() { m.promotion() } else { pc.piece_type() };
            let attacks = if mover_type == PieceType::Pawn {
                pawn_attacks(us, m.to_sq())
            } else {
                attacks_bb(mover_type, m.to_sq(), occ_after)
            };
            let mut victims = attacks & pos.pieces_c(them) & !pos.pieces_p(PieceType::King);
            let mut best_vv = 0;
            let mut best_sq = SQ_NONE;
            while victims != 0 {
                let s = pop_lsb(&mut victims);
                let vv = piece_value_of(pos.piece_on(s));
                if vv > best_vv {
                    best_vv = vv;
                    best_sq = s;
                }
            }
            if best_vv > 0 {
                let mut q = best_vv * 6;
                if !pos.attackers_to_exist(best_sq, occ_after, them) {
                    q *= 2;
                }
                score += q.min(12000);
            }

            let mover_val = piece_value_of(pc);
            if mover_val >= 250
                && pos.attackers_to_exist(m.from_sq(), pos.pieces(), them)
                && !pos.attackers_to_exist(m.to_sq(), occ_after, them)
            {
                score += (mover_val * 5).min(10000);
            }
        }

        score
    }

    fn score_evasion(&self, pos: &Position, searcher: &Searcher, m: Move) -> i32 {
        if pos.is_capture(m) {
            pos.captured_type(m).map_or(0, piece_value) + (1 << 28)
        } else {
            self.score_quiet(pos, searcher, m)
        }
    }

    fn generate_captures(&mut self, pos: &Position, searcher: &Searcher) {
        let mut list = MoveList::new();
        movegen::generate_captures(pos, &mut list);
        for &m in list.iter() {
            if m == self.tt_move {
                continue;
            }
            let s = Self::score_capture(pos, searcher, m);
            self.push(m, s);
        }
    }

    fn generate_quiets(&mut self, pos: &Position, searcher: &Searcher) {
        let mut list = MoveList::new();
        movegen::generate_quiets(pos, &mut list);
        for &m in list.iter() {
            if m == self.tt_move || m == self.killers[0] || m == self.killers[1] {
                continue;
            }
            let s = self.score_quiet(pos, searcher, m);
            self.push(m, s);
        }
    }

    fn generate_evasions(&mut self, pos: &Position, searcher: &Searcher) {
        let mut list = MoveList::new();
        movegen::generate_evasions(pos, &mut list);
        for &m in list.iter() {
            if m == self.tt_move {
                continue;
            }
            let s = self.score_evasion(pos, searcher, m);
            self.push(m, s);
        }
    }

    /// Next pseudo-legal move, or `None` when exhausted. Legality is the caller's job.
    pub fn next(&mut self, pos: &Position, searcher: &Searcher) -> Option<Move> {
        loop {
            match self.stage {
                Stage::MainTT | Stage::EvasionTT | Stage::ProbCutTT => {
                    self.stage = match self.stage {
                        Stage::MainTT => Stage::CaptureInit,
                        Stage::EvasionTT => Stage::EvasionInit,
                        _ => Stage::ProbCutInit,
                    };
                    return Some(self.tt_move);
                }

                Stage::CaptureInit | Stage::ProbCutInit => {
                    self.generate_captures(pos, searcher);
                    self.cur = 0;
                    self.end_bad_captures = 0;
                    self.end_captures = self.len;
                    partial_sort(&mut self.moves[..self.len], i32::MIN);
                    self.stage = if self.stage == Stage::CaptureInit { Stage::GoodCapture } else { Stage::ProbCut };
                }

                Stage::GoodCapture => {
                    while self.cur < self.end_captures {
                        let sm = self.moves[self.cur];
                        self.cur += 1;
                        if pos.see_ge(sm.m, GOOD_CAPTURE_SEE) {
                            return Some(sm.m);
                        }
                        // Losing capture: park it at the front for the bad-capture stage.
                        self.moves.swap(self.end_bad_captures, self.cur - 1);
                        self.end_bad_captures += 1;
                    }
                    self.stage = Stage::Killer1;
                }

                Stage::Killer1 => {
                    self.stage = Stage::Killer2;
                    if self.skip_quiets {
                        continue;
                    }
                    let k = self.killers[0];
                    if k.is_some() && k != self.tt_move && !pos.is_capture_stage(k) && pos.pseudo_legal(k) {
                        return Some(k);
                    }
                }

                Stage::Killer2 => {
                    self.stage = Stage::QuietInit;
                    if self.skip_quiets {
                        continue;
                    }
                    let k = self.killers[1];
                    if k.is_some() && k != self.tt_move && k != self.killers[0] && !pos.is_capture_stage(k) && pos.pseudo_legal(k) {
                        return Some(k);
                    }
                }

                Stage::QuietInit => {
                    if self.skip_quiets {
                        self.cur = 0;
                        self.stage = Stage::BadCapture;
                        continue;
                    }
                    let quiet_start = self.len;
                    self.generate_quiets(pos, searcher);
                    self.end_generated = self.len;
                    let limit = -QUIET_SORT_LIMIT_PER_DEPTH * self.depth;
                    partial_sort(&mut self.moves[quiet_start..self.end_generated], limit);
                    self.cur = quiet_start;
                    self.stage = Stage::GoodQuiet;
                }

                Stage::GoodQuiet => {
                    if self.skip_quiets {
                        self.cur = 0;
                        self.stage = Stage::BadCapture;
                        continue;
                    }
                    while self.cur < self.end_generated {
                        let sm = self.moves[self.cur];
                        self.cur += 1;
                        if sm.score > GOOD_QUIET_THRESHOLD {
                            return Some(sm.m);
                        }
                    }
                    self.cur = 0;
                    self.stage = Stage::BadCapture;
                }

                Stage::BadCapture => {
                    if self.cur < self.end_bad_captures {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
                    }
                    self.cur = self.end_captures;
                    self.stage = Stage::BadQuiet;
                }

                Stage::BadQuiet => {
                    if self.skip_quiets {
                        self.stage = Stage::Done;
                        return None;
                    }
                    while self.cur < self.end_generated {
                        let sm = self.moves[self.cur];
                        self.cur += 1;
                        if sm.score <= GOOD_QUIET_THRESHOLD {
                            return Some(sm.m);
                        }
                    }
                    self.stage = Stage::Done;
                }

                Stage::EvasionInit => {
                    self.generate_evasions(pos, searcher);
                    self.end_generated = self.len;
                    self.cur = 0;
                    partial_sort(&mut self.moves[..self.len], i32::MIN);
                    self.stage = Stage::Evasion;
                }

                Stage::Evasion => {
                    if self.cur < self.end_generated {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
                    }
                    self.stage = Stage::Done;
                }

                Stage::ProbCut => {
                    while self.cur < self.end_captures {
                        let sm = self.moves[self.cur];
                        self.cur += 1;
                        if pos.see_ge(sm.m, self.threshold) {
                            return Some(sm.m);
                        }
                    }
                    self.stage = Stage::Done;
                }

                Stage::Done => return None,
            }
        }
    }
}

/// MVV-LVA key for quiescence ordering (promotion gain counted as victim value).
#[inline]
pub fn capture_sort_key(pos: &Position, m: Move) -> i32 {
    let attacker_val = piece_value_of(pos.moved_piece(m));
    let victim_val = pos.captured_type(m).map_or(0, piece_value);
    let promo_gain = if m.is_promotion() { piece_value(m.promotion()) - piece_value(PieceType::Pawn) } else { 0 };
    (victim_val + promo_gain) * 10 - attacker_val
}

/// Sorts a capture list by MVV-LVA, hoisting the TT move to the front if present.
pub fn sort_captures(pos: &Position, list: &mut MoveList, tt_move: Move) {
    let n = list.len();
    let mut scores = [0i32; MAX_MOVES];
    for i in 0..n {
        scores[i] = if list[i] == tt_move { i32::MAX } else { capture_sort_key(pos, list[i]) };
    }
    // Selection sort: capture lists are short.
    for i in 0..n.saturating_sub(1) {
        let mut best = i;
        for j in (i + 1)..n {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        if best != i {
            list.swap(i, best);
            scores.swap(i, best);
        }
    }
}
