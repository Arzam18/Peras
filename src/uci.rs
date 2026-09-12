//! UCI protocol front end. Searches run on a worker thread so `stop` and `quit` are
//! honoured mid-search; per-thread searchers (and their histories) persist across
//! moves and are handed back when the worker finishes.

use crate::position::{Position, START_FEN};
use crate::search::{Limits, Searcher, SearchOutcome, TimeManager, TranspositionTable, format_score, think};
use crate::types::*;
use crate::{ENGINE_AUTHOR, ENGINE_NAME, ENGINE_VERSION};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

const DEFAULT_HASH_MB: usize = 64;
const MAX_THREADS: usize = 256;
/// Strength of each `Skill Level`. Spacing comes from 598 games played between the levels
/// themselves; the absolute placement comes from a further 360 games in which levels 0, 4,
/// 8 and 16 were matched directly against opponents of known rating, chosen so each score
/// landed near even. Levels never played directly are interpolated between their neighbours.
///
/// Roughly +/- 100 Elo: the reference opponents are Stockfish's rating-limited modes, whose
/// own scale is only approximately calibrated.
///
/// The curve is not smooth, so no formula reproduces it. Levels 12 to 16 span 20 Elo while
/// 19 to 20 spans 275, because the weakening discards up to a pawn at random right through
/// level 19 and is then switched off entirely at 20.
#[rustfmt::skip]
const SKILL_ELO: [i32; 21] = [
    2301, 2364, 2428, 2491, 2555, 2624, 2692,
    2760, 2828, 2874, 2919, 2964, 3010, 3015,
    3020, 3025, 3030, 3065, 3099, 3125, 3400,
];

/// Range spanned by `UCI_Elo`: the weakest and strongest the engine actually plays.
const UCI_ELO_MIN: i32 = SKILL_ELO[0];
const UCI_ELO_MAX: i32 = SKILL_ELO[20];

struct Options {
    hash_mb: usize,
    threads: usize,
    move_overhead: u64,
    multipv: usize,
    contempt: Value,
    chess960: bool,
    analyse_mode: bool,
    skill_level: i32,
    limit_strength: bool,
    elo: i32,
    variant: Variant,
}

struct Engine {
    pos: Position,
    options: Options,
    tt: Arc<TranspositionTable>,
    stop: Arc<AtomicBool>,
    ponderhit: Arc<AtomicBool>,
    #[allow(clippy::vec_box)]
    searchers: Option<Vec<Box<Searcher>>>,
    #[allow(clippy::vec_box)]
    worker: Option<JoinHandle<(Vec<Box<Searcher>>, SearchOutcome)>>,
}

impl Engine {
    fn new() -> Engine {
        let tt = Arc::new(TranspositionTable::new(DEFAULT_HASH_MB));
        let stop = Arc::new(AtomicBool::new(false));
        let ponderhit = Arc::new(AtomicBool::new(false));
        let mut e = Engine {
            pos: Position::startpos(),
            options: Options {
                hash_mb: DEFAULT_HASH_MB,
                threads: 1,
                move_overhead: 10,
                multipv: 1,
                contempt: crate::search::params::DEFAULT_CONTEMPT,
                chess960: false,
                analyse_mode: false,
                skill_level: 20,
                limit_strength: false,
                elo: UCI_ELO_MIN,
                variant: Variant::Standard,
            },
            tt,
            stop,
            ponderhit,
            searchers: None,
            worker: None,
        };
        e.rebuild_searchers();
        e
    }

    fn rebuild_searchers(&mut self) {
        self.join_worker();
        let counters = Arc::new((0..self.options.threads).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
        let mut v = Vec::with_capacity(self.options.threads);
        for i in 0..self.options.threads {
            let s = Searcher::new(i, Arc::clone(&self.tt), Arc::clone(&self.stop), Arc::clone(&counters));
            v.push(Box::new(s));
        }
        self.searchers = Some(v);
        self.apply_searcher_options();
    }

    /// Effective skill level from either the explicit level or a UCI_Elo target.
    fn effective_skill(&self) -> i32 {
        if self.options.limit_strength {
            // The level whose measured strength is nearest the target. The table rises,
            // so this is the first level sitting above the midpoint with its successor.
            let target = self.options.elo;
            let mut lvl = 0;
            while lvl < 20 && target > (SKILL_ELO[lvl] + SKILL_ELO[lvl + 1]) / 2 {
                lvl += 1;
            }
            lvl as i32
        } else {
            self.options.skill_level
        }
    }

    /// Waits for a running search (if any) and reclaims the searchers.
    fn join_worker(&mut self) {
        if let Some(h) = self.worker.take() {
            match h.join() {
                Ok((searchers, _)) => self.searchers = Some(searchers),
                Err(_) => {
                    eprintln!("info string search thread panicked; rebuilding searchers");
                    self.searchers = None;
                }
            }
            if self.searchers.is_none() {
                let counters = Arc::new((0..self.options.threads).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
                let v = (0..self.options.threads)
                    .map(|i| Box::new(Searcher::new(i, Arc::clone(&self.tt), Arc::clone(&self.stop), Arc::clone(&counters))))
                    .collect();
                self.searchers = Some(v);
            }
        }
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.join_worker();
    }

    fn new_game(&mut self) {
        self.stop_search();
        self.tt.clear();
        if let Some(s) = self.searchers.as_mut() {
            for sr in s.iter_mut() {
                sr.clear();
            }
        }
        self.pos = Position::from_fen_variant(self.options.variant.start_fen(), self.options.variant).expect("variant start position");
        self.pos.set_chess960(self.options.chess960);
    }

    fn set_variant(&mut self, v: Variant) {
        self.stop_search();
        self.options.variant = v;
        self.options.chess960 = v == Variant::Chess960;
        self.apply_searcher_options();
        self.new_game();
    }

    fn set_option(&mut self, name: &str, value: &str) {
        let lname = name.to_ascii_lowercase();
        match lname.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.stop_search();
                    self.options.hash_mb = mb.clamp(1, 1 << 20);
                    self.tt = Arc::new(TranspositionTable::new(self.options.hash_mb));
                    self.rebuild_searchers();
                }
            }
            "clear hash" => {
                self.stop_search();
                self.tt.clear();
            }
            "threads" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.options.threads = n.clamp(1, MAX_THREADS);
                    self.rebuild_searchers();
                }
            }
            "move overhead" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.options.move_overhead = ms.min(5000);
                }
            }
            "multipv" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.options.multipv = n.clamp(1, 256);
                    self.apply_searcher_options();
                }
            }
            "contempt" => {
                if let Ok(c) = value.parse::<i32>() {
                    self.options.contempt = c.clamp(-100, 100);
                    self.apply_searcher_options();
                }
            }
            "uci_chess960" => {
                let on = value.eq_ignore_ascii_case("true");
                if on {
                    self.set_variant(Variant::Chess960);
                } else if self.options.variant == Variant::Chess960 {
                    self.set_variant(Variant::Standard);
                }
            }
            "uci_variant" => match Variant::parse(value.trim()) {
                Some(v) => self.set_variant(v),
                None => println!("info string unknown variant '{}'", value),
            },
            "uci_analysemode" => {
                self.options.analyse_mode = value.eq_ignore_ascii_case("true");
                self.apply_searcher_options();
            }
            "skill level" => {
                if let Ok(l) = value.parse::<i32>() {
                    self.options.skill_level = l.clamp(0, 20);
                    self.apply_searcher_options();
                }
            }
            "uci_limitstrength" => {
                self.options.limit_strength = value.eq_ignore_ascii_case("true");
                self.apply_searcher_options();
            }
            "uci_elo" => {
                if let Ok(e) = value.parse::<i32>() {
                    self.options.elo = e.clamp(UCI_ELO_MIN, UCI_ELO_MAX);
                    self.apply_searcher_options();
                }
            }
            "ponder" | "uci_showwdl" => {}
            _ => println!("info string unknown option '{}'", name),
        }
    }

    fn apply_searcher_options(&mut self) {
        self.join_worker();
        let skill = self.effective_skill();
        let contempt = if self.options.analyse_mode { 0 } else { self.options.contempt };
        let ponderhit = Arc::clone(&self.ponderhit);
        if let Some(s) = self.searchers.as_mut() {
            for sr in s.iter_mut() {
                sr.multipv = self.options.multipv;
                sr.contempt = contempt;
                sr.chess960 = self.options.chess960;
                sr.skill_level = skill;
                sr.ponderhit = Arc::clone(&ponderhit);
            }
        }
    }

    fn set_position(&mut self, tokens: &[&str]) {
        self.stop_search();
        let moves_idx = tokens.iter().position(|&t| t == "moves");
        let fen_end = moves_idx.unwrap_or(tokens.len());
        let variant = self.options.variant;
        let new_pos = match tokens.first() {
            Some(&"startpos") => Position::from_fen_variant(variant.start_fen(), variant),
            Some(&"fen") => Position::from_fen_variant(&tokens[1..fen_end].join(" "), variant),
            Some(_) => Position::from_fen_variant(&tokens[..fen_end].join(" "), variant),
            None => return,
        };
        match new_pos {
            Ok(p) => {
                self.pos = p;
                self.pos.set_chess960(self.options.chess960);
            }
            Err(e) => {
                println!("info string invalid position: {}", e);
                return;
            }
        }
        if let Some(mi) = moves_idx {
            for &mv in &tokens[mi + 1..] {
                match self.pos.parse_uci_move(mv) {
                    Some(m) => {
                        let gc = self.pos.gives_check(m);
                        self.pos.make_move(m, gc);
                    }
                    None => {
                        println!("info string illegal move '{}' in position command", mv);
                        break;
                    }
                }
            }
        }
    }

    fn go(&mut self, tokens: &[&str]) {
        self.stop_search();
        let mut limits = Limits::default();
        let mut i = 0;
        let next_num = |i: &mut usize| -> u64 {
            *i += 1;
            tokens.get(*i).and_then(|s| s.parse::<i64>().ok()).unwrap_or(0).max(0) as u64
        };
        while i < tokens.len() {
            match tokens[i] {
                "wtime" => limits.time[0] = next_num(&mut i),
                "btime" => limits.time[1] = next_num(&mut i),
                "winc" => limits.inc[0] = next_num(&mut i),
                "binc" => limits.inc[1] = next_num(&mut i),
                "movestogo" => limits.movestogo = next_num(&mut i),
                "movetime" => limits.movetime = next_num(&mut i),
                "depth" => limits.depth = next_num(&mut i) as i32,
                "nodes" => limits.nodes = next_num(&mut i),
                "mate" => limits.mate = next_num(&mut i) as i32,
                "infinite" => limits.infinite = true,
                "ponder" => limits.ponder = true,
                "searchmoves" => {
                    while let Some(tok) = tokens.get(i + 1) {
                        match self.pos.parse_uci_move(tok) {
                            Some(m) => limits.searchmoves.push(m),
                            None => break,
                        }
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        self.ponderhit.store(false, Ordering::Relaxed);

        let tm = TimeManager::init(&limits, self.pos.side_to_move(), self.pos.game_ply(), self.options.move_overhead);
        let pos = self.pos.clone();
        let mut searchers = self.searchers.take().expect("searchers available");
        let stop = Arc::clone(&self.stop);
        self.worker = Some(std::thread::spawn(move || {
            let outcome = think(&pos, &limits, tm, &mut searchers, &stop);
            let mut out = io::stdout().lock();
            if outcome.best_move.is_none() {
                let _ = writeln!(out, "bestmove 0000");
            } else if outcome.ponder.is_some() {
                let mut after = pos.clone();
                let gc = after.gives_check(outcome.best_move);
                after.make_move(outcome.best_move, gc);
                let _ = writeln!(out, "bestmove {} ponder {}", pos.move_to_uci(outcome.best_move), after.move_to_uci(outcome.ponder));
            } else {
                let _ = writeln!(out, "bestmove {}", pos.move_to_uci(outcome.best_move));
            }
            let _ = out.flush();
            (searchers, outcome)
        }));
    }

    fn bench(&mut self, depth: i32) {
        self.stop_search();
        let fens = BENCH_FENS;
        let saved_pos = self.pos.clone();
        let mut total_nodes = 0u64;
        let start = std::time::Instant::now();
        for (i, fen) in fens.iter().enumerate() {
            self.tt.clear();
            if let Some(s) = self.searchers.as_mut() {
                for sr in s.iter_mut() {
                    sr.clear();
                }
            }
            let pos = Position::from_fen(fen).unwrap();
            println!("\nPosition: {}/{} ({})", i + 1, fens.len(), fen);
            let limits = Limits {
                depth,
                ..Limits::default()
            };
            let mut searchers = self.searchers.take().unwrap();
            let outcome = think(&pos, &limits, TimeManager::untimed(), &mut searchers, &self.stop);
            self.searchers = Some(searchers);
            println!("bestmove {}", pos.move_to_uci(outcome.best_move));
            total_nodes += outcome.nodes;
        }
        let elapsed = start.elapsed().as_millis().max(1) as u64;
        println!("\n===========================");
        println!("Total time (ms) : {}", elapsed);
        println!("Nodes searched  : {}", total_nodes);
        println!("Nodes/second    : {}", total_nodes * 1000 / elapsed);
        self.pos = saved_pos;
        self.tt.clear();
    }
}

/// Mixed opening/middlegame/endgame positions for `bench`.
pub const BENCH_FENS: [&str; 14] = [
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "4rrk1/pp1n3p/3q2pQ/2p1pb1P/2P1p3/2P3N1/2P1P1P1/2P3K1 w - - 0 1",
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3P1p1/1R1RB1P1/5K2 b - - 0 1",
    "8/8/1p1r1k2/p1pPN1p1/P3KnP1/1P6/8/3R4 b - - 0 1",
    "5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - - 0 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
];

pub fn run(args: Vec<String>) {
    let mut engine = Engine::new();

    // Command-line shortcuts: `peras bench [depth]`, `peras perft <n> [fen]`.
    if let Some(cmd) = args.first() {
        match cmd.as_str() {
            "bench" => {
                let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(12);
                engine.bench(depth);
                return;
            }
            "perft" => {
                let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
                if args.len() > 2 {
                    engine.pos = Position::from_fen(&args[2..].join(" ")).unwrap_or_else(|e| {
                        eprintln!("{}", e);
                        std::process::exit(1)
                    });
                }
                perft_cmd(&mut engine.pos, depth, true);
                return;
            }
            _ => {}
        }
    }

    println!("{} {} by {}", ENGINE_NAME, ENGINE_VERSION, ENGINE_AUTHOR);
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else { continue };
        match cmd {
            "uci" => {
                println!("id name {} {}", ENGINE_NAME, ENGINE_VERSION);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name Hash type spin default {} min 1 max 1048576", DEFAULT_HASH_MB);
                println!("option name Clear Hash type button");
                println!("option name Threads type spin default 1 min 1 max {}", MAX_THREADS);
                println!("option name Move Overhead type spin default 10 min 0 max 5000");
                println!("option name MultiPV type spin default 1 min 1 max 256");
                println!("option name Contempt type spin default {} min -100 max 100", crate::search::params::DEFAULT_CONTEMPT);
                println!("option name UCI_Chess960 type check default false");
                println!("option name UCI_AnalyseMode type check default false");
                println!("option name Ponder type check default false");
                println!("option name Skill Level type spin default 20 min 0 max 20");
                println!("option name UCI_LimitStrength type check default false");
                println!("option name UCI_Elo type spin default {} min {} max {}", UCI_ELO_MIN, UCI_ELO_MIN, UCI_ELO_MAX);
                let vars: Vec<String> = Variant::ALL.iter().map(|v| format!("var {}", v.name())).collect();
                println!("option name UCI_Variant type combo default standard {}", vars.join(" "));
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "setoption" => {
                let name_at = tokens.iter().position(|t| t.eq_ignore_ascii_case("name"));
                let value_at = tokens.iter().position(|t| t.eq_ignore_ascii_case("value"));
                if let Some(n) = name_at {
                    let end = value_at.unwrap_or(tokens.len());
                    let name = tokens[n + 1..end].join(" ");
                    let value = value_at.map(|v| tokens[v + 1..].join(" ")).unwrap_or_default();
                    engine.set_option(&name, &value);
                }
            }
            "ucinewgame" => engine.new_game(),
            "position" => engine.set_position(&tokens[1..]),
            "go" => engine.go(&tokens[1..]),
            "stop" => engine.stop_search(),
            "ponderhit" => engine.ponderhit.store(true, Ordering::Relaxed),
            "quit" => {
                engine.stop_search();
                break;
            }
            "d" | "display" => {
                engine.join_worker();
                println!("{}", engine.pos.pretty());
            }
            "eval" => {
                engine.join_worker();
                let v = crate::eval::evaluate(&mut engine.pos);
                println!("info string static eval (side to move): {} ({})", v, format_score(v));
            }
            "perft" => {
                engine.join_worker();
                let depth = tokens.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
                perft_cmd(&mut engine.pos, depth, false);
            }
            "divide" => {
                engine.join_worker();
                let depth = tokens.get(1).and_then(|d| d.parse().ok()).unwrap_or(3);
                perft_cmd(&mut engine.pos, depth, true);
            }
            "bench" => {
                let depth = tokens.get(1).and_then(|d| d.parse().ok()).unwrap_or(12);
                engine.bench(depth);
            }
            _ => println!("info string unknown command '{}'", line),
        }
        let _ = io::stdout().flush();
    }
    engine.stop_search();
}

fn perft_cmd(pos: &mut Position, depth: u32, divide: bool) {
    let start = std::time::Instant::now();
    let nodes = if divide { crate::movegen::perft_divide(pos, depth) } else { crate::movegen::perft(pos, depth) };
    let ms = start.elapsed().as_millis().max(1);
    println!("\nNodes searched: {}\nTime: {} ms\nNPS: {}", nodes, ms, nodes as u128 * 1000 / ms);
}
