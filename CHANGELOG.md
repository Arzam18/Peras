# Changelog

All notable changes to Peras (formerly known as HydroChess) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/), matching the `version` field in `Cargo.toml`.

### Versioning policy

`v1.0.0` is a deliberate baseline reset that coincides with the HydroChess → Peras rename. Nothing before it was ever released or numbered; `v0.1.0` is recorded after the fact, as the last HydroChess state, so the timeline has a starting point to measure against.

Version bumps are decided by what changed rather than by an accumulator:

- **major** for a new evaluation or anything else that changes how the engine plays across the board
- **minor** for accumulated search and evaluation gains
- **patch** for fixes that do not change playing strength

The bold Elo line under each release comes from a directly measured head-to-head match against the previous release at 10s+0.1s on one thread. Where that match is too one-sided for the rating formula to resolve, the figure is taken instead from each release's own position on a ladder of Stockfish's rating-limited modes, which is noted when it happens.

## v2.1.0 (2026-09-12)
[compare to v2.0.1](https://github.com/FirePlank/Peras/compare/v2.0.1...v2.1.0)

**No change to full-strength play.** `UCI_Elo` now maps onto strengths this engine was measured at, instead of a curve fitted to a different engine.

### Fixed
- `UCI_Elo` advertised a floor of 1320, which the engine cannot reach. Its weakest setting plays at about 2300, so every request below that silently returned a far stronger opponent. The advertised range is now the range it actually spans
- Requests between roughly 3010 and 3030 all collapsed onto nearly the same opponent, since `Skill Level` 12 to 16 spans only 20 Elo

### Changed
- The rating-to-level mapping is a table of each level's measured strength, replacing an inherited cubic. Spacing comes from 598 games between the levels themselves; absolute placement from a further 360 games matching levels 0, 4, 8 and 16 against opponents of known rating, each chosen so the score landed near even

### Known issues
- The table is good to roughly +/- 100 Elo, since the reference opponents are Stockfish's rating-limited modes and that scale is itself only approximately calibrated. Its rungs disagreed by 170 Elo when measuring v0.1.0, and by 134 when measuring `Skill Level` 4
- There is a 275-Elo step between levels 19 and 20, in the weakening itself, which discards up to a pawn at random through level 19 and is then switched off at 20. Smoothing it changes what every level plays at, so it needs its own measurement pass

## v2.0.1 (2026-09-12)
[compare to v2.0.0](https://github.com/FirePlank/Peras/compare/v2.0.0...v2.0.1)

**Elo-neutral, and not a regression.** The time manager was rebuilt on this engine's own measurements rather than inherited constants.

```
Score of new vs old: 184 - 161 - 342  [0.517] 687
Elo: +11.8 +/- 18.4   (95%: -6.5 to +30.2), 0 time losses
Clock used per game: new 15.17s, old 15.23s
Mean search depth:   new 18.89,  old 18.88
SPRT bounds elo0=-5 elo1=0, alpha=beta=0.05, 10s+0.1s, one thread
```

### Changed
- The time manager plans against a horizon measured from 660 of this engine's own games instead of a fixed guess. Expected remaining moves is not monotonic: it falls to a minimum near ply 115 and then rises, because the games still running that late are the long drawish ones. No decaying formula expresses that shape, so the measured curve is tabulated and interpolated
- The allocator is now a budget divided by that horizon with a single urgency scale, replacing the previous logarithm-of-clock polynomial and the per-game adjustment constant it threaded through the UCI layer
- The Zobrist seed is the engine's own

### Fixed
- The horizon taken straight from the statistics under-spent the clock by 19% and cost nearly a full ply of depth, worth about 45 Elo. Time never spent is wasted and games often end before the horizon, so the urgency scale is calibrated against measured clock usage rather than read off the distribution

## v2.0.0 (2026-09-12)
[compare to v1.0.0](https://github.com/FirePlank/Peras/compare/v1.0.0...v2.0.0)

**It is about 580 Elo better than v1.0.0.** Against Stockfish's rating-limited modes it measures about 3400 on one thread at 10s+0.1s, where v1.0.0 measured about 2800. A direct 200-game match finished 196-2-2, which is too lopsided for the rating formula to resolve, so the ladder figure is the one quoted here.

### Added
- NNUE evaluation replacing the hand-crafted one: a `(768x10hm -> 1024)x2 -> 8` network with SCReLU activation and material-indexed output buckets, its weights quantised to `i16` and compiled into the binary
- The network is trained on Leela Chess Zero self-play positions, tablebase-rescored and distributed in Stockfish binpack format, with each position labelled 70% by its search score and 30% by the game result
- Accumulators are updated lazily: a move records only which pieces moved, and the vectors are brought up to date on the first evaluation that actually needs them, so the many nodes that never evaluate cost nothing
- A per-perspective accumulator cache keyed on king bucket and mirror side, so a king crossing a bucket boundary updates against the nearest cached board instead of rebuilding from scratch
- Hand-written AVX2 kernels for the accumulator update and the output dot product, with a scalar fallback
- `EVALFILE` build variable to compile against a network other than `nets/peras.nnue`
- A random-playout test asserting that every incrementally updated evaluation equals a fresh evaluation of the same position, covering the dirty-piece replay, the bucket refreshes and the cache

### Changed
- `UCI_Elo` spans 1320 to 3400, since the ceiling tracks what the engine actually measures, and the skill interpolation is rescaled to match
- `evaluate` takes the position by mutable reference, since the accumulator stack now lives alongside the board
- The hand-crafted evaluation is compiled only with `--features variants`, where crazyhouse, three-check and king of the hill still use it; the network was not trained on them
- Insufficient-material detection, mop-up guidance and fifty-move damping still wrap the network, since none of them are things an evaluation trained on normal play learns reliably

### Removed
- Pawnless-leader and drawish-ending scaling, which existed to correct the hand-crafted evaluation's habit of claiming a full material lead in positions no force can win, and which the network does not need

## v1.0.0 (2026-09-12)
[compare to v0.1.0](https://github.com/FirePlank/Peras/compare/v0.1.0...v1.0.0)

**It is about 700 Elo better than v0.1.0.** Against Stockfish's rating-limited modes it measures about 2800 on one thread at 10s+0.1s, where v0.1.0 measures about 2100. This is also the point at which HydroChess became Peras.

### Added
- Lazy SMP up to 256 threads, with `Threads` wired through UCI and time management that adapts to search stability
- Chess960 in the standard build, with Shredder and X-FEN castling, since it is part of the UCI specification and every serious engine ships it
- Crazyhouse, antichess, three-check, racing kings and king of the hill behind the `variants` build feature, selected through `UCI_Variant` with the usual aliases accepted
- `MultiPV`, `Ponder`, `UCI_AnalyseMode`, `Contempt` and `searchmoves` support
- Perft counts for Chess960 and every variant, checked against independent references

### Changed
- The engine is rebuilt on bitboards with its search adapted from Apeiron, the infinite chess engine: iterative deepening PVS with aspiration windows, a shared lock-free transposition table, and the full modern pruning set
- Move ordering driven by butterfly, capture, continuation, pawn, low-ply and countermove histories, with correction history on the static evaluation
- The separate fairy fork is merged back into one tree. Without the `variants` feature every variant branch, table and state field is compiled out, verified by identical fixed-depth node counts, principal variations and best moves against the pre-merge binary, and by a 500-game match that finished dead level
- `UCI_Elo` spans 1320 to 2800 rather than an inherited 3190 ceiling, and the skill interpolation is rescaled to match; 2800 is what the engine actually measured over 270 games against Stockfish's rating-limited modes
- README rewritten around what the engine is and how to run it

### Fixed
- Crazyhouse drops could overflow the fixed move buffers, since a position with a full pocket exceeds 256 legal moves; the buffer is now 512 with the `variants` feature and 256 without
- Three-check FEN parsing accepts both the lichess form giving checks remaining and the appended form giving checks delivered

## v0.1.0 (2024-11-07)
[the last HydroChess commit](https://github.com/FirePlank/Peras/commit/6b12040)

**The last version under the HydroChess name, recorded here as the baseline the rest is measured against.** It plays at about 2100 on one thread at 10s+0.1s, measured against Stockfish's rating-limited modes. That figure is softer than the later ones: Stockfish's rating limiting is poorly calibrated this low, and its 1800 and 2000 rungs imply 2030 while its 2200 rung implies 2200.

### Added
- Magic bitboards, a transposition table, static exchange evaluation and move ordering
- A hand-crafted evaluation
- UCI, with perft and a board display

### Removed
- Variant support, which moved to a separate Fairy-HydroChess fork; it returns in v1.0.0 behind a build feature
