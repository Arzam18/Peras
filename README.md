# Peras

A strong UCI chess engine in Rust, for standard chess, Chess960 and five other variants.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

Peras takes its search from [Apeiron](https://github.com/FirePlank/infinite-chess-engine), the infinite chess engine, and rebuilds it on bitboards for the 64-square board, with its own network evaluation. It has no dependencies and builds into a single binary.

## Quick Start

```bash
curl -sL https://github.com/FirePlank/Peras-networks/releases/download/peras-v3/peras-v3.nnue -o nets/peras.nnue
cargo build --release
./target/release/peras
```

The network is not kept in this repository, since each one is tens of megabytes. It lives in [Peras-networks](https://github.com/FirePlank/Peras-networks/releases), which holds every network the engine has shipped so any tagged commit can still be built. The build compiles it into the binary from `nets/peras.nnue`; set `EVALFILE=/path/to/net` to use another one. That binary speaks [UCI](https://www.chessprogramming.org/UCI), so point any chess GUI at it (Cute Chess, Arena, Banksia) or run it from the terminal:

```
uci
setoption name Hash value 256
setoption name Threads value 4
position startpos moves e2e4 c7c5
go wtime 60000 btime 60000 winc 1000 binc 1000
```

## Features

- **Search**: iterative deepening PVS with aspiration windows and MultiPV, a shared lock-free transposition table, and the full modern pruning set (razoring, reverse futility, null move, ProbCut, late move pruning and reductions, SEE and history pruning, singular and check extensions).
- **Move ordering**: staged move picking driven by butterfly, capture, continuation, pawn, low-ply and countermove histories, with correction history on the static evaluation.
- **Evaluation**: NNUE, a `(768x10hm + threats + pawn pairs -> 1024)x2 -> 8` network with SCReLU activation trained on Leela Chess Zero data, seeing which piece attacks which alongside where the pieces stand, with lazily updated accumulators, king-bucket refresh caching and AVX2 inference; plus insufficient-material and mop-up knowledge.
- **Board**: magic bitboards, incremental Zobrist keys, static exchange evaluation, and full Chess960 support.
- **Performance**: Lazy SMP up to 256 threads and time management that adapts to search stability.
- **Play control**: `Skill Level` and `UCI_Elo` for weaker opponents, pondering, and `searchmoves` for restricted analysis.

## Strength

About 3500 Elo on one thread at 10s+0.1s, measured against Stockfish's rating-limited modes. [CHANGELOG.md](CHANGELOG.md) records what each release measured.

## Variants

Chess960 is always available. The rest are an optional build feature, so the default binary contains no variant code. Crazyhouse, three-check and king of the hill keep the hand-crafted evaluation, since the network was not trained on them:

```bash
cargo build --release --features variants
```

| `UCI_Variant`   | Rules                                                                                      |
|-----------------|---------------------------------------------------------------------------------------------|
| `standard`      | Orthodox chess                                                                              |
| `chess960`      | Fischer random, with Shredder and X-FEN castling                                            |
| `crazyhouse`    | Captured pieces join your pocket and can be dropped (`P@e4`)                                |
| `antichess`     | Captures are compulsory, the king is an ordinary piece, and losing everything wins          |
| `3check`        | The third check wins                                                                        |
| `racingkings`   | No pawns, no checks allowed, first king to the eighth rank wins                             |
| `kingofthehill` | Reach d4, d5, e4 or e5 with your king to win                                                |

Most GUIs set `UCI_Variant` for you, and the usual aliases (`fischerandom`, `giveaway`, `koth`, `zh`) are accepted.

```bash
cutechess-cli -variant crazyhouse -each proto=uci tc=10+0.1 \
  -engine cmd=peras -engine cmd=peras option.Threads=2 -rounds 50
```

## UCI Options

| Option              | Default    | Description                                    |
|---------------------|------------|------------------------------------------------|
| `Hash`              | 64         | Transposition table size in MB                 |
| `Clear Hash`        |            | Empties the transposition table                |
| `Threads`           | 1          | Search threads                                 |
| `Move Overhead`     | 10         | Milliseconds reserved per move for latency     |
| `MultiPV`           | 1          | Principal variations to report                 |
| `Normalize`         | true       | Report scores so +1.00 is a 50% win chance     |
| `UCI_ShowWDL`       | false      | Append win, draw and loss counts per mille     |
| `UCI_Chess960`      | false      | Chess960 castling rules and notation           |
| `UCI_Variant`       | `standard` | Variant to play                                |
| `Ponder`            | false      | Allow `go ponder` and `ponderhit`              |
| `Skill Level`       | 20         | Lower to weaken play                           |
| `UCI_LimitStrength` | false      | Enables `UCI_Elo`                              |
| `UCI_Elo`           | 2301       | Target rating, 2301 to 3500                    |

The console also takes `d` to print the board, `eval` for a static score, `perft N` and `divide N` for move counts, and `bench [depth]` for a fixed-depth benchmark. `peras bench` and `peras perft N [fen]` work as command-line arguments too.

## Testing

```bash
cargo test --release                     # perft, evaluation and transposition table
cargo test --release --features variants # the above plus every variant
```

Perft counts are checked against independent references, Chess960 and the variants included.

## Acknowledgements

Peras uses a neural network trained on data provided by the [Leela Chess Zero](https://lczero.org/) project, made available under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/), with individual contents under the [Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/).

The threat and pawn-pair input features follow [Stockfish](https://github.com/official-stockfish/Stockfish)'s design; both projects are GPL-3.0.

## License

GPL-3.0. See [LICENSE](LICENSE).

## Links

- [Apeiron](https://github.com/FirePlank/infinite-chess-engine) - The infinite chess engine whose search Peras is adapted from
- [Stockfish](https://github.com/official-stockfish/Stockfish) - The world's strongest open-source chess engine
- [Leela Chess Zero](https://lczero.org/) - The project whose open training data the network is trained on
- [Chess Programming Wiki](https://www.chessprogramming.org/) - Engine development resources
