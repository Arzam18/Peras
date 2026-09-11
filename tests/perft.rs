use peras::movegen::perft;
use peras::position::Position;

fn check(fen: &str, depth: u32, expected: u64) {
    peras::init();
    let mut pos = Position::from_fen(fen).unwrap();
    let nodes = perft(&mut pos, depth);
    assert_eq!(nodes, expected, "perft({}) of '{}'", depth, fen);
    // The position must be restored exactly.
    assert_eq!(pos.fen(), Position::from_fen(fen).unwrap().fen());
}

#[test]
fn perft_startpos() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    check(fen, 1, 20);
    check(fen, 2, 400);
    check(fen, 3, 8_902);
    check(fen, 4, 197_281);
    check(fen, 5, 4_865_609);
}

#[test]
fn perft_kiwipete() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    check(fen, 1, 48);
    check(fen, 2, 2_039);
    check(fen, 3, 97_862);
    check(fen, 4, 4_085_603);
}

#[test]
fn perft_position3() {
    let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    check(fen, 1, 14);
    check(fen, 2, 191);
    check(fen, 3, 2_812);
    check(fen, 4, 43_238);
    check(fen, 5, 674_624);
    check(fen, 6, 11_030_083);
}

#[test]
fn perft_position4() {
    let fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    check(fen, 1, 6);
    check(fen, 2, 264);
    check(fen, 3, 9_467);
    check(fen, 4, 422_333);
    check(fen, 5, 15_833_292);
}

#[test]
fn perft_position4_mirrored() {
    let fen = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";
    check(fen, 4, 422_333);
}

#[test]
fn perft_position5() {
    let fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    check(fen, 1, 44);
    check(fen, 2, 1_486);
    check(fen, 3, 62_379);
    check(fen, 4, 2_103_487);
}

#[test]
fn perft_position6() {
    let fen = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
    check(fen, 1, 46);
    check(fen, 2, 2_079);
    check(fen, 3, 89_890);
    check(fen, 4, 3_894_594);
}

#[test]
fn perft_en_passant_pins_and_checks() {
    // Discovered check through the captured pawn and horizontal pins.
    check("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1_440_467);
    check("8/8/8/8/k1p4R/8/3P4/3K4 w - - 0 1", 6, 1_134_888);
    check("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1_134_888);
    check("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1_015_133);
    check("8/5k2/8/2Pp4/2B5/1K6/8/8 w - d6 0 1", 6, 1_440_467);
    check("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661_072);
    check("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803_711);
    check("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1_274_206);
    check("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1_720_476);
    check("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3_821_001);
    check("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1_004_658);
    check("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217_342);
    check("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92_683);
    check("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2_217);
    check("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567_584);
    check("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23_527);
}

#[test]
fn perft_chess960() {
    peras::init();
    for (fen, counts) in [
        ("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", [21u64, 528, 12_189, 326_672]),
        ("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/1PP1P3/BQNNRKRB w GEge - 1 9", [21, 790, 17_834, 638_637]),
        ("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9", [20, 479, 10_471, 273_318]),
        ("qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PP3PP/R1BBNNKR w HAha - 0 9", [19, 553, 11_642, 352_687]),
    ] {
        let mut pos = Position::from_fen(fen).unwrap();
        pos.set_chess960(true);
        for (i, &expected) in counts.iter().enumerate() {
            assert_eq!(perft(&mut pos, i as u32 + 1), expected, "chess960 perft({}) of '{}'", i + 1, fen);
        }
    }
}
