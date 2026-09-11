#![cfg(feature = "variants")]

use peras::movegen::perft;
use peras::position::Position;
use peras::types::*;

fn check(variant: Variant, fen: &str, depth: u32, expected: u64) {
    peras::init();
    let mut pos = Position::from_fen_variant(fen, variant).unwrap();
    let nodes = perft(&mut pos, depth);
    assert_eq!(nodes, expected, "{:?} perft({}) of '{}'", variant, depth, fen);
    assert_eq!(pos.fen(), Position::from_fen_variant(fen, variant).unwrap().fen());
}

#[test]
fn antichess_perft() {
    let fen = Variant::Antichess.start_fen();
    check(Variant::Antichess, fen, 1, 20);
    check(Variant::Antichess, fen, 2, 400);
    check(Variant::Antichess, fen, 3, 8_067);
    check(Variant::Antichess, fen, 4, 153_299);
    check(Variant::Antichess, fen, 5, 2_732_672);
}

#[test]
fn antichess_forced_capture_and_king_promotion() {
    peras::init();
    // A capture exists, so only captures are legal.
    let pos = Position::from_fen_variant("8/8/8/3p4/4P3/8/8/8 w - - 0 1", Variant::Antichess).unwrap();
    assert!(pos.must_capture());
    let mut list = MoveList::new();
    peras::movegen::generate_legal(&pos, &mut list);
    assert_eq!(list.len(), 1);
    assert_eq!(pos.move_to_uci(list[0]), "e4d5");

    // Quiet promotion offers five pieces, king included.
    let pos = Position::from_fen_variant("8/P7/8/8/8/8/8/8 w - - 0 1", Variant::Antichess).unwrap();
    let mut list = MoveList::new();
    peras::movegen::generate_legal(&pos, &mut list);
    let mut names: Vec<String> = list.iter().map(|m| pos.move_to_uci(*m)).collect();
    names.sort();
    assert_eq!(names, vec!["a7a8b", "a7a8k", "a7a8n", "a7a8q", "a7a8r"]);

    // King promotion round-trips through make/unmake.
    let mut pos = Position::from_fen_variant("8/P7/8/8/8/8/8/7k w - - 0 1", Variant::Antichess).unwrap();
    let before = pos.fen();
    let m = pos.parse_uci_move("a7a8k").unwrap();
    pos.make_move(m, false);
    assert_eq!(pos.piece_on(squares::A8), Piece::W_KING);
    pos.unmake_move(m);
    assert_eq!(pos.fen(), before);
}

#[test]
fn racing_kings_perft() {
    let fen = Variant::RacingKings.start_fen();
    check(Variant::RacingKings, fen, 1, 21);
    check(Variant::RacingKings, fen, 2, 421);
    check(Variant::RacingKings, fen, 3, 11_264);
    check(Variant::RacingKings, fen, 4, 296_242);
}

#[test]
fn racing_kings_rules() {
    peras::init();
    // Both kings on the eighth rank with White to move: drawn race.
    let pos = Position::from_fen_variant("K6k/8/8/8/8/8/8/8 w - - 0 1", Variant::RacingKings).unwrap();
    assert_eq!(pos.racing_kings_result(0), Some(VALUE_DRAW));
    // White reached first and Black failed to follow: White wins.
    let pos = Position::from_fen_variant("K7/8/7k/8/8/8/8/8 w - - 0 1", Variant::RacingKings).unwrap();
    assert_eq!(pos.racing_kings_result(0), Some(mate_in(0)));
    // White reached first but Black is to move: undecided.
    let pos = Position::from_fen_variant("K7/7k/8/8/8/8/8/8 b - - 0 1", Variant::RacingKings).unwrap();
    assert_eq!(pos.racing_kings_result(0), None);
}

#[test]
fn three_check_perft_matches_standard_until_checks_matter() {
    let fen = Variant::ThreeCheck.start_fen();
    check(Variant::ThreeCheck, fen, 1, 20);
    check(Variant::ThreeCheck, fen, 2, 400);
    check(Variant::ThreeCheck, fen, 3, 8_902);
    check(Variant::ThreeCheck, fen, 4, 197_281);
}

#[test]
fn three_check_counts_and_fen() {
    peras::init();
    let pos = Position::from_fen_variant("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+1 0 1", Variant::ThreeCheck).unwrap();
    assert_eq!(pos.checks_given(Color::White), 0);
    assert_eq!(pos.checks_given(Color::Black), 2);
    assert!(pos.fen().contains(" 3+1 "));
    let pos2 = Position::from_fen_variant("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 +0+2", Variant::ThreeCheck).unwrap();
    assert_eq!(pos2.key(), pos.key());

    // Delivering a check increments the giver's counter and changes the hash.
    let mut pos = Position::from_fen_variant("4k3/8/8/8/8/8/8/4K2R w - - 0 1", Variant::ThreeCheck).unwrap();
    let key = pos.key();
    let m = pos.parse_uci_move("h1h8").unwrap();
    assert!(pos.gives_check(m));
    pos.make_move(m, true);
    assert_eq!(pos.checks_given(Color::White), 1);
    assert_ne!(pos.key(), key);
    pos.unmake_move(m);
    assert_eq!(pos.checks_given(Color::White), 0);
    assert_eq!(pos.key(), key);
}

#[test]
fn crazyhouse_perft() {
    // Reference counts from python-chess's CrazyhouseBoard.
    check(Variant::Crazyhouse, Variant::Crazyhouse.start_fen(), 3, 8_902);
    check(Variant::Crazyhouse, Variant::Crazyhouse.start_fen(), 4, 197_281);
    for (fen, counts) in [
        ("rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR[Pp] w KQkq - 0 3", [62u64, 4_715, 197_413]),
        ("rnb1kbnr/pppp1ppp/8/8/7q/8/PPPPP1PP/RNBQKBNR[Pp] w KQkq - 0 1", [3, 224, 6_249]),
        ("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1[Nb] b kq - 0 7", [72, 4_697, 268_611]),
    ] {
        for (i, &n) in counts.iter().enumerate() {
            check(Variant::Crazyhouse, fen, i as u32 + 1, n);
        }
    }
}

#[test]
fn crazyhouse_rules() {
    peras::init();
    let mut pos = Position::from_fen_variant("rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR[Pp] w KQkq - 0 3", Variant::Crazyhouse).unwrap();
    assert_eq!(pos.hand(Color::White, PieceType::Pawn), 1);
    assert_eq!(pos.hand(Color::Black, PieceType::Pawn), 1);
    let before = pos.fen();
    assert!(before.contains("[Pp]"));
    let key = pos.key();
    let m = pos.parse_uci_move("P@e4").unwrap();
    assert!(m.is_drop());
    let gc = pos.gives_check(m);
    pos.make_move(m, gc);
    assert_eq!(pos.piece_on(squares::E4), Piece::W_PAWN);
    assert_eq!(pos.hand(Color::White, PieceType::Pawn), 0);
    assert_ne!(pos.key(), key);
    pos.unmake_move(m);
    assert_eq!(pos.fen(), before);
    assert_eq!(pos.key(), key);

    // A captured promoted piece returns to the pocket as a pawn.
    let mut pos = Position::from_fen_variant("4k3/8/8/8/8/8/8/4K2Q~[] b - - 0 1", Variant::Crazyhouse).unwrap();
    assert!(pos.fen().contains("Q~"));
    let pos2 = Position::from_fen_variant("4k3/8/8/8/8/8/8/4K2Q[] b - - 0 1", Variant::Crazyhouse).unwrap();
    assert_ne!(pos.key(), pos2.key());
    let _ = &mut pos;
    let mut pos = Position::from_fen_variant("4k3/8/8/8/8/8/7r/4K2Q~[] b - - 0 1", Variant::Crazyhouse).unwrap();
    let m = pos.parse_uci_move("h2h1").unwrap();
    pos.make_move(m, pos.gives_check(m));
    assert_eq!(pos.hand(Color::Black, PieceType::Pawn), 1);
    assert_eq!(pos.hand(Color::Black, PieceType::Queen), 0);
    pos.unmake_move(m);
    assert_eq!(pos.hand(Color::Black, PieceType::Pawn), 0);

    // Drops must block a check.
    let pos = Position::from_fen_variant("rnb1kbnr/pppp1ppp/8/8/7q/8/PPPPP1PP/RNBQKBNR[Pp] w KQkq - 0 1", Variant::Crazyhouse).unwrap();
    let mut list = MoveList::new();
    peras::movegen::generate_legal(&pos, &mut list);
    let names: Vec<String> = list.iter().map(|m| pos.move_to_uci(*m)).collect();
    assert!(names.contains(&"P@f2".to_string()) && names.contains(&"P@g3".to_string()));
    assert_eq!(names.len(), 3);
}

#[test]
fn king_of_the_hill_perft_and_rule() {
    check(Variant::KingOfTheHill, Variant::KingOfTheHill.start_fen(), 3, 8_902);
    peras::init();
    let pos = Position::from_fen_variant("4k3/8/8/8/3K4/8/8/8 b - - 0 1", Variant::KingOfTheHill).unwrap();
    assert!(pos.pieces_cp(Color::White, PieceType::King) & peras::bitboard::CENTER != 0);
}
