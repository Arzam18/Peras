//! Debug helper: prints a perft divide for FEN/depth given via env vars.
//! `PERAS_FEN="..." PERAS_MOVES="e2e4 e7e5" PERAS_DEPTH=3 cargo test --release --test divide_debug -- --nocapture`

use peras::movegen::perft_divide;
use peras::position::Position;

#[test]
fn divide() {
    let Ok(fen) = std::env::var("PERAS_FEN") else { return };
    let depth: u32 = std::env::var("PERAS_DEPTH").ok().and_then(|d| d.parse().ok()).unwrap_or(1);
    peras::init();
    let mut pos = Position::from_fen(&fen).unwrap();
    if let Ok(moves) = std::env::var("PERAS_MOVES") {
        for m in moves.split_whitespace() {
            let mv = pos.parse_uci_move(m).unwrap_or_else(|| panic!("illegal move {} in {}", m, pos.fen()));
            let gc = pos.gives_check(mv);
            pos.make_move(mv, gc);
        }
    }
    println!("{}", pos.pretty());
    let total = perft_divide(&mut pos, depth);
    println!("total: {}", total);
}
