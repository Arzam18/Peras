//! Peras: a UCI chess engine for standard 8x8 chess. The search is adapted from the
//! Apeiron infinite-chess engine's and rebuilt on bitboards; the evaluation is its own
//! network, trained separately. Building with the `variants` feature adds crazyhouse,
//! antichess, three-check, racing kings and king of the hill, which keep the
//! hand-crafted evaluation; without it none of that code exists.

pub mod bitboard;
pub mod eval;
pub mod movegen;
pub mod nnue;
pub mod position;
pub mod search;
pub mod types;
pub mod uci;
pub mod zobrist;

#[cfg(not(feature = "variants"))]
pub const ENGINE_NAME: &str = "Peras";
#[cfg(feature = "variants")]
pub const ENGINE_NAME: &str = "Fairy-Peras";
pub const ENGINE_AUTHOR: &str = "FirePlank";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialises every lookup table. Safe to call more than once.
pub fn init() {
    bitboard::init();
    zobrist::init();
    search::init();
}
