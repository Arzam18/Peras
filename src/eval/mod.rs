//! Static evaluation entry point: insufficient-material draws, the network (or, for the
//! variants it was not trained for, the hand-crafted evaluation), the KX-vs-K
//! specialisation and fifty-move damping.

#[cfg(feature = "variants")]
pub mod hce;
pub mod mopup;
#[cfg(feature = "variants")]
pub mod variants;

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;


/// Evaluation from the side to move's perspective, clamped below the mate range.
#[inline]
pub fn evaluate(pos: &mut Position) -> Value {
    #[cfg(feature = "variants")]
    match pos.variant() {
        Variant::Antichess => return variants::antichess(pos).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX),
        Variant::RacingKings => return variants::racing_kings(pos).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX),
        Variant::ThreeCheck => {
            let v = hce::evaluate(pos) + variants::three_check_bonus(pos);
            return apply_rule50_damping(pos, v).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
        }
        Variant::Crazyhouse => {
            let v = hce::evaluate(pos) + variants::crazyhouse_hand(pos);
            return apply_rule50_damping(pos, v).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
        }
        Variant::KingOfTheHill => {
            let v = hce::evaluate(pos) + variants::king_of_the_hill_bonus(pos);
            return apply_rule50_damping(pos, v).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
        }
        _ => {}
    }
    if pos.is_insufficient_material() {
        return VALUE_DRAW;
    }
    if let Some(v) = mopup::kx_vs_k(pos) {
        return apply_rule50_damping(pos, v).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
    }
    let raw = pos.nnue_evaluate();
    let v = apply_rule50_damping(pos, raw);
    v.clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX)
}

/// True when `c`, having no pawns, cannot force mate against a bare king.
pub fn side_cannot_mate(pos: &Position, c: Color) -> bool {
    if pos.pieces_cp(c, PieceType::Pawn) != 0 {
        return false;
    }
    if pos.pieces_cpp(c, PieceType::Rook, PieceType::Queen) != 0 {
        return false;
    }
    let knights = pos.pieces_cp(c, PieceType::Knight);
    let bishops = pos.pieces_cp(c, PieceType::Bishop);
    if bishops == 0 {
        // Knights alone: two knights cannot force mate.
        return popcount(knights) <= 2;
    }
    if knights == 0 {
        // Bishops on one square colour never mate.
        return bishops & LIGHT_SQUARES == 0 || bishops & DARK_SQUARES == 0;
    }
    false
}

#[inline]
fn apply_rule50_damping(pos: &Position, v: Value) -> Value {
    let clock = pos.rule50_count().min(199);
    if clock == 0 {
        return v;
    }
    v - v * clock / 199
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_is_balanced() {
        crate::init();
        let mut pos = Position::startpos();
        assert!(evaluate(&mut pos).abs() < 100);
    }

    #[test]
    fn eval_is_symmetric() {
        crate::init();
        let fens = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for fen in fens {
            let mut pos = Position::from_fen(fen).unwrap();
            let mut flipped = Position::from_fen(&flip_fen(fen)).unwrap();
            assert_eq!(evaluate(&mut pos), evaluate(&mut flipped), "asymmetric eval for {}", fen);
        }
    }

    /// Mirrors a FEN vertically and swaps colours.
    fn flip_fen(fen: &str) -> String {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let ranks: Vec<String> = parts[0]
            .split('/')
            .rev()
            .map(|r| {
                r.chars()
                    .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
                    .collect()
            })
            .collect();
        let side = if parts[1] == "w" { "b" } else { "w" };
        let castling: String = parts[2]
            .chars()
            .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
            .collect();
        let ep = if parts[3] == "-" {
            "-".to_string()
        } else {
            let b = parts[3].as_bytes();
            format!("{}{}", b[0] as char, (b'9' - (b[1] - b'0')) as char)
        };
        format!("{} {} {} {} {} {}", ranks.join("/"), side, castling, ep, parts.get(4).unwrap_or(&"0"), parts.get(5).unwrap_or(&"1"))
    }

    #[test]
    fn material_advantage_shows() {
        crate::init();
        let mut pos = Position::from_fen("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&mut pos) > 500);
        let mut pos = Position::from_fen("4k3/8/8/8/3Q4/8/8/4K3 b - - 0 1").unwrap();
        assert!(evaluate(&mut pos) < -500);
    }

    #[test]
    fn insufficient_material_is_draw() {
        crate::init();
        for fen in ["4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4KB2 w - - 0 1", "4k3/8/8/8/8/8/8/4KN2 b - - 0 1", "4k3/8/8/8/8/8/2b5/4KB2 w - - 0 1"] {
            assert_eq!(evaluate(&mut Position::from_fen(fen).unwrap()), 0, "{}", fen);
        }
    }
}
