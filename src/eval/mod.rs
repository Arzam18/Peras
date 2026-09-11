//! Static evaluation entry point: insufficient-material draws, the hand-crafted
//! evaluation, endgame mop-up guidance, drawish-ending scaling and fifty-move damping.

pub mod hce;
pub mod mopup;
#[cfg(feature = "variants")]
pub mod variants;

use crate::bitboard::*;
use crate::position::{Position, piece_value};
use crate::types::*;

/// Largest slice of the evaluation the halfmove-clock damping may remove while a
/// mop-up term is active: TT scores don't know the clock, so fully damping a huge
/// mop-up eval lets stale shuffle scores beat fresh progress.
const RULE50_DAMP_CAP: Value = 700;

/// Evaluation from the side to move's perspective, clamped below the mate range.
#[inline]
pub fn evaluate(pos: &Position) -> Value {
    #[cfg(feature = "variants")]
    match pos.variant() {
        Variant::Antichess => return variants::antichess(pos).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX),
        Variant::RacingKings => return variants::racing_kings(pos).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX),
        Variant::ThreeCheck => {
            let v = hce::evaluate(pos) + variants::three_check_bonus(pos);
            return apply_rule50_damping(pos, v, false).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
        }
        Variant::Crazyhouse => {
            let v = hce::evaluate(pos) + variants::crazyhouse_hand(pos);
            return apply_rule50_damping(pos, v, false).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
        }
        Variant::KingOfTheHill => {
            let v = hce::evaluate(pos) + variants::king_of_the_hill_bonus(pos);
            return apply_rule50_damping(pos, v, false).clamp(-VALUE_EVAL_MAX, VALUE_EVAL_MAX);
        }
        _ => {}
    }
    if pos.is_insufficient_material() {
        return VALUE_DRAW;
    }
    let raw = hce::evaluate(pos);
    let (mop, mop_active) = mopup::mop_up_term(pos);
    let mut v = raw + mop;
    v = apply_pawnless_scale(pos, v);
    v = apply_drawish_scale(pos, v);
    v = apply_rule50_damping(pos, v, mop_active);
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

/// A pawnless leader whose force can never mate can't win however large the
/// material lead; the insufficient-material rule only fires once the board is
/// nearly empty, so without this the eval claims full material up to the draw.
fn apply_pawnless_scale(pos: &Position, v: Value) -> Value {
    if v == 0 {
        return v;
    }
    let leader = if v > 0 { pos.side_to_move() } else { pos.side_to_move().flip() };
    if pos.pieces_cp(leader, PieceType::Pawn) != 0 {
        return v;
    }
    // King plus at most four pieces: bigger forces always have mating material.
    if popcount(pos.pieces_c(leader)) > 5 {
        return v;
    }
    if side_cannot_mate(pos, leader) { v / 8 } else { v }
}

/// Rook/minor endings that are drawn with correct defence (R+minor vs R, R vs minor)
/// are pulled hard toward the draw when the stronger side has no pawns.
fn apply_drawish_scale(pos: &Position, v: Value) -> Value {
    if v == 0 || pos.count_all() > 6 {
        return v;
    }
    if pos.pieces_p(PieceType::Queen) != 0 {
        return v;
    }
    let npm_w = pos.non_pawn_material(Color::White);
    let npm_b = pos.non_pawn_material(Color::Black);
    let (strong, strong_npm, weak_npm) = if npm_w > npm_b {
        (Color::White, npm_w, npm_b)
    } else if npm_b > npm_w {
        (Color::Black, npm_b, npm_w)
    } else {
        return v;
    };
    if pos.pieces_cp(strong, PieceType::Pawn) != 0 {
        return v;
    }
    // Only the stronger side's own claim is scaled (eval is side-to-move relative).
    let strong_to_move = pos.side_to_move() == strong;
    if (v > 0) != strong_to_move {
        return v;
    }
    if strong_npm - weak_npm <= piece_value(PieceType::Bishop) { v / 8 } else { v }
}

#[inline]
fn apply_rule50_damping(pos: &Position, v: Value, mop_active: bool) -> Value {
    let clock = pos.rule50_count().min(199);
    if clock == 0 {
        return v;
    }
    let dampable = if mop_active { v.clamp(-RULE50_DAMP_CAP, RULE50_DAMP_CAP) } else { v };
    v - dampable * clock / 199
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_is_balanced() {
        crate::init();
        let pos = Position::startpos();
        assert_eq!(evaluate(&pos), 0);
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
            let pos = Position::from_fen(fen).unwrap();
            let flipped = Position::from_fen(&flip_fen(fen)).unwrap();
            assert_eq!(evaluate(&pos), evaluate(&flipped), "asymmetric eval for {}", fen);
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
        let pos = Position::from_fen("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&pos) > 800);
        let pos = Position::from_fen("4k3/8/8/8/3Q4/8/8/4K3 b - - 0 1").unwrap();
        assert!(evaluate(&pos) < -800);
    }

    #[test]
    fn insufficient_material_is_draw() {
        crate::init();
        for fen in ["4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4KB2 w - - 0 1", "4k3/8/8/8/8/8/8/4KN2 b - - 0 1", "4k3/8/8/8/8/8/2b5/4KB2 w - - 0 1"] {
            assert_eq!(evaluate(&Position::from_fen(fen).unwrap()), 0, "{}", fen);
        }
    }
}
