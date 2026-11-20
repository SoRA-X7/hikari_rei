//! Shared Hikari geometry, input cost and lock rules for the development TBP arena.
use hikari_rei::{
    board::{Board, PieceKind},
    movegen::find_path,
    protocol::from_tbp,
};
use serde_json::{Value, json};
use std::io::{self, BufRead};
fn handle(v: Value) -> Result<Value, String> {
    let mut b: Board = serde_json::from_value(v["board"].clone()).map_err(|e| e.to_string())?;
    let q: Vec<PieceKind> =
        serde_json::from_value(v["queue"].clone()).map_err(|e| e.to_string())?;
    if v["op"] == "legal" {
        return Ok(
            json!({"legal": ([q[0], b.hold.unwrap_or(q[1])].into_iter().any(|k| !hikari_rei::movegen::generate(&b,k).is_empty()))}),
        );
    }
    let requested = from_tbp(&v["move"])?;
    // Mirai retains one representative for physically equivalent non-T rotations.
    // Preserve the original wire move for the bot, and replay its identical cells.
    let mut cells = requested.cells();
    cells.sort_unstable();
    let p = hikari_rei::movegen::generate(&b, requested.kind)
        .into_iter()
        .find(|placement| {
            let mut candidate = placement.piece.cells();
            candidate.sort_unstable();
            candidate == cells && placement.piece.tspin() == requested.tspin()
        })
        .ok_or("unreachable physical placement under shared SRS/input rules")?
        .piece;
    let hold = p.kind != q[0];
    let consumed = if hold && b.hold.is_none() { 2 } else { 1 };
    if hold && p.kind != b.hold.unwrap_or(q[1]) {
        return Err("piece is neither active nor hold".into());
    }
    let path =
        find_path(&b, p, hold).ok_or("unreachable placement under shared SRS/input rules")?;
    if b.collides(p) || !b.collides(p.shifted(0, -1)) {
        return Err("placement is not grounded".into());
    }
    if hold {
        b.hold = Some(q[0]);
    }
    for &kind in &q[..consumed] {
        if !b.take_from_bag(kind) {
            return Err("inconsistent seven bag".into());
        }
    }
    let lock = b.lock(p);
    Ok(
        json!({"board":b,"consumed":consumed,"frames":path.time.max(1),"path":path,"attack":lock.attack().max(0),"lines":lock.lines(),"death":lock.death}),
    )
}
fn main() {
    for line in io::stdin().lock().lines() {
        let result = line
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .and_then(handle);
        println!(
            "{}",
            match result {
                Ok(v) => v,
                Err(e) => json!({"error":e}),
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equivalent_s_rotation_and_empty_hold_preserve_cells_and_draws() {
        let out = handle(json!({"board":Board::default(),"queue":["I","S","T"],
            "move":{"location":{"type":"S","orientation":"east","x":0,"y":1},"spin":"none"}}))
        .unwrap();
        assert_eq!(out["consumed"], 2);
        assert_eq!(out["board"]["hold"], "I");
        assert_eq!(out["board"]["rows"][0], 2);
        assert_eq!(out["board"]["rows"][1], 3);
        assert_eq!(out["board"]["rows"][2], 1);
        assert_eq!(out["board"]["bag"], 94);
    }
    #[test]
    fn shared_perfect_clear_attack_is_ten() {
        let mut board = Board::default();
        board.rows[0] = 1008;
        let out = handle(json!({"board":board,"queue":["I","S","T"],
            "move":{"location":{"type":"I","orientation":"north","x":1,"y":0},"spin":"none"}}))
        .unwrap();
        assert_eq!(out["attack"], 10);
        assert_eq!(out["lines"], 1);
        assert!(
            out["board"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .all(|x| x == 0)
        );
    }
    #[test]
    fn wrong_piece_is_rejected() {
        assert!(
            handle(json!({"board":Board::default(),"queue":["I","S","T"],
            "move":{"location":{"type":"O","orientation":"north","x":0,"y":0},"spin":"none"}}))
            .is_err()
        );
    }
}
