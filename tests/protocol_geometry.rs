//! Adapter checks against TBP's independent true-rotation definition.
//! Sources: tetris-bot-protocol/tbp-spec text/0000-mvp.md and
//! text/0001-randomizer.md. These fixtures do not use the adapter offsets.
use hikari_rei::board::{Board, Piece, PieceKind, TSpin};
use hikari_rei::protocol::{from_tbp, parse_position, to_tbp};
use serde_json::{Value, json};

fn tbp_cells(movement: &Value) -> Vec<(i32, i32)> {
    let location = &movement["location"];
    let mut cells = match location["type"].as_str().unwrap() {
        "I" => vec![(-1, 0), (0, 0), (1, 0), (2, 0)],
        "O" => vec![(0, 0), (1, 0), (0, 1), (1, 1)],
        "T" => vec![(-1, 0), (0, 0), (1, 0), (0, 1)],
        "J" => vec![(-1, 0), (0, 0), (1, 0), (-1, 1)],
        "L" => vec![(-1, 0), (0, 0), (1, 0), (1, 1)],
        "S" => vec![(-1, 0), (0, 0), (0, 1), (1, 1)],
        "Z" => vec![(0, 0), (1, 0), (-1, 1), (0, 1)],
        other => panic!("unexpected type {other}"),
    };
    let turns = match location["orientation"].as_str().unwrap() {
        "north" => 0,
        "east" => 1,
        "south" => 2,
        "west" => 3,
        other => panic!("unexpected orientation {other}"),
    };
    for _ in 0..turns {
        for point in &mut cells {
            *point = (point.1, -point.0);
        }
    }
    let x = location["x"].as_i64().unwrap() as i32;
    let y = location["y"].as_i64().unwrap() as i32;
    for point in &mut cells {
        point.0 += x;
        point.1 += y;
    }
    cells.sort();
    cells
}

#[test]
fn true_rotation_centers_preserve_all_four_physical_cells() {
    for kind in PieceKind::ALL {
        for rotation in 0..4 {
            let piece = Piece::new(kind, 3, 8, rotation);
            let wire = to_tbp(piece);
            let mut native = piece.cells().to_vec();
            native.sort();
            assert_eq!(tbp_cells(&wire), native, "{kind} rotation {rotation}");
            assert_eq!(from_tbp(&wire).unwrap(), piece);
        }
    }
}

#[test]
fn true_rotation_roundtrip_includes_floor_wall_and_tspin() {
    let pieces = [
        Piece::new(PieceKind::I, -2, 0, 1),
        Piece::new(PieceKind::I, 6, -2, 0),
        Piece::new(PieceKind::O, -1, -1, 0),
        Piece::new(PieceKind::O, 7, -1, 3),
        Piece::new(PieceKind::T, 3, 0, 2).with_tspin(TSpin::Full),
    ];
    for piece in pieces {
        let wire = to_tbp(piece);
        let mut cells = piece.cells().to_vec();
        cells.sort();
        assert_eq!(tbp_cells(&wire), cells);
        assert_eq!(from_tbp(&wire).unwrap(), piece);
    }
}

fn start(queue: &[&str], tail: &[&str]) -> Value {
    json!({"queue":queue,"board":[],"hold":null,"combo":0,"back_to_back":false,
        "randomizer":{"type":"seven_bag","bag_state":tail}})
}

fn consume_and_check_tail(value: &Value, expected_before: u8, expected_after: u8) {
    let (mut board, queue) = parse_position(value).unwrap();
    assert_eq!(board.bag, expected_before);
    for kind in queue {
        assert!(board.take_from_bag(kind));
    }
    assert_eq!(board.bag_candidates(), expected_after);
}

#[test]
fn bag_reconstruction_rewinds_known_preview_across_refill() {
    // Start of a fresh bag, then a preview extending one piece into the next.
    consume_and_check_tail(
        &start(&["I", "O", "T"], &["J", "L", "S", "Z"]),
        0,
        0b1111000,
    );
    consume_and_check_tail(
        &start(
            &["I", "O", "T", "J", "L", "S", "Z", "I"],
            &["O", "T", "J", "L", "S", "Z"],
        ),
        0,
        0b1111110,
    );
    // The initial board is already late in a bag; rewind must not fill it early.
    consume_and_check_tail(
        &start(&["S", "Z", "I", "O"], &["T", "J", "L", "S", "Z"]),
        0b1100000,
        0b1111100,
    );
    // An exhausted tail is represented by the next complete bag by TBP.
    consume_and_check_tail(
        &start(&["S", "Z"], &["I", "O", "T", "J", "L", "S", "Z"]),
        0b1100000,
        0b1111111,
    );
}

#[test]
fn malformed_bag_and_row_masks_are_rejected() {
    assert!(parse_position(&start(&["I"], &["I", "O"])).is_err());
    assert!(parse_position(&start(&["I"], &["O", "O"])).is_err());
    let mut position = start(&["I"], &["O", "T", "J", "L", "S", "Z"]);
    position["board"] = json!([1024]);
    assert!(parse_position(&position).is_err());
    let mut board = Board::default();
    board.rows[0] = 1024;
    position["board"] = serde_json::to_value(board).unwrap();
    assert!(parse_position(&position).is_err());
}

#[test]
fn malformed_move_coordinates_return_errors_without_panicking() {
    for coordinate in [i64::MIN, i64::MAX] {
        for axis in ["x", "y"] {
            let mut movement =
                json!({"location":{"type":"I","orientation":"north","x":4,"y":0},"spin":"none"});
            movement["location"][axis] = json!(coordinate);
            let result = std::panic::catch_unwind(|| from_tbp(&movement));
            assert!(
                result.is_ok(),
                "malformed {axis}={coordinate} must not panic"
            );
            assert!(result.unwrap().is_err());
        }
    }
    for invalid in [
        json!({"location":{"type":"Q","orientation":"north","x":4,"y":0},"spin":"none"}),
        json!({"location":{"type":"T","orientation":"up","x":4,"y":0},"spin":"none"}),
        json!({"location":{"type":"T","orientation":"north","x":"4","y":0},"spin":"none"}),
        json!({"location":{"type":"T","orientation":"north","x":4,"y":0},"spin":"invalid"}),
    ] {
        assert!(from_tbp(&invalid).is_err());
    }
}

#[test]
fn native_board_rejects_negative_combo() {
    let mut position = start(&["I"], &["O", "T", "J", "L", "S", "Z"]);
    let board = Board {
        ren: -1,
        ..Board::default()
    };
    position["board"] = serde_json::to_value(board).unwrap();
    assert!(parse_position(&position).is_err());
}
