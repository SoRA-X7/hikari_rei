//! TBP adapter. The worker starts pondering on `start`, including the countdown.
//! `suggest` only reads a published snapshot; `play` confirms and advances a move.
use crate::board::{Board, HEIGHT, Piece, PieceKind, TSpin};
use crate::eval::Weights;
use crate::movegen::{Path, find_path};
use crate::search::{Config, Search};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::sync::{Arc, RwLock, mpsc};
use std::thread;
use std::time::Duration;

fn kind(value: &Value) -> Result<PieceKind, String> {
    value.as_str().ok_or("piece must be a letter")?.parse()
}

/// Accept bottom-first TBP cell rows, compact row masks, or native Board JSON.
pub fn parse_position(value: &Value) -> Result<(Board, Vec<PieceKind>), String> {
    let mut board = Board::default();
    if let Some(input) = value.get("board") {
        if input.is_object() {
            board = serde_json::from_value(input.clone()).map_err(|e| e.to_string())?;
            if !(0..=1_000_000).contains(&board.ren) {
                return Err("native board ren must be between 0 and 1000000".into());
            }
        } else {
            let rows = input.as_array().ok_or("board must contain rows")?;
            for (y, row) in rows.iter().enumerate() {
                let mask = if let Some(mask) = row.as_u64() {
                    if mask > 1023 {
                        return Err("row mask exceeds 10 bits".into());
                    }
                    mask as u16
                } else {
                    let cells = row.as_array().ok_or("invalid board row")?;
                    if cells.len() != 10 {
                        return Err("each board row needs 10 cells".into());
                    }
                    cells.iter().enumerate().fold(0, |mask, (x, cell)| {
                        mask | if cell.is_null() || cell == false || cell == 0 {
                            0
                        } else {
                            1 << x
                        }
                    })
                };
                if y < HEIGHT {
                    board.rows[y] = mask;
                } else if mask != 0 {
                    return Err("Hikari supports occupied rows 0 through 23".into());
                }
            }
        }
    }
    if board.rows.iter().any(|row| *row > 1023) {
        return Err("row mask exceeds 10 bits".into());
    }
    let queue = value["queue"]
        .as_array()
        .ok_or("queue must be an array")?
        .iter()
        .map(kind)
        .collect::<Result<Vec<_>, _>>()?;
    if queue.is_empty() {
        return Err("start needs at least one known piece".into());
    }
    if let Some(hold) = value.get("hold") {
        board.hold = if hold.is_null() {
            None
        } else {
            Some(kind(hold)?)
        };
    }
    if let Some(raw) = value.get("combo") {
        let combo = raw.as_i64().ok_or("combo must be an integer")?;
        board.ren = i32::try_from(combo.max(0)).map_err(|_| "combo is too large")?;
    }
    if !(0..=1_000_000).contains(&board.ren) {
        return Err("combo must be between 0 and 1000000".into());
    }
    if let Some(raw) = value.get("back_to_back") {
        board.back_to_back = raw.as_bool().ok_or("back_to_back must be a boolean")?;
    }
    if let Some(raw) = value.get("bag") {
        let bag = raw.as_u64().ok_or("bag must be an unsigned integer")?;
        if bag > 127 {
            return Err("bag mask exceeds 7 bits".into());
        }
        board.bag = bag as u8;
    } else if let Some(tail) = value["randomizer"]["bag_state"].as_array() {
        // TBP gives the bag after the visible preview, Hikari stores the bag
        // before drawing the current piece. Rewind only publicly known pieces.
        let mut mask = 0u8;
        for piece in tail {
            let bit = 1 << kind(piece)? as u8;
            if mask & bit != 0 {
                return Err("duplicate in bag_state".into());
            }
            mask |= bit;
        }
        for &piece in queue.iter().rev() {
            let bit = 1 << piece as u8;
            if mask & bit != 0 {
                if mask != 127 {
                    return Err("queue and seven-bag state are inconsistent".into());
                }
                mask = 0;
            }
            mask |= bit;
        }
        board.bag = if mask == 127 { 0 } else { mask };
    }
    if board.bag > 127 {
        return Err("bag mask exceeds 7 bits".into());
    }
    Ok((board, queue))
}

fn center_offset(piece: PieceKind, rotation: u8) -> (i8, i8) {
    match piece {
        PieceKind::I => [(1, 2), (2, 2), (2, 1), (1, 1)][rotation as usize],
        PieceKind::O => [(1, 1), (1, 2), (2, 2), (2, 1)][rotation as usize],
        _ => (1, 1),
    }
}

pub fn to_tbp(piece: Piece) -> Value {
    let (dx, dy) = center_offset(piece.kind, piece.rotation);
    json!({"location":{"type":piece.kind,"orientation":(["north","east","south","west"][piece.rotation as usize]),
        "x":piece.x as i32 + dx as i32,"y":piece.y as i32 + dy as i32},"spin":piece.tspin})
}

pub fn from_tbp(value: &Value) -> Result<Piece, String> {
    let loc = &value["location"];
    let piece_kind = kind(&loc["type"])?;
    let rotation = match loc["orientation"].as_str() {
        Some("north") => 0,
        Some("east") => 1,
        Some("south") => 2,
        Some("west") => 3,
        _ => return Err("invalid orientation".into()),
    };
    let (dx, dy) = center_offset(piece_kind, rotation);
    let x = loc["x"]
        .as_i64()
        .ok_or("missing x")?
        .checked_sub(dx as i64)
        .ok_or("x coordinate overflow")?;
    let y = loc["y"]
        .as_i64()
        .ok_or("missing y")?
        .checked_sub(dy as i64)
        .ok_or("y coordinate overflow")?;
    if !(-3..=9).contains(&x) || !(-3..=23).contains(&y) {
        return Err("piece coordinates outside Hikari board".into());
    }
    let tspin = match value["spin"].as_str().unwrap_or("none") {
        "none" => TSpin::None,
        "mini" => TSpin::Mini,
        "full" => TSpin::Full,
        _ => return Err("invalid spin".into()),
    };
    Ok(Piece {
        kind: piece_kind,
        x: x as i8,
        y: y as i8,
        rotation,
        tspin,
    })
}

enum Action {
    Start(Board, Vec<PieceKind>, u64),
    Reset(Board, u64),
    Play(Piece, u64),
    Reveal(PieceKind),
    Stop,
    Quit,
}
struct Request {
    action: Action,
    reply: mpsc::SyncSender<Result<(), String>>,
}

#[derive(Default)]
struct SnapshotCache {
    key: Option<(Board, Piece, bool)>,
    path: Option<Path>,
}

fn snapshot(search: &Search, position_id: u64, cache: &mut SnapshotCache) -> Value {
    let mut chosen = search.best_move();
    let mut plan = search.current_plan();
    // Standard TBP infers hold from a change of piece kind. A same-kind empty
    // hold, or an unexpanded hold child, cannot represent a physical suggestion.
    if chosen.as_ref().is_some_and(|m| m.piece.is_none())
        && (search.queue().get(1) == search.queue().first()
            || plan.get(1).copied().flatten().is_none())
    {
        chosen = search
            .ranked_moves()
            .into_iter()
            .find(|m| m.piece.is_some());
        plan = chosen.as_ref().map(|m| vec![m.piece]).unwrap_or_default();
    }
    // TBP places a physical piece; its empty hold consumes two Hikari layers.
    let physical = chosen
        .as_ref()
        .and_then(|m| m.piece.or_else(|| plan.get(1).copied().flatten()));
    let hold = chosen.as_ref().is_some_and(|m| m.hold);
    let key = physical.map(|piece| (*search.board(), piece, hold));
    if key != cache.key {
        cache.path = physical.and_then(|piece| find_path(search.board(), piece, hold));
        cache.key = key;
    }
    let path = &cache.path;
    json!({"type":"suggestion","position_id":position_id,
        "moves":physical.filter(|_| path.is_some()).map(to_tbp).into_iter().collect::<Vec<_>>(),
        "move_info":{"search":search.stats(),"chosen":chosen,"plan":plan,"path":path},
        "board":search.board(),"queue":search.queue()})
}

fn worker(rx: mpsc::Receiver<Request>, published: Arc<RwLock<Value>>, mut search: Search) {
    let mut running = false;
    let mut position_id = 0;
    let mut cache = SnapshotCache::default();
    let mut pending = None;
    loop {
        let request = if pending.is_some() {
            pending.take()
        } else if running {
            match rx.try_recv() {
                Ok(request) => Some(request),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(request) => Some(request),
                Err(_) => break,
            }
        };
        if let Some(request) = request {
            let result = match request.action {
                Action::Start(board, queue, id) => {
                    search.start(board, queue);
                    position_id = id;
                    running = true;
                    // Materialize enough of the plan to represent an empty hold
                    // as a physical TBP move before publishing the initial state.
                    for _ in 0..3 {
                        search.search_batch();
                        if snapshot(&search, position_id, &mut cache)["moves"]
                            .as_array()
                            .is_some_and(|a| !a.is_empty())
                        {
                            break;
                        }
                    }
                    Ok(())
                }
                Action::Play(piece, id) => (|| {
                    let active = *search.queue().first().ok_or("play before start")?;
                    if find_path(search.board(), piece, piece.kind != active).is_none() {
                        return Err("play is not a reachable Hikari placement".into());
                    }
                    search.commit_physical(piece)?;
                    position_id = id;
                    Ok(())
                })(),
                Action::Reset(board, id) => {
                    search.reset(board);
                    position_id = id;
                    Ok(())
                }
                Action::Reveal(piece) => search.add_next_piece(piece),
                Action::Stop => {
                    running = false;
                    Ok(())
                }
                Action::Quit => {
                    let _ = request.reply.send(Ok(()));
                    break;
                }
            };
            *published.write().unwrap() = snapshot(&search, position_id, &mut cache);
            let _ = request.reply.send(result);
        } else {
            let expanded = search.search_batch();
            *published.write().unwrap() = snapshot(&search, position_id, &mut cache);
            if expanded == 0 {
                // At the configured depth/memory bound, block briefly on input
                // instead of burning a core repeatedly publishing the same tree.
                match rx.recv_timeout(Duration::from_millis(2)) {
                    Ok(request) => {
                        pending = Some(request);
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        }
    }
}

pub fn run_tbp(config: Config, weights: Weights) -> Result<(), String> {
    let info = json!({"type":"info","name":"Hikari Rust","version":env!("CARGO_PKG_VERSION"),"author":"SoRA_X7",
        "features":["randomizer","move_info"],"reference_commit":crate::REFERENCE_COMMIT,"config":config});
    let search = Search::new(config, weights)?;
    let published = Arc::new(RwLock::new(
        json!({"type":"suggestion","moves":[],"position_id":0}),
    ));
    let shared = published.clone();
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || worker(rx, shared, search));
    let result = (|| {
        let mut output = io::BufWriter::new(io::stdout().lock());
        send(&mut output, &info)?;
        let mut position_id = 0u64;
        for line in io::stdin().lock().lines() {
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let action: Result<Option<Action>, String> = (|| {
                let message: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
                match message["type"].as_str().unwrap_or("") {
                    "rules" => {
                        if !matches!(
                            message["randomizer"].as_str().unwrap_or("seven_bag"),
                            "seven_bag" | "unknown"
                        ) {
                            return Err("unsupported randomizer".into());
                        }
                        send(&mut output, &json!({"type":"ready"}))?;
                        Ok(None)
                    }
                    "start" => {
                        let (board, queue) = parse_position(&message)?;
                        position_id = message["position_id"].as_u64().unwrap_or(position_id + 1);
                        Ok(Some(Action::Start(board, queue, position_id)))
                    }
                    "reset" => {
                        let mut position = message.clone();
                        // Reset changes the authoritative board; the worker
                        // keeps the current public queue and bag.
                        position["queue"] = json!(["I"]);
                        position.as_object_mut().unwrap().remove("randomizer");
                        let (board, _) = parse_position(&position)?;
                        position_id = message["position_id"].as_u64().unwrap_or(position_id + 1);
                        Ok(Some(Action::Reset(board, position_id)))
                    }
                    "play" => {
                        let piece = from_tbp(&message["move"])?;
                        let id = message["position_id"].as_u64().unwrap_or(position_id + 1);
                        position_id = id;
                        Ok(Some(Action::Play(piece, id)))
                    }
                    "new_piece" => Ok(Some(Action::Reveal(kind(&message["piece"])?))),
                    "suggest" => {
                        send(&mut output, &published.read().unwrap())?;
                        Ok(None)
                    }
                    "stop" => Ok(Some(Action::Stop)),
                    "quit" => Ok(Some(Action::Quit)),
                    _ => Err("unknown message type".into()),
                }
            })();
            let action = match action {
                Ok(action) => action,
                Err(reason) => {
                    send(&mut output, &json!({"type":"error","reason":reason}))?;
                    continue;
                }
            };
            if let Some(action) = action {
                let quit = matches!(action, Action::Quit);
                let (reply, done) = mpsc::sync_channel(1);
                tx.send(Request { action, reply })
                    .map_err(|e| e.to_string())?;
                match done.recv().map_err(|e| e.to_string())? {
                    Ok(()) => {}
                    Err(reason) => send(&mut output, &json!({"type":"error","reason":reason}))?,
                }
                if quit {
                    break;
                }
            }
        }
        Ok(())
    })();
    drop(tx);
    handle
        .join()
        .map_err(|_| "search worker panicked".to_string())?;
    result
}

fn send(output: &mut impl Write, value: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *output, value).map_err(|e| e.to_string())?;
    output
        .write_all(b"\n")
        .and_then(|_| output.flush())
        .map_err(|e| e.to_string())
}
