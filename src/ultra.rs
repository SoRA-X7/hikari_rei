//! A real-clock, 60 Hz Ultra demonstration of the native Hikari search.
//!
//! Inputs, the colored field, and the randomizer live on the game thread. The
//! search owns its retained DAG on another thread and receives only public
//! pieces and confirmed placements. No search budget is substituted for time.

use std::collections::{HashMap, VecDeque};
use std::path::Path as FilePath;
use std::sync::{Arc, RwLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value as Json, json};

use crate::board::{Board, HEIGHT, LockResult, Piece, PieceKind, WIDTH};
use crate::eval::Weights;
use crate::movegen::{Instruction, Path, find_path};
use crate::search::{Config, Search, SearchStats};

const FPS: u32 = 60;
const ENTRY_DELAY: u32 = 3;
const CLEAR_DELAY: u32 = 38;

#[derive(Clone)]
struct Choice {
    piece: Piece,
    path: Path,
    empty_hold: bool,
}

#[derive(Clone, Default)]
struct Snapshot {
    position: u64,
    choice: Option<Choice>,
    stats: SearchStats,
}

enum Command {
    Reveal(PieceKind),
    EmptyHold {
        expected: Board,
        queue: Vec<PieceKind>,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    Commit {
        piece: Piece,
        expected: Board,
        queue: Vec<PieceKind>,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    Quit,
}

struct Thinker {
    tx: mpsc::Sender<Command>,
    snapshot: Arc<RwLock<Snapshot>>,
    join: Option<JoinHandle<()>>,
}

impl Thinker {
    fn new(config: Config, queue: Vec<PieceKind>) -> Result<Self, String> {
        let mut search = Search::new(config, Weights::default())?;
        // This precedes both the worker and the actual countdown clock.
        search.start(Board::default(), queue);
        let (tx, rx) = mpsc::channel();
        let snapshot = Arc::new(RwLock::new(Snapshot::default()));
        let published = Arc::clone(&snapshot);
        let join = thread::Builder::new()
            .name("hikari-ultra-search".into())
            .spawn(move || {
                let mut position = 0;
                let mut cache: Option<(Board, Piece, bool, Path)> = None;
                loop {
                    let mut command = rx.try_recv().ok();
                    while let Some(request) = command {
                        match request {
                            Command::Quit => return,
                            Command::Reveal(kind) => {
                                if search.add_next_piece(kind).is_err() {
                                    return;
                                }
                            }
                            Command::EmptyHold { expected, queue, reply } => {
                                let result = (|| {
                                    search.commit(None)?;
                                    if search.board() != &expected || search.queue() != queue {
                                        return Err("game/search mismatch after confirmed empty hold".into());
                                    }
                                    Ok(())
                                })();
                                let failed = result.is_err();
                                let _ = reply.send(result);
                                if failed { return; }
                            }
                            Command::Commit {
                                piece,
                                expected,
                                queue,
                                reply,
                            } => {
                                let result = (|| {
                                    search.commit_physical(piece)?;
                                    if search.board() != &expected {
                                        return Err("game/search board mismatch after confirmed lock".into());
                                    }
                                    if search.queue() != queue {
                                        return Err("game/search public queue mismatch after confirmed lock".into());
                                    }
                                    position += 1;
                                    Ok(())
                                })();
                                let failed = result.is_err();
                                *published.write().unwrap() = Snapshot {
                                    position,
                                    choice: None,
                                    stats: search.stats(),
                                };
                                let _ = reply.send(result);
                                if failed {
                                    return;
                                }
                            }
                        }
                        command = rx.try_recv().ok();
                    }
                    let expanded = search.search_batch();
                    let chosen = search.best_move();
                    let empty_hold = chosen.is_some_and(|chosen| chosen.piece.is_none());
                    let piece = chosen.and_then(|chosen| {
                        chosen.piece.or_else(|| search.current_plan().get(1).copied().flatten())
                    });
                    let choice = piece.and_then(|piece| {
                        let hold = chosen.is_some_and(|chosen| chosen.hold);
                        if cache.as_ref().is_none_or(|(board, target, held, _)| {
                            board != search.board() || *target != piece || *held != hold
                        }) {
                            cache = find_path(search.board(), piece, hold)
                                .map(|path| (*search.board(), piece, hold, path));
                        }
                        cache.as_ref().map(|(_, _, _, path)| Choice {
                            piece,
                            path: path.clone(),
                            empty_hold,
                        })
                    });
                    *published.write().unwrap() = Snapshot {
                        position,
                        choice,
                        stats: search.stats(),
                    };
                    if expanded == 0 {
                        // Keep the command latency below one frame when a
                        // memory/depth bound leaves nothing further to expand.
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            tx,
            snapshot,
            join: Some(join),
        })
    }

    fn read(&self) -> Snapshot {
        self.snapshot.read().unwrap().clone()
    }

    fn reveal(&self, kind: PieceKind) -> Result<(), String> {
        self.tx
            .send(Command::Reveal(kind))
            .map_err(|e| e.to_string())
    }

    fn commit(
        &self,
        piece: Piece,
        expected: Board,
        queue: Vec<PieceKind>,
    ) -> Result<mpsc::Receiver<Result<(), String>>, String> {
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .send(Command::Commit {
                piece,
                expected,
                queue,
                reply,
            })
            .map_err(|e| e.to_string())?;
        Ok(rx)
    }

    fn empty_hold(
        &self,
        expected: Board,
        queue: Vec<PieceKind>,
    ) -> Result<mpsc::Receiver<Result<(), String>>, String> {
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .send(Command::EmptyHold {
                expected,
                queue,
                reply,
            })
            .map_err(|e| e.to_string())?;
        Ok(rx)
    }
}

impl Drop for Thinker {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Quit);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

struct SevenBag {
    state: u64,
    bag: Vec<PieceKind>,
}

impl SevenBag {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            bag: Vec::new(),
        }
    }

    fn draw(&mut self) -> PieceKind {
        if self.bag.is_empty() {
            self.bag = PieceKind::ALL.to_vec();
            for i in (1..self.bag.len()).rev() {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                self.bag.swap(i, self.state as usize % (i + 1));
            }
        }
        self.bag.pop().unwrap()
    }
}

#[derive(Default)]
struct Controller {
    choice: Option<Choice>,
    next: usize,
    tap_wait: u32,
    sonic: bool,
    soft_ticks: u32,
    held: bool,
}

enum Input {
    None,
    Hold,
    Tap(Instruction),
    SoftDrop,
    HardDrop,
}

impl Controller {
    fn install(&mut self, choice: Choice) {
        *self = Self {
            choice: Some(choice),
            ..Self::default()
        };
    }

    fn input(&mut self, grounded: bool) -> Input {
        let Some(choice) = &self.choice else {
            return Input::None;
        };
        if choice.path.hold && !self.held {
            self.held = true;
            return Input::Hold;
        }
        if self.sonic && grounded {
            self.sonic = false;
        }
        if self.tap_wait > 0 {
            self.tap_wait -= 1;
            return if self.sonic {
                Input::SoftDrop
            } else {
                Input::None
            };
        }
        self.tap_wait = 1;
        if self.sonic {
            return Input::SoftDrop;
        }
        if let Some(&inst) = choice.path.instructions.get(self.next) {
            self.next += 1;
            if inst == Instruction::SonicDrop {
                self.sonic = true;
                self.soft_ticks = 0;
                Input::SoftDrop
            } else {
                Input::Tap(inst)
            }
        } else {
            Input::HardDrop
        }
    }
}

struct Recording {
    boards: Vec<String>,
    board_ids: HashMap<String, usize>,
    poses: Vec<[i32; 4]>,
    pose_ids: HashMap<[i32; 4], usize>,
    queues: Vec<Vec<u8>>,
    queue_ids: HashMap<Vec<u8>, usize>,
    keys: Vec<[i64; 11]>,
}

impl Recording {
    fn new() -> Self {
        Self {
            boards: Vec::new(),
            board_ids: HashMap::new(),
            poses: Vec::new(),
            pose_ids: HashMap::new(),
            queues: Vec::new(),
            queue_ids: HashMap::new(),
            keys: Vec::new(),
        }
    }

    fn pose(&mut self, piece: Option<Piece>) -> i64 {
        let Some(piece) = piece else { return -1 };
        let value = [
            piece.kind as i32 + 1,
            piece.x as i32,
            piece.y as i32,
            piece.rotation as i32,
        ];
        if let Some(&index) = self.pose_ids.get(&value) {
            return index as i64;
        }
        let index = self.poses.len();
        self.pose_ids.insert(value, index);
        self.poses.push(value);
        index as i64
    }

    fn record(&mut self, frame: u32, game: &Game, stats: SearchStats) {
        let board: String = game
            .colors
            .iter()
            .flatten()
            .map(|&n| (b'0' + n) as char)
            .collect();
        let board_id = *self.board_ids.entry(board.clone()).or_insert_with(|| {
            let index = self.boards.len();
            self.boards.push(board);
            index
        });
        let queue: Vec<u8> = game
            .queue
            .iter()
            .skip(usize::from(game.active.is_some()))
            .take(5)
            .map(|&kind| kind as u8 + 1)
            .collect();
        let queue_id = *self.queue_ids.entry(queue.clone()).or_insert_with(|| {
            let index = self.queues.len();
            self.queues.push(queue);
            index
        });
        let pose = self.pose(game.active);
        let ghost = self.pose(game.active.and_then(|piece| game.board.sonic_drop(piece)));
        self.keys.push([
            frame as i64,
            board_id as i64,
            pose,
            ghost,
            game.board.hold.map_or(0, |kind| kind as i64 + 1),
            queue_id as i64,
            game.attack as i64,
            0,
            game.locks as i64,
            stats.nodes as i64,
            stats.max_depth as i64,
        ]);
    }
}

struct Game {
    board: Board,
    colors: [[u8; WIDTH]; HEIGHT],
    queue: VecDeque<PieceKind>,
    random: SevenBag,
    active: Option<Piece>,
    controller: Controller,
    hold_locked: bool,
    gravity: u32,
    spawn_at: Option<u32>,
    commit_reply: Option<mpsc::Receiver<Result<(), String>>>,
    position: u64,
    attack: i32,
    lines: u32,
    locks: u32,
    topout: bool,
    events: Vec<Json>,
}

impl Game {
    fn new(seed: u64, previews: usize) -> Self {
        let mut random = SevenBag::new(seed);
        let queue: VecDeque<_> = (0..previews).map(|_| random.draw()).collect();
        Self {
            board: Board::default(),
            colors: [[0; WIDTH]; HEIGHT],
            active: queue.front().and_then(|&kind| Board::default().spawn(kind)),
            queue,
            random,
            controller: Controller::default(),
            hold_locked: false,
            gravity: 0,
            spawn_at: None,
            commit_reply: None,
            position: 0,
            attack: 0,
            lines: 0,
            locks: 0,
            topout: false,
            events: Vec::new(),
        }
    }

    fn hold(&mut self, thinker: &Thinker) -> Result<(), String> {
        if self.hold_locked {
            return Err("controller attempted a second hold before lock".into());
        }
        let current = self.active.ok_or("hold without an active piece")?.kind;
        let held = self.board.hold.replace(current);
        let next = if let Some(held) = held {
            held
        } else {
            self.board.take_from_bag(current);
            self.queue.pop_front();
            self.commit_reply =
                Some(thinker.empty_hold(self.board, self.queue.iter().copied().collect())?);
            let revealed = self.random.draw();
            self.queue.push_back(revealed);
            thinker.reveal(revealed)?;
            *self.queue.front().ok_or("empty queue after hold")?
        };
        self.active = self.board.spawn(next);
        self.hold_locked = true;
        self.gravity = 0;
        self.topout = self.active.is_none();
        Ok(())
    }

    fn color_lock(&mut self, piece: Piece) -> Result<(), String> {
        for (x, y) in piece.cells() {
            if (0..WIDTH as i32).contains(&x) && (0..HEIGHT as i32).contains(&y) {
                self.colors[y as usize][x as usize] = piece.kind as u8 + 1;
            }
        }
        let mut write = 0;
        for read in 0..HEIGHT {
            if self.colors[read].iter().all(|&n| n != 0) {
                continue;
            }
            self.colors[write] = self.colors[read];
            write += 1;
        }
        for row in &mut self.colors[write..] {
            *row = [0; WIDTH];
        }
        for (y, row) in self.colors.iter().enumerate() {
            let bits = row
                .iter()
                .enumerate()
                .fold(0, |bits, (x, &color)| bits | (u16::from(color != 0) << x));
            if bits != self.board.rows[y] {
                return Err("colored field differs from physical board".into());
            }
        }
        Ok(())
    }

    fn lock(&mut self, frame: u32, thinker: &Thinker) -> Result<LockResult, String> {
        let current = self.active.ok_or("hard drop without an active piece")?;
        let physical = self
            .board
            .sonic_drop(current)
            .ok_or("hard drop has no landing")?;
        let choice = self
            .controller
            .choice
            .clone()
            .ok_or("hard drop without selected path")?;
        if physical != choice.piece {
            return Err(format!(
                "input replay differs from selected placement: actual={physical}, selected={}",
                choice.piece
            ));
        }
        let empty_hold = choice.empty_hold;
        self.board
            .take_from_bag(*self.queue.front().ok_or("empty queue at lock")?);
        let result = self.board.lock(physical);
        self.color_lock(physical)?;
        self.queue.pop_front();
        self.attack += result.attack().max(0);
        self.lines += u32::from(result.lines());
        self.locks += 1;
        self.events.push(json!({"frame":frame,"type":"lock","piece":physical,"hold":choice.path.hold,
            "empty_hold":empty_hold,"lock":result,"input_path":choice.path.instructions,"search":thinker.read().stats}));
        self.commit_reply =
            Some(thinker.commit(physical, self.board, self.queue.iter().copied().collect())?);
        self.position += 1;
        self.active = None;
        self.controller = Controller::default();
        self.spawn_at = Some(
            frame
                + ENTRY_DELAY
                + if result.lines() > 0 && !result.perfect_clear {
                    CLEAR_DELAY
                } else {
                    0
                }
                + 1,
        );
        self.topout = result.death;
        Ok(result)
    }

    fn tick(&mut self, frame: u32, thinker: &Thinker) -> Result<(), String> {
        if let Some(reply) = &self.commit_reply {
            match reply.try_recv() {
                Ok(result) => {
                    result?;
                    self.commit_reply = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err("search stopped before confirming placement".into());
                }
            }
        }
        if self.spawn_at.is_some_and(|at| frame >= at) && self.commit_reply.is_none() {
            let revealed = self.random.draw();
            self.queue.push_back(revealed);
            thinker.reveal(revealed)?;
            self.active = self.queue.front().and_then(|&kind| self.board.spawn(kind));
            self.topout = self.active.is_none();
            self.spawn_at = None;
            self.hold_locked = false;
            self.gravity = 0;
            self.events
                .push(json!({"frame":frame,"type":"spawn","revealed":revealed}));
        }
        if self.active.is_none() || self.topout {
            return Ok(());
        }
        // A confirmed empty hold is acknowledged asynchronously. Wait to send
        // further inputs until its root transition is complete.
        if self.commit_reply.is_some() {
            return Ok(());
        }
        if self.controller.choice.is_none() {
            let snapshot = thinker.read();
            if snapshot.position == self.position
                && let Some(choice) = snapshot.choice
            {
                self.controller.install(choice);
            }
        }
        let active = self.active.unwrap();
        let grounded = self.board.collides(active.shifted(0, -1));
        match self.controller.input(grounded) {
            Input::Hold => {
                self.events.push(json!({"frame":frame,"type":"hold"}));
                return self.hold(thinker);
            }
            Input::HardDrop => {
                self.lock(frame, thinker)?;
                return Ok(());
            }
            Input::Tap(inst) => {
                let next = match inst {
                    Instruction::Left => Some(active.shifted(-1, 0)),
                    Instruction::Right => Some(active.shifted(1, 0)),
                    Instruction::Cw => self.board.rotate(active, true),
                    Instruction::Ccw => self.board.rotate(active, false),
                    _ => Some(active),
                };
                let next = next
                    .filter(|&piece| !self.board.collides(piece))
                    .ok_or_else(|| format!("input {inst:?} failed at frame {frame}"))?;
                self.active = Some(next);
                self.events
                    .push(json!({"frame":frame,"type":"tap","input":inst}));
            }
            Input::SoftDrop => {
                self.controller.soft_ticks += 1;
                if self.controller.soft_ticks >= 2 {
                    self.controller.soft_ticks = 0;
                    let current = self.active.unwrap();
                    if !self.board.collides(current.shifted(0, -1)) {
                        // SimpleBoard.SonicDrop clears a spin when it travels.
                        self.active = Some(Piece::new(
                            current.kind,
                            current.x,
                            current.y - 1,
                            current.rotation,
                        ));
                    }
                }
                self.gravity = 0;
                return Ok(());
            }
            Input::None => {}
        }
        self.gravity += 2;
        if self.gravity >= FPS {
            self.gravity -= FPS;
            let current = self.active.unwrap();
            if !self.board.collides(current.shifted(0, -1)) {
                self.active = Some(current.shifted(0, -1));
            }
        }
        Ok(())
    }
}

fn save(path: &FilePath, value: &Json) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

/// Run a native Hikari demonstration against an environment-only seven-bag.
/// `output` is the replay JSON file; progress.json is written beside it.
pub fn run(
    config: Config,
    seconds: u32,
    countdown_ms: u64,
    output: &FilePath,
) -> Result<Json, String> {
    config.validate()?;
    if config.previews < 2 || config.previews > 12 {
        return Err("Ultra requires 2..=12 public pieces including the active piece".into());
    }
    if seconds == 0 {
        return Err("Ultra duration must be positive".into());
    }
    let directory = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(FilePath::new("."));
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let mut game = Game::new(config.seed, 6.min(config.previews));
    let thinker = Thinker::new(config.clone(), game.queue.iter().copied().collect())?;
    let countdown_start = Instant::now();
    let progress_path = directory.join("progress.json");
    save(
        &progress_path,
        &json!({"phase":"countdown","seconds":seconds,"countdown_ms":countdown_ms,"config":config}),
    )?;
    while countdown_start.elapsed() < Duration::from_millis(countdown_ms) {
        thread::sleep(
            Duration::from_millis(5)
                .min(Duration::from_millis(countdown_ms).saturating_sub(countdown_start.elapsed())),
        );
    }
    let actual_countdown = countdown_start.elapsed().as_secs_f64();
    let start = Instant::now();
    let end_frame = seconds
        .checked_mul(FPS)
        .ok_or("Ultra duration overflows frame counter")?;
    let mut recording = Recording::new();
    let mut error = None;
    let mut final_frame = end_frame;
    let mut late_frames = 0u32;
    let mut maximum_lag = 0f64;
    recording.record(0, &game, thinker.read().stats);
    for frame in 0..end_frame {
        let deadline = start + Duration::from_secs_f64(f64::from(frame) / f64::from(FPS));
        if let Some(wait) = deadline.checked_duration_since(Instant::now()) {
            thread::sleep(wait);
        }
        let lag = Instant::now()
            .saturating_duration_since(deadline)
            .as_secs_f64();
        maximum_lag = maximum_lag.max(lag);
        if lag > 1.0 / f64::from(FPS) {
            late_frames += 1;
        }
        if let Err(reason) = game.tick(frame, &thinker) {
            error = Some(reason);
            final_frame = frame + 1;
        }
        recording.record(frame + 1, &game, thinker.read().stats);
        if (frame + 1) % (10 * FPS) == 0 || error.is_some() || game.topout {
            save(
                &progress_path,
                &json!({"phase":"playing","frame":frame+1,"seconds":seconds,
                "attack":game.attack,"lines":game.lines,"locks":game.locks,"stats":thinker.read().stats,
                "topout":game.topout,"error":error}),
            )?;
        }
        if error.is_some() || game.topout {
            final_frame = frame + 1;
            break;
        }
    }
    if final_frame == end_frame {
        let deadline = start + Duration::from_secs(u64::from(seconds));
        if let Some(wait) = deadline.checked_duration_since(Instant::now()) {
            thread::sleep(wait);
        }
    }
    // A final asynchronous commit can complete after the last game frame.
    if let Some(reply) = game.commit_reply.take() {
        match reply.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => {
                error = Some(reason);
            }
            Err(reason) => {
                error = Some(format!("final placement acknowledgement: {reason}"));
            }
        }
    }
    let stats = thinker.read().stats;
    let completed = final_frame == end_frame && !game.topout && error.is_none();
    let summary = json!({"end_frame":final_frame,"completed":completed,"attack":game.attack,"score":0,
        "score_available":false,"lines":game.lines,"locks":game.locks,"topout":game.topout,"error":error,
        "wall_seconds":start.elapsed().as_secs_f64(),"late_frames":late_frames,"maximum_lag_seconds":maximum_lag,"stats":stats});
    let result = json!({"metadata":{"engine":"Hikari Rust","reference_commit":crate::REFERENCE_COMMIT,
        "seed":config.seed,"seconds":seconds,"real_time":true,"fps":FPS,"countdown_ms":countdown_ms,
        "actual_countdown_seconds":actual_countdown,"config":config,"rules":"Hikari rules, realtime Ultra demo: SimpleBoard 24 rows/attack; 2-frame taps; soft drop 2 frames/cell; gravity 2 cells/sec; entry 3 frames + non-PC clear 38 frames; no opponent or garbage",
        "score_available":false,"public_piece_count_includes_active":true,"known_queue":6.min(config.previews),"visible_next":5.min(config.previews-1)},
        "summary":summary,"events":game.events,
        "playback":{"end":final_frame,"completed":completed,"engine":"Hikari Rust","seed":config.seed,
            "boards":recording.boards,"poses":recording.poses,"queues":recording.queues,"keys":recording.keys}});
    save(output, &result)?;
    save(
        &progress_path,
        &json!({"phase":"finished","summary":summary,"output":output}),
    )?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_bag_contains_each_piece_once_per_bag() {
        let mut bag = SevenBag::new(10_000_000);
        for _ in 0..100 {
            let mut kinds = (0..7).map(|_| bag.draw() as u8).collect::<Vec<_>>();
            kinds.sort_unstable();
            assert_eq!(kinds, (0..7).collect::<Vec<_>>());
        }
    }

    #[test]
    fn controller_taps_are_separated_and_soft_drop_waits_for_ground() {
        let piece = Piece::new(PieceKind::T, 0, 0, 0);
        let mut controller = Controller::default();
        controller.install(Choice {
            piece,
            empty_hold: false,
            path: Path {
                hold: true,
                hold_only: false,
                instructions: vec![Instruction::Left, Instruction::SonicDrop, Instruction::Cw],
                result: Some(piece),
                time: 0,
            },
        });
        assert!(matches!(controller.input(false), Input::Hold));
        assert!(matches!(
            controller.input(false),
            Input::Tap(Instruction::Left)
        ));
        assert!(matches!(controller.input(false), Input::None));
        assert!(matches!(controller.input(false), Input::SoftDrop));
        for _ in 0..4 {
            assert!(matches!(controller.input(false), Input::SoftDrop));
        }
        assert!(matches!(controller.input(true), Input::None));
        assert!(matches!(
            controller.input(true),
            Input::Tap(Instruction::Cw)
        ));
        assert!(matches!(controller.input(true), Input::None));
        assert!(matches!(controller.input(true), Input::HardDrop));
    }

    #[test]
    fn colored_rows_clear_with_the_physical_board() {
        let mut game = Game::new(1, 12);
        for x in 0..WIDTH {
            if x != 4 && x != 5 {
                game.board.rows[0] |= 1 << x;
                game.board.rows[1] |= 1 << x;
                game.colors[0][x] = 3;
                game.colors[1][x] = 3;
            }
        }
        assert_eq!(game.board.rows[0], crate::board::FILLED & !(3 << 4));
        let piece = Piece::new(PieceKind::O, 3, -1, 0);
        assert_eq!(game.board.lock(piece).lines(), 2);
        game.color_lock(piece).unwrap();
        assert_eq!(game.colors, [[0; WIDTH]; HEIGHT]);
    }

    #[test]
    fn empty_hold_is_confirmed_once_even_when_bag_boundary_repeats_kind() {
        let config = Config {
            max_nodes: 20_000,
            max_depth: 4,
            ..Config::default()
        };
        let mut game = Game::new(7, 12);
        let current = *game.queue.front().unwrap();
        game.queue[1] = current;
        let thinker = Thinker::new(config, game.queue.iter().copied().collect()).unwrap();
        game.hold(&thinker).unwrap();
        assert!(game.hold_locked);
        assert_eq!(game.active.unwrap().kind, current);
        assert_eq!(game.board.hold, Some(current));
        assert!(game.hold(&thinker).is_err());
    }
}
