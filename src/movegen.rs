//! Original Mirai movement iterator and selected-move path reconstruction.
//!
//! Costs are Hikari input costs: repeated inputs add a frame, sonic drop
//! costs two frames per row, and the final hard drop has no added cost.
use crate::board::{Board, Piece, PieceKind, TSpin};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
#[cfg(feature = "fast-low-stack")]
use std::sync::OnceLock;

pub const MAX_INSTRUCTIONS: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum Instruction {
    #[default]
    None,
    Left,
    Right,
    Cw,
    Ccw,
    SonicDrop,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Path {
    pub hold: bool,
    pub hold_only: bool,
    pub instructions: Vec<Instruction>,
    pub result: Option<Piece>,
    pub time: i32,
}
impl Path {
    pub fn hold_only() -> Self {
        Self {
            hold: true,
            hold_only: true,
            instructions: Vec::new(),
            result: None,
            time: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub piece: Piece,
    pub time: i32,
}

#[derive(Clone, Copy, Debug)]
struct Step {
    parent: Option<Piece>,
    cost: i32,
    depth: usize,
    piece: Piece,
    inst: Instruction,
}

#[derive(Clone, Copy)]
struct StepRef {
    piece: Piece,
    cost: i32,
}

#[cfg(feature = "fast-low-stack")]
struct AirStarts {
    tree: HashMap<Piece, Step>,
    order: Vec<StepRef>,
}

/// Empty-space movement at the normal spawn height is independent of a low
/// stack. Cache the actual Mirai input paths rather than inventing cheaper
/// horizontal/rotation sequences or changing the coordinate convention.
#[cfg(feature = "fast-low-stack")]
fn air_starts(kind: PieceKind) -> &'static AirStarts {
    static STARTS: OnceLock<[AirStarts; 7]> = OnceLock::new();
    &STARTS.get_or_init(|| {
        std::array::from_fn(|index| {
            let board = Board::default();
            let kind = PieceKind::ALL[index];
            let spawned = Piece::spawn(kind);
            let mut mirai = Mirai::new(&board);
            mirai.tree.insert(
                spawned,
                Step {
                    parent: None,
                    cost: 0,
                    depth: 0,
                    piece: spawned,
                    inst: Instruction::None,
                },
            );
            mirai.next.push(StepRef {
                piece: spawned,
                cost: 0,
            });
            let mut order = Vec::with_capacity(40);
            while let Some(current) = mirai.next.pop() {
                let origin = mirai.tree[&current.piece];
                order.push(current);
                if origin.depth < MAX_INSTRUCTIONS {
                    mirai.append_air_moves(origin, kind);
                }
            }
            mirai.tree.shrink_to_fit();
            AirStarts {
                tree: mirai.tree,
                order,
            }
        })
    })[kind as usize]
}

/// The same binary min heap as `NativePriorityQueue<StepRef>`: equal costs do
/// not swap, and equal children prefer the left child. First-discovered steps
/// are retained by Mirai, so preserving ties also preserves path selection.
#[derive(Default)]
struct Queue(Vec<StepRef>);
impl Queue {
    fn push(&mut self, value: StepRef) {
        self.0.push(value);
        let mut index = self.0.len() - 1;
        while index != 0 {
            let parent = (index - 1) / 2;
            if self.0[index].cost < self.0[parent].cost {
                self.0.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }
    fn pop(&mut self) -> Option<StepRef> {
        if self.0.is_empty() {
            return None;
        }
        let value = self.0.swap_remove(0);
        let mut parent = 0;
        loop {
            let mut child = 2 * parent + 1;
            if child >= self.0.len() {
                break;
            }
            if child + 1 < self.0.len() && self.0[child].cost > self.0[child + 1].cost {
                child += 1;
            }
            if self.0[parent].cost > self.0[child].cost {
                self.0.swap(parent, child);
                parent = child;
            } else {
                break;
            }
        }
        Some(value)
    }
}

struct Mirai<'a> {
    board: &'a Board,
    tree: HashMap<Piece, Step>,
    next: Queue,
    locked: Vec<(Piece, StepRef)>,
    locked_keys: HashSet<Piece>,
    // Original uses a 32-bit hash of the four cells plus spin. Store the exact
    // cell set and spin here so unrelated placements cannot collide and vanish.
    deduplicator: HashSet<([(i32, i32); 4], TSpin)>,
    #[cfg(feature = "prune-low-stack")]
    prune_heights: Option<[u8; crate::board::WIDTH]>,
}
impl<'a> Mirai<'a> {
    fn new(board: &'a Board) -> Self {
        Self {
            board,
            tree: HashMap::with_capacity(500),
            next: Queue(Vec::with_capacity(500)),
            locked: Vec::with_capacity(100),
            locked_keys: HashSet::with_capacity(100),
            deduplicator: HashSet::with_capacity(100),
            #[cfg(feature = "prune-low-stack")]
            prune_heights: None,
        }
    }
    fn generate(&mut self, spawned: Piece) {
        self.generate_until(spawned, |_| false);
    }

    fn generate_until(&mut self, spawned: Piece, stop_after: impl Fn(Piece) -> bool) {
        #[cfg(feature = "prune-low-stack")]
        {
            self.prune_heights = None;
        }
        #[cfg(feature = "fast-low-stack")]
        if self.board.max_height() < 16 && spawned == Piece::spawn(spawned.kind) {
            self.generate_low_stack(spawned.kind, stop_after);
            return;
        }
        self.generate_full(spawned, stop_after);
    }
    fn generate_full(&mut self, spawned: Piece, stop_after: impl Fn(Piece) -> bool) {
        if self.tree.contains_key(&spawned) {
            return;
        }
        self.deduplicator.clear();
        let root = Step {
            parent: None,
            cost: 0,
            depth: 0,
            piece: spawned,
            inst: Instruction::None,
        };
        self.tree.insert(spawned, root);
        self.next.push(StepRef {
            piece: spawned,
            cost: 0,
        });
        self.drain_queue(spawned.kind, stop_after);
    }
    #[cfg(feature = "fast-low-stack")]
    fn generate_low_stack(&mut self, kind: PieceKind, stop_after: impl Fn(Piece) -> bool) {
        if self.tree.contains_key(&Piece::spawn(kind)) {
            return;
        }
        self.deduplicator.clear();
        let starts = air_starts(kind);
        if self.tree.is_empty() {
            self.tree.clone_from(&starts.tree);
        } else {
            self.tree.extend(starts.tree.iter().map(|(&p, &s)| (p, s)));
        }
        // All cached paths are clear when max_height < 16: their lowest
        // occupied cell is at least row 17. Register their direct hard drops,
        // then retain the original cost/depth gates for further stack moves.
        for &current in &starts.order {
            let origin = self.tree[&current.piece];
            let Some(dropped) = self.board.sonic_drop(current.piece) else {
                continue;
            };
            if origin.depth < MAX_INSTRUCTIONS && dropped.y != current.piece.y {
                self.append(origin, Some(dropped), Instruction::SonicDrop, true);
            }
            let previous_locks = self.locked.len();
            self.record_lock(dropped, current);
            if stop_after(dropped) && self.locked.len() > previous_locks {
                return;
            }
        }
        // Cached-only mode retains the full stack exploration, although
        // seeding can change first-discovery ties and placement order.
        #[cfg(feature = "prune-low-stack")]
        {
            // The optional stronger experiment mirrors Cold-clear's fast
            // mode: after the initial surface seeds, stop non-spin routes
            // that return entirely above the column tops. Such states are
            // usually covered by another seed, but kicks and subsequent
            // tucks can require them, so this pruning is not complete.
            self.prune_heights = Some(std::array::from_fn(|x| {
                self.board
                    .rows
                    .iter()
                    .rposition(|row| row & (1 << x) != 0)
                    .map_or(0, |y| (y + 1) as u8)
            }));
        }
        self.drain_queue(kind, stop_after);
    }
    fn drain_queue(&mut self, kind: PieceKind, stop_after: impl Fn(Piece) -> bool) {
        while let Some(current) = self.next.pop() {
            let piece = current.piece;
            let origin = self.tree[&piece];
            let Some(dropped) = self.board.sonic_drop(piece) else {
                continue;
            };
            if origin.depth < MAX_INSTRUCTIONS {
                self.append_air_moves(origin, kind);
                if dropped.y != piece.y {
                    self.append(origin, Some(dropped), Instruction::SonicDrop, true);
                }
            }
            let previous_locks = self.locked.len();
            self.record_lock(dropped, current);
            if stop_after(dropped) && self.locked.len() > previous_locks {
                return;
            }
        }
    }
    fn append_air_moves(&mut self, origin: Step, kind: PieceKind) {
        let piece = origin.piece;
        self.append(origin, Some(piece.shifted(-1, 0)), Instruction::Left, false);
        self.append(origin, Some(piece.shifted(1, 0)), Instruction::Right, false);
        if kind != PieceKind::O {
            self.append(
                origin,
                self.board.rotate(piece, true),
                Instruction::Cw,
                true,
            );
            self.append(
                origin,
                self.board.rotate(piece, false),
                Instruction::Ccw,
                true,
            );
        }
    }
    fn record_lock(&mut self, dropped: Piece, current: StepRef) {
        if self
            .deduplicator
            .insert((dropped.cells(), current.piece.tspin()))
            && self.locked_keys.insert(dropped)
        {
            self.locked.push((dropped, current));
        }
    }
    fn append(&mut self, origin: Step, result: Option<Piece>, inst: Instruction, skip_check: bool) {
        let Some(result) = result else {
            return;
        };
        if !skip_check && self.board.collides(result) {
            return;
        }
        #[cfg(feature = "prune-low-stack")]
        if result.tspin() == TSpin::None
            && self.prune_heights.as_ref().is_some_and(|heights| {
                result
                    .cells()
                    .iter()
                    .all(|&(x, y)| y >= heights[x as usize] as i32)
            })
        {
            return;
        }
        let mut time = if inst == Instruction::SonicDrop {
            2 * (origin.piece.y as i32 - result.y as i32)
        } else {
            1
        };
        if origin.inst == inst {
            time += 1;
        }
        let step = Step {
            parent: Some(origin.piece),
            cost: origin.cost + time,
            depth: origin.depth + 1,
            piece: result,
            inst,
        };
        if let std::collections::hash_map::Entry::Vacant(entry) = self.tree.entry(result) {
            entry.insert(step);
            // The depth-16 state is stored for reconstruction but never expanded.
            if step.depth < MAX_INSTRUCTIONS && (result.kind == PieceKind::T || step.cost < 20) {
                self.next.push(StepRef {
                    piece: result,
                    cost: step.cost,
                });
            }
        }
    }
    fn rebuild(&self, to: Piece, hold: bool) -> Option<Path> {
        let (_, current) = self.locked.iter().find(|(piece, _)| *piece == to)?;
        let mut instructions = Vec::new();
        let mut leaf = current.piece;
        while let Some(parent) = self.tree[&leaf].parent {
            instructions.push(self.tree[&leaf].inst);
            leaf = parent;
        }
        instructions.reverse();
        Some(Path {
            hold,
            hold_only: false,
            instructions,
            result: Some(to),
            time: current.cost,
        })
    }
}

pub fn generate(board: &Board, kind: PieceKind) -> Vec<Placement> {
    let Some(spawned) = board.spawn(kind) else {
        return Vec::new();
    };
    let mut mirai = Mirai::new(board);
    mirai.generate(spawned);
    mirai
        .locked
        .into_iter()
        .map(|(piece, step)| Placement {
            piece,
            time: step.cost,
        })
        .collect()
}

pub mod path {
    use super::*;
    pub fn find_path(board: &Board, piece: Piece, hold: bool) -> Option<Path> {
        let spawned = board.spawn(piece.kind)?;
        let mut mirai = Mirai::new(board);
        // A geometrically equivalent lock can shadow the requested rotation.
        // Stop only after the exact piece has been accepted by both deduplicators.
        mirai.generate_until(spawned, |locked| locked == piece);
        mirai.rebuild(piece, hold)
    }
}

pub use path::find_path;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "fast-low-stack")]
    fn check_replay(board: &Board, path: &Path) {
        let target = path.result.unwrap();
        let mut current = board.spawn(target.kind).unwrap();
        let mut time = 0;
        let mut previous = Instruction::None;
        assert!(path.instructions.len() < MAX_INSTRUCTIONS);
        for &instruction in &path.instructions {
            let from = current;
            current = match instruction {
                Instruction::None => current,
                Instruction::Left => current.shifted(-1, 0),
                Instruction::Right => current.shifted(1, 0),
                Instruction::Cw => board.rotate(current, true).unwrap(),
                Instruction::Ccw => board.rotate(current, false).unwrap(),
                Instruction::SonicDrop => board.sonic_drop(current).unwrap(),
            };
            assert!(!board.collides(current), "{target}: {instruction:?}");
            time += if instruction == Instruction::SonicDrop {
                2 * (from.y as i32 - current.y as i32)
            } else {
                1
            };
            if instruction == previous {
                time += 1;
            }
            previous = instruction;
        }
        assert_eq!(board.sonic_drop(current), Some(target));
        assert_eq!(path.time, time);
        assert!(target.kind == PieceKind::T || time < 20);
    }
    #[test]
    fn targeted_path_search_matches_exhaustive_paths() {
        for height in [0, 4, 10, 15, 16, 19] {
            let mut board = Board::default();
            for (y, row) in board.rows[..height].iter_mut().enumerate() {
                *row = if y % 3 == 0 {
                    0b1100111011
                } else {
                    0b1000111001
                };
            }
            for kind in PieceKind::ALL {
                let Some(spawned) = board.spawn(kind) else {
                    continue;
                };
                let mut original = Mirai::new(&board);
                original.generate(spawned);
                for &(piece, _) in &original.locked {
                    assert_eq!(
                        find_path(&board, piece, true),
                        original.rebuild(piece, true)
                    );
                }
                assert_eq!(find_path(&board, Piece::new(kind, 3, 24, 0), false), None);
            }
        }
    }

    #[test]
    fn targeted_search_stops_at_accepted_lock_and_skips_shadowed_rotation() {
        let board = Board::default();
        let spawn = Piece::spawn(PieceKind::I);
        let canonical = Piece::new(PieceKind::I, 3, -2, 0);
        let shadowed = Piece::new(PieceKind::I, 3, -1, 2);
        assert_eq!(canonical.cells(), shadowed.cells());

        let mut early = Mirai::new(&board);
        early.generate_until(spawn, |piece| piece == canonical);
        assert_eq!(early.locked.len(), 1);
        assert_eq!(early.locked[0].0, canonical);
        assert_eq!(find_path(&board, canonical, false).unwrap().time, 0);
        assert_eq!(find_path(&board, shadowed, false), None);
        assert_eq!(
            find_path(&board, canonical.with_tspin(TSpin::Full), false),
            None
        );
    }

    #[test]
    fn original_mirai_generate_empty_o_golden() {
        // Exact fixture from Assets/AI/Tests/MiraiTests.cs GenerateEmptyO.
        let board = Board::default();
        let mut placements = generate(&board, PieceKind::O);
        placements.sort_by_key(|p| p.piece.x);
        let actual: Vec<_> = placements
            .into_iter()
            .map(|p| (p.piece.x, p.piece.y, p.piece.rotation(), p.time))
            .collect();
        assert_eq!(
            actual,
            vec![
                (-1, -1, 0, 7),
                (0, -1, 0, 5),
                (1, -1, 0, 3),
                (2, -1, 0, 1),
                (3, -1, 0, 0),
                (4, -1, 0, 1),
                (5, -1, 0, 3),
                (6, -1, 0, 5),
                (7, -1, 0, 7)
            ]
        );
    }
    #[test]
    fn original_mirai_generate_dead_moves_golden() {
        // Exact board and manually chosen spawn from GenerateDeadMoves.
        let mut board = Board::default();
        board.lock(Piece::new(PieceKind::I, 1, 17, 0));
        board.lock(Piece::new(PieceKind::I, 5, 17, 0));
        let mut mirai = Mirai::new(&board);
        mirai.generate(Piece::new(PieceKind::O, 3, 19, 0));
        assert_eq!(mirai.locked.len(), 9);
        for (piece, _) in mirai.locked {
            let mut clone = board;
            assert!(clone.lock(piece).death);
        }
    }
    #[test]
    fn every_empty_placement_has_replayable_selected_path() {
        let board = Board::default();
        for kind in PieceKind::ALL {
            let placements = generate(&board, kind);
            let expected_count = match kind {
                PieceKind::O => 9,
                PieceKind::I | PieceKind::S | PieceKind::Z => 17,
                _ => 34,
            };
            assert_eq!(placements.len(), expected_count, "{kind}");
            for placement in placements {
                let path = find_path(&board, placement.piece, true).unwrap();
                let mut current = board.spawn(kind).unwrap();
                for inst in path.instructions {
                    current = match inst {
                        Instruction::None => current,
                        Instruction::Left => current.shifted(-1, 0),
                        Instruction::Right => current.shifted(1, 0),
                        Instruction::Cw => board.rotate(current, true).unwrap(),
                        Instruction::Ccw => board.rotate(current, false).unwrap(),
                        Instruction::SonicDrop => board.sonic_drop(current).unwrap(),
                    };
                    assert!(!board.collides(current));
                }
                assert_eq!(board.sonic_drop(current), Some(placement.piece));
                assert_eq!(path.time, placement.time);
            }
        }
    }
    #[cfg(feature = "fast-low-stack")]
    #[test]
    fn cached_air_paths_stay_above_the_low_stack_and_replay() {
        let mut board = Board::default();
        board.rows[14] = 0b1010101010;
        assert_eq!(board.max_height(), 15);
        for kind in PieceKind::ALL {
            let starts = air_starts(kind);
            for current in &starts.order {
                assert!(current.piece.cells().iter().all(|&(_, y)| y >= 17));
                let mut instructions = Vec::new();
                let mut leaf = current.piece;
                while let Some(parent) = starts.tree[&leaf].parent {
                    instructions.push(starts.tree[&leaf].inst);
                    leaf = parent;
                }
                instructions.reverse();
                assert!(!instructions.contains(&Instruction::SonicDrop));
                check_replay(
                    &board,
                    &Path {
                        hold: false,
                        hold_only: false,
                        instructions,
                        result: board.sonic_drop(current.piece),
                        time: current.cost,
                    },
                );
            }
        }
    }
    #[cfg(feature = "fast-low-stack")]
    #[test]
    fn high_stack_fallback_keeps_full_mirai_order_costs_and_paths() {
        let mut board = Board::default();
        board.rows[15] = 0b1000010001;
        assert_eq!(board.max_height(), 16);
        for kind in PieceKind::ALL {
            let spawned = board.spawn(kind).unwrap();
            let mut full = Mirai::new(&board);
            full.generate_full(spawned, |_| false);
            let mut fallback = Mirai::new(&board);
            fallback.generate(spawned);
            let placements = |mirai: &Mirai<'_>| {
                mirai
                    .locked
                    .iter()
                    .map(|&(piece, step)| Placement {
                        piece,
                        time: step.cost,
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(placements(&fallback), placements(&full), "{kind}");
            for &(piece, _) in &full.locked {
                assert_eq!(fallback.rebuild(piece, true), full.rebuild(piece, true));
            }
        }
    }
    #[cfg(feature = "fast-low-stack")]
    #[test]
    fn seeded_stack_moves_keep_replayable_paths_and_original_cost_limits() {
        for height in [4, 10, 15] {
            let mut board = Board::default();
            for y in 0..height {
                board.rows[y] = if y % 3 == 0 {
                    0b1100111011
                } else {
                    0b1000111001
                };
            }
            for kind in PieceKind::ALL {
                for placement in generate(&board, kind) {
                    let path = find_path(&board, placement.piece, true).unwrap();
                    assert_eq!(placement.time, path.time);
                    check_replay(&board, &path);
                }
            }
        }
    }
    #[cfg(feature = "prune-low-stack")]
    #[test]
    fn pruned_surface_search_keeps_replayable_drops_and_tspin_routes() {
        let mut board = Board::default();
        board.rows[0] = 1 << 3;
        board.rows[2] = (1 << 3) | (1 << 5);
        let target = Piece::new(PieceKind::T, 3, 0, 0).with_tspin(TSpin::Full);
        let t_moves = generate(&board, PieceKind::T);
        assert!(t_moves.iter().any(|placement| placement.piece == target));
        for kind in PieceKind::ALL {
            let placements = generate(&board, kind);
            assert!(!placements.is_empty());
            for placement in placements {
                let path = find_path(&board, placement.piece, false).unwrap();
                assert_eq!(placement.time, path.time);
                check_replay(&board, &path);
            }
        }
    }
}
