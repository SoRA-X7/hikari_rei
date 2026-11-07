//! Original Mirai movement iterator and selected-move path reconstruction.
//!
//! Costs are Hikari input costs: repeated inputs add a frame, sonic drop
//! costs two frames per row, and the final hard drop has no added cost.
use crate::board::{Board, Piece, PieceKind, TSpin};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
        }
    }
    fn generate(&mut self, spawned: Piece) {
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
        while let Some(current) = self.next.pop() {
            let piece = current.piece;
            let origin = self.tree[&piece];
            let Some(dropped) = self.board.sonic_drop(piece) else {
                continue;
            };
            if origin.depth < MAX_INSTRUCTIONS {
                self.append(origin, Some(piece.shifted(-1, 0)), Instruction::Left, false);
                self.append(origin, Some(piece.shifted(1, 0)), Instruction::Right, false);
                if spawned.kind != PieceKind::O {
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
                if dropped.y != piece.y {
                    self.append(origin, Some(dropped), Instruction::SonicDrop, true);
                }
            }
            if self.deduplicator.insert((dropped.cells(), piece.tspin))
                && self.locked_keys.insert(dropped)
            {
                self.locked.push((dropped, current));
            }
        }
    }
    fn append(&mut self, origin: Step, result: Option<Piece>, inst: Instruction, skip_check: bool) {
        let Some(result) = result else {
            return;
        };
        if !skip_check && self.board.collides(result) {
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
        mirai.generate(spawned);
        mirai.rebuild(piece, hold)
    }
}

pub use path::find_path;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_mirai_generate_empty_o_golden() {
        // Exact fixture from Assets/AI/Tests/MiraiTests.cs GenerateEmptyO.
        let board = Board::default();
        let mut placements = generate(&board, PieceKind::O);
        placements.sort_by_key(|p| p.piece.x);
        let actual: Vec<_> = placements
            .into_iter()
            .map(|p| (p.piece.x, p.piece.y, p.piece.rotation, p.time))
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
}
