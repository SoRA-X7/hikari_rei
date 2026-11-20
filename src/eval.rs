//! Hikari's deterministic integer evaluator.
//!
//! Ported from Assets/AI/Eval at Hikari revision
//! 90aef085a53d6048cbd19de7f4c9c53a6f3c8960. Deliberate source quirks
//! (zero based column tops, the initial bumpiness offset, and passing column
//! bitsets to the cavity calculation) are retained for evaluation parity.

use crate::board::{Board, LockResult, Piece, PieceKind, TSpin};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul};

/// Rust's fixed array replaces the original unsafe ten-integer `int10` struct.
pub type Int10 = [i32; 10];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reward {
    pub evaluation: i32,
    /// -1 interrupts a spike; 0 (including a hold-only move) preserves it.
    pub attack: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Value {
    /// Field safety, field power, accumulated move reward, and a spare lane.
    pub value: [i32; 4],
    pub spike: i32,
}

impl Value {
    #[inline]
    pub fn score(self) -> i32 {
        self.value.iter().copied().fold(0, i32::wrapping_add)
    }

    /// Score and spike are improved independently, exactly as in Value.cs.
    #[inline]
    pub fn improve(self, next: Self) -> Self {
        Self {
            value: if next.score() > self.score() {
                next.value
            } else {
                self.value
            },
            spike: self.spike.max(next.spike),
        }
    }

    #[inline]
    pub fn with_reward(mut self, reward: Reward) -> Self {
        self.value[2] = self.value[2].wrapping_add(reward.evaluation);
        self.spike = if reward.attack < 0 {
            0
        } else {
            self.spike.wrapping_add(reward.attack)
        };
        self
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            value: std::array::from_fn(|i| self.value[i].wrapping_add(rhs.value[i])),
            spike: self.spike.wrapping_add(rhs.spike),
        }
    }
}

impl Add<Reward> for Value {
    type Output = Self;

    fn add(self, rhs: Reward) -> Self {
        self.with_reward(rhs)
    }
}

impl Mul<i32> for Value {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self {
        Self {
            value: self.value.map(|v| v.wrapping_mul(rhs)),
            // Source multiplication and division intentionally omit spike.
            spike: 0,
        }
    }
}

impl Div<i32> for Value {
    type Output = Self;

    fn div(self, rhs: i32) -> Self {
        Self {
            value: self.value.map(|v| v / rhs),
            spike: 0,
        }
    }
}

impl Mul<i32> for Reward {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self {
        Self {
            evaluation: self.evaluation.wrapping_mul(rhs),
            attack: 0,
        }
    }
}

impl Div<i32> for Reward {
    type Output = Self;

    fn div(self, rhs: i32) -> Self {
        Self {
            evaluation: self.evaluation / rhs,
            attack: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Weights {
    pub clear1: i32,
    pub clear2: i32,
    pub clear3: i32,
    pub clear4: i32,
    pub t_spin1: i32,
    pub t_spin2: i32,
    pub t_spin3: i32,
    pub t_mini1: i32,
    pub t_mini2: i32,
    pub perfect: i32,
    pub t_hole: i32,
    pub tst_hole: i32,
    pub fin_hole: i32,
    pub b2b_continue: i32,
    pub b2b_destroy: i32,
    pub ren: i32,
    pub wasted_t: i32,
    pub hold_t: i32,
    pub bump_sum: i32,
    pub bump_sum_sq: i32,
    pub max_height_diff: i32,
    pub hole_depth: i32,
    pub max_height: i32,
    pub top50: i32,
    pub top75: i32,
    pub danger: i32,
    pub placement_height: i32,
    pub move_time: i32,
    pub bridge: i32,
    pub downstack: i32,
    pub cavities: i32,
    pub cavities_sq: i32,
    pub cavities_move: i32,
    pub overhangs: i32,
    pub overhangs_sq: i32,
    pub overhangs_move: i32,
    pub covered_cells: i32,
    pub covered_cells_sq: i32,
    pub row_transitions: i32,
    pub well_x: Int10,
    pub no_tspin: bool,
}

impl Weights {
    pub const fn zeroed() -> Self {
        Self {
            clear1: 0,
            clear2: 0,
            clear3: 0,
            clear4: 0,
            t_spin1: 0,
            t_spin2: 0,
            t_spin3: 0,
            t_mini1: 0,
            t_mini2: 0,
            perfect: 0,
            t_hole: 0,
            tst_hole: 0,
            fin_hole: 0,
            b2b_continue: 0,
            b2b_destroy: 0,
            ren: 0,
            wasted_t: 0,
            hold_t: 0,
            bump_sum: 0,
            bump_sum_sq: 0,
            max_height_diff: 0,
            hole_depth: 0,
            max_height: 0,
            top50: 0,
            top75: 0,
            danger: 0,
            placement_height: 0,
            move_time: 0,
            bridge: 0,
            downstack: 0,
            cavities: 0,
            cavities_sq: 0,
            cavities_move: 0,
            overhangs: 0,
            overhangs_sq: 0,
            overhangs_move: 0,
            covered_cells: 0,
            covered_cells_sq: 0,
            row_transitions: 0,
            well_x: [0; 10],
            no_tspin: false,
        }
    }

    pub const fn changed() -> Self {
        Self {
            clear1: -610,
            clear2: -555,
            clear3: -370,
            clear4: 500,
            t_spin1: 130,
            t_spin2: 540,
            t_spin3: 920,
            t_mini1: -680,
            t_mini2: -500,
            perfect: 2800,
            t_hole: 140,
            tst_hole: 160,
            fin_hole: 150,
            b2b_continue: 540,
            b2b_destroy: -350,
            ren: 60,
            wasted_t: -250,
            hold_t: 10,
            bump_sum: 130,
            bump_sum_sq: -20,
            hole_depth: 50,
            max_height: -70,
            top50: -50,
            top75: -370,
            danger: -30,
            move_time: -4,
            cavities: -200,
            overhangs: -90,
            covered_cells: -120,
            row_transitions: -21,
            well_x: [150, -100, 300, 20, 60, 60, 15, 280, -110, 140],
            ..Self::zeroed()
        }
    }

    pub const fn cc_like() -> Self {
        Self {
            clear1: -143,
            clear2: -100,
            clear3: -58,
            clear4: 490,
            t_spin1: 221,
            t_spin2: 510,
            t_spin3: 702,
            t_mini1: -58,
            t_mini2: 7,
            perfect: 999,
            t_hole: 96,
            tst_hole: 203,
            fin_hole: 200,
            b2b_continue: 52,
            b2b_destroy: -100,
            ren: 150,
            wasted_t: -152,
            bump_sum: -24,
            bump_sum_sq: -7,
            hole_depth: 57,
            max_height: -39,
            top50: -150,
            top75: -511,
            danger: -11,
            move_time: -3,
            cavities: -173,
            cavities_sq: -3,
            overhangs: -34,
            overhangs_sq: -1,
            covered_cells: -17,
            covered_cells_sq: -1,
            row_transitions: -5,
            well_x: [20, 23, 20, 50, 59, 21, 59, 10, -10, 24],
            ..Self::zeroed()
        }
    }

    pub const fn all_clear() -> Self {
        Self {
            perfect: 60000,
            move_time: -200,
            max_height: -1000,
            placement_height: -1000,
            clear1: -1000,
            clear2: 1400,
            clear3: 1600,
            clear4: 6000,
            no_tspin: true,
            ..Self::zeroed()
        }
    }
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            clear1: -500,
            clear2: -375,
            clear3: -160,
            clear4: 800,
            t_spin1: 220,
            t_spin2: 1000,
            t_spin3: 1270,
            t_mini1: -880,
            t_mini2: -500,
            perfect: 3800,
            t_hole: 200,
            tst_hole: 180,
            fin_hole: 150,
            b2b_continue: 340,
            b2b_destroy: -380,
            ren: 100,
            wasted_t: -300,
            hold_t: 10,
            bump_sum: 20,
            bump_sum_sq: -10,
            hole_depth: 100,
            max_height: 20,
            top50: -60,
            top75: -300,
            danger: -20,
            move_time: -6,
            cavities: -400,
            cavities_sq: -10,
            overhangs: -170,
            overhangs_sq: -2,
            row_transitions: -5,
            well_x: [150, -100, 300, 20, 60, 60, 15, 280, -110, 140],
            ..Self::zeroed()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TSpinHole {
    pub lines: i32,
    pub x: i32,
    pub y: i32,
    pub rotation: u8,
}

#[derive(Clone, Copy)]
pub(crate) enum CavityMode {
    Reference,
    Geometric,
}

pub fn evaluate_board(board: &Board, w: &Weights) -> Value {
    evaluate_board_with_cavities(board, w, CavityMode::Reference)
}

pub(crate) fn evaluate_board_with_cavities(
    board: &Board,
    w: &Weights,
    cavity_mode: CavityMode,
) -> Value {
    let mut clone = *board;
    let mut safety = 0;
    let mut power = 0;
    let (mut columns, mut heights) = columns(&clone);
    let mut max_height = max_height(&clone, 0x3ff);

    if !w.no_tspin {
        // Bag.Contains deliberately inspects the raw bag, while an exhausted
        // bag contributes through Count <= 3, just as in the original.
        let t_count = i32::from(clone.bag & (1 << PieceKind::T as u8) != 0)
            + i32::from(clone.hold == Some(PieceKind::T))
            + i32::from(clone.bag.count_ones() <= 3);
        for _ in 0..t_count {
            let hole = if let Some(hole) = scan_tspin(&clone, &heights, max_height) {
                power += hole.lines * w.t_hole;
                hole
            } else if let Some(hole) = scan_tst(&clone, &heights, max_height) {
                power += hole.lines * w.tst_hole;
                hole
            } else {
                break;
            };
            clone.lock(
                Piece::new(PieceKind::T, hole.x as i8, hole.y as i8, hole.rotation)
                    .with_tspin(TSpin::Full),
            );
            (columns, heights) = self::columns(&clone);
            max_height -= hole.lines;
        }
    }

    let (well, depth) = hole_position(&clone, &heights);
    let (bump, bump_sq) = bumpiness(&heights, well);
    safety += bump * w.bump_sum + bump_sq * w.bump_sum_sq;
    power += depth * w.hole_depth;
    if depth >= 2 {
        power += w.well_x[well];
    }

    if w.max_height_diff != 0 {
        let mut max_diff = 0;
        // The source intentionally stops before the 8/9 column pair.
        for x in 0..8 {
            if x == well || x + 1 == well {
                continue;
            }
            max_diff = max_diff.max((heights[x + 1] - heights[x]).abs());
        }
        safety += max_diff * w.max_height_diff;
    }
    safety += max_height * w.max_height;
    safety += (max_height - 10).max(0) * w.top50;
    safety += (max_height - 15).max(0) * w.top75;

    let thresholds = match cavity_mode {
        // The reference compares row indices against whole column masks.
        CavityMode::Reference => columns,
        CavityMode::Geometric => geometric_tops(&columns, &heights),
    };
    let (cavities, overhangs) = cavities_and_overhangs(&columns, &thresholds, max_height);
    safety += cavities * w.cavities + cavities * cavities * w.cavities_sq;
    safety += overhangs * w.overhangs + overhangs * overhangs * w.overhangs_sq;
    if w.covered_cells != 0 || w.covered_cells_sq != 0 {
        let (covered, covered_sq) = covered_cells(&clone, &heights);
        safety += covered * w.covered_cells + covered_sq * w.covered_cells_sq;
    }

    if w.row_transitions != 0 {
        let transitions: i32 = clone
            .rows
            .iter()
            .map(|&row| {
                let row = u32::from(row);
                ((row | (1 << 10)) ^ (1 | (row << 1))).count_ones() as i32
            })
            .sum();
        safety += transitions * w.row_transitions;
    }
    power += floor_log2(clone.ren) * w.downstack;

    Value {
        value: [safety, power, 0, 0],
        spike: 0,
    }
}

pub fn evaluate_move(
    mut time: i32,
    piece: Option<Piece>,
    board: &Board,
    placement: &LockResult,
    parent_is_b2b: bool,
    w: &Weights,
) -> Reward {
    let line_clear = placement.placement & 7 != 0;
    let continuous = placement.placement >= 4;
    let mut evaluation = 0;
    if let Some(piece) = piece {
        if !placement.perfect_clear && line_clear {
            time += 40;
        }
        evaluation += time * w.move_time;
        if w.danger != 0 {
            evaluation += center_height_excess(board) * time * w.danger;
        }
        if line_clear && placement.back_to_back {
            evaluation += w.b2b_continue;
        }
        if parent_is_b2b && !placement.back_to_back {
            evaluation += w.b2b_destroy;
        }
        evaluation += i32::from(piece.y) * w.placement_height;
        if placement.perfect_clear {
            evaluation += w.perfect;
        } else {
            let ren_attack = ren_attack(placement.ren - 1);
            evaluation += w.ren * ren_attack * ren_attack;
            evaluation += match placement.placement {
                1 => w.clear1,
                2 => w.clear2,
                3 => w.clear3,
                4 => w.clear4,
                9 => w.t_mini1,
                10 => w.t_mini2,
                17 => w.t_spin1,
                18 => w.t_spin2,
                19 => w.t_spin3,
                _ => 0,
            };
        }
        if piece.kind == PieceKind::T && !(placement.placement & 0b10000 != 0 && line_clear) {
            evaluation += w.wasted_t;
        }
    }
    if board.hold == Some(PieceKind::T) && !continuous {
        evaluation += w.hold_t;
    }
    Reward {
        evaluation,
        attack: placement.attack(),
    }
}

#[inline]
fn floor_log2(value: i32) -> i32 {
    31 - (value as u32).leading_zeros() as i32
}

#[inline]
fn ren_attack(ren: i32) -> i32 {
    const ATTACKS: [i32; 12] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5];
    ATTACKS[ren.clamp(0, 11) as usize]
}

/// `tops` are zero based, with zero also used for an empty column.
fn columns(board: &Board) -> ([i32; 10], [i32; 10]) {
    let (columns, tops) = board.columns();
    (columns.map(|column| column as i32), tops.map(i32::from))
}

fn max_height(board: &Board, filter: u16) -> i32 {
    board
        .rows
        .iter()
        .rposition(|row| row & filter != 0)
        .unwrap_or(0) as i32
}

fn center_height_excess(board: &Board) -> i32 {
    // Row 15 and everything below it have no danger penalty.
    board.rows[16..]
        .iter()
        .rposition(|row| row & 0b0001111000 != 0)
        .map_or(0, |row| row as i32 + 1)
}

fn hole_position(board: &Board, heights: &[i32; 10]) -> (usize, i32) {
    let mut hole = 0;
    for x in 1..10 {
        if heights[x] <= heights[hole] {
            hole = x;
        }
    }
    let mut depth = 0;
    for y in heights[hole]..24 {
        if board.rows[y as usize].count_ones() == 9 && !board.occupied(hole as i32, y) {
            depth += 1;
        } else {
            break;
        }
    }
    (hole, depth)
}

fn bumpiness(heights: &[i32; 10], well: usize) -> (i32, i32) {
    let mut bump: i32 = -1;
    let mut sq: i32 = -1;
    let mut prev: i32 = 0;
    for (x, &height) in heights.iter().enumerate() {
        if x == well {
            continue;
        }
        let dh = (prev - height).abs();
        bump += dh;
        sq += dh * dh;
        prev = height;
    }
    (bump.abs(), sq.abs())
}

fn covered_cells(board: &Board, heights: &[i32; 10]) -> (i32, i32) {
    let mut count = 0;
    let mut sq = 0;
    for (x, &height) in heights.iter().enumerate() {
        for y in (0..=height - 3).rev() {
            if !board.occupied(x as i32, y) {
                let cells = height - y - 1;
                count += cells;
                sq += cells * cells;
            }
        }
    }
    (count, sq)
}

fn geometric_tops(columns: &Int10, heights: &Int10) -> Int10 {
    // Row zero can contain a block. Track an empty column below that row.
    std::array::from_fn(|x| if columns[x] == 0 { -1 } else { heights[x] })
}

fn cavities_and_overhangs(columns: &Int10, thresholds: &Int10, height: i32) -> (i32, i32) {
    let height = height.max(0);
    let mut empty_cells = 0;
    let mut overhangs = 0;
    for (x, &column) in columns.iter().enumerate() {
        let limit = height.min(thresholds[x]).max(0);
        let empty = ((1u32 << limit) - 1) & !(column as u32);
        let left = if x >= 2 {
            (thresholds[x - 1] + 1).max(thresholds[x - 2])
        } else {
            height
        };
        let right = if x < 8 {
            (thresholds[x + 1] + 1).max(thresholds[x + 2])
        } else {
            height
        };
        let overhang_start = height.min(left).min(right);
        empty_cells += empty.count_ones() as i32;
        overhangs += (empty >> overhang_start).count_ones() as i32;
    }
    (empty_cells - overhangs, overhangs)
}

fn shape_collides(board: &Board, shape: [u8; 4], x: i32, y: i32) -> bool {
    for (dy, row) in shape.into_iter().enumerate() {
        for dx in 0..4 {
            if row & (1 << dx) != 0 && board.occupied(x + dx, y + dy as i32) {
                return true;
            }
        }
    }
    false
}

fn scan_tspin(board: &Board, heights: &[i32; 10], height: i32) -> Option<TSpinHole> {
    let mut best: Option<TSpinHole> = None;
    for y in (0..=height - 3).rev() {
        for x in 0..8 {
            if shape_collides(board, [0b010, 0b111, 0, 0], x, y)
                || !board.occupied(x, y)
                || !board.occupied(x + 2, y)
            {
                continue;
            }
            if board.occupied(x, y + 2) {
                if heights[x as usize + 1] > y + 1 || heights[x as usize + 2] > y + 1 {
                    continue;
                }
            } else if board.occupied(x + 2, y + 2) {
                if heights[x as usize + 1] > y + 1 || heights[x as usize] > y + 1 {
                    continue;
                }
            } else {
                continue;
            }
            let lines = i32::from(board.rows[y as usize].count_ones() == 9)
                + i32::from(board.rows[y as usize + 1].count_ones() == 7);
            if lines != 0 && best.is_none_or(|hole| hole.lines < lines) {
                best = Some(TSpinHole {
                    lines,
                    x,
                    y,
                    rotation: 2,
                });
            }
        }
    }
    best
}

fn scan_tst(board: &Board, heights: &[i32; 10], height: i32) -> Option<TSpinHole> {
    let mut best: Option<TSpinHole> = None;
    for y in (0..=height - 5).rev() {
        for x in -1..8 {
            let rotation = if !shape_collides(board, [0b010, 0b110, 0b010, 0b110], x, y)
                && !board.occupied(x + 3, y + 3)
                && board.occupied(x + 1, y + 4)
                && board.occupied(x + 2, y + 2)
                && (board.occupied(x + 2, y) || board.occupied(x + 1, y - 1))
                && board.occupied(x + 4, y + 3) == board.occupied(x + 4, y + 4)
            {
                if x + 3 < 10
                    && (heights[(x + 2) as usize] > y + 2 || heights[(x + 3) as usize] > y + 2)
                {
                    continue;
                }
                if x + 4 < 10 && !board.occupied(x + 4, y + 3) && heights[(x + 4) as usize] > y + 2
                {
                    continue;
                }
                1
            } else if !shape_collides(board, [0b010, 0b011, 0b010, 0b011], x, y)
                && !board.occupied(x - 1, y + 3)
                && board.occupied(x + 1, y + 4)
                && board.occupied(x, y + 2)
                && (board.occupied(x, y) || board.occupied(x + 1, y - 1))
                && board.occupied(x - 2, y + 3) == board.occupied(x - 2, y + 4)
            {
                if x >= 1 && (heights[x as usize] > y + 2 || heights[(x - 1) as usize] > y + 2) {
                    continue;
                }
                if x >= 2 && !board.occupied(x - 2, y + 3) && heights[(x - 2) as usize] > y + 2 {
                    continue;
                }
                3
            } else {
                continue;
            };
            let filled = [(x, y), (x + 2, y), (x + 2, y + 2), (x, y + 2)]
                .into_iter()
                .filter(|&(x, y)| board.occupied(x, y))
                .count();
            if filled < 3 {
                continue;
            }
            let lines = i32::from(board.rows[y as usize].count_ones() == 9)
                + i32::from(board.rows[y as usize + 1].count_ones() == 8)
                + i32::from(board.rows[y as usize + 2].count_ones() == 9);
            if lines != 0 && best.is_none_or(|hole| hole.lines < lines) {
                best = Some(TSpinHole {
                    lines,
                    x,
                    y,
                    rotation,
                });
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> Board {
        Board {
            rows: [0; 24],
            back_to_back: false,
            hold: None,
            bag: 0x7f,
            ren: 0,
        }
    }

    #[test]
    fn center_danger_matches_reference_across_board_rows() {
        let mut board = board();
        assert_eq!(center_height_excess(&board), 0);
        for y in 0..board.rows.len() {
            for x in 0..10 {
                board.rows[y] = 1 << x;
                let expected = (max_height(&board, 0b0001111000) - 15).max(0);
                assert_eq!(center_height_excess(&board), expected, "({x}, {y})");
            }
            board.rows[y] = 0;
        }
    }

    #[test]
    fn empty_board_golden_includes_source_bump_offset_and_wall_transitions() {
        assert_eq!(
            evaluate_board(&board(), &Weights::default()),
            Value {
                value: [-230, 0, 0, 0],
                spike: 0
            }
        );
    }

    #[test]
    fn two_line_well_goldens_keep_original_column_top_convention() {
        let mut b = board();
        b.rows[..2].fill(0x1ff);
        assert_eq!(
            evaluate_board(&b, &Weights::default()).value,
            [-220, 340, 0, 0]
        );
        b.rows[..2].fill(0x3ff ^ (1 << 4));
        assert_eq!(
            evaluate_board(&b, &Weights::default()).value,
            [-220, 260, 0, 0]
        );
    }

    #[test]
    fn suspended_cell_cavity_and_overhang_goldens() {
        let mut b = board();
        b.rows[1] = 1 << 4;
        assert_eq!(
            evaluate_board(&b, &Weights::default()).value,
            [-630, 0, 0, 0]
        );
        b.rows[1] = 0;
        b.rows[2] = 1 << 4;
        assert_eq!(
            evaluate_board(&b, &Weights::default()).value,
            [-802, 0, 0, 0]
        );
        b.rows[0] = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6);
        let (bits, _) = columns(&b);
        assert_eq!(cavities_and_overhangs(&bits, &bits, 2), (2, 0));
    }

    #[test]
    fn geometric_cavities_use_column_tops_and_physical_occupancy() {
        let mut b = board();
        b.rows[0] = 1 << 6;
        b.rows[1] = 1 << 6;
        b.rows[3] = 1 << 4;
        let (bits, heights) = columns(&b);
        let thresholds = geometric_tops(&bits, &heights);
        assert_eq!(cavities_and_overhangs(&bits, &bits, 3), (1, 3));
        assert_eq!(cavities_and_overhangs(&bits, &thresholds, 3), (0, 3));

        b.rows[0] |= 1 << 5;
        let (bits, heights) = columns(&b);
        let thresholds = geometric_tops(&bits, &heights);
        assert_eq!(cavities_and_overhangs(&bits, &thresholds, 3), (0, 3));

        b.rows[0] |= 1 << 3;
        let (bits, heights) = columns(&b);
        let thresholds = geometric_tops(&bits, &heights);
        assert_eq!(cavities_and_overhangs(&bits, &thresholds, 3), (1, 2));
    }

    #[test]
    fn column_masks_match_row_wise_cavity_classification() {
        let mut state = 0x5746_8b39_u32;
        for height in 0..24 {
            for density in [1, 2, 3] {
                let mut board = Board::default();
                for row in &mut board.rows[..=height] {
                    *row = 1023;
                    for _ in 0..density {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        *row &= (state >> 16) as u16;
                    }
                }
                let (columns, _) = columns(&board);
                for limit in [-1, height as i32 / 2, height as i32] {
                    let mut expected = (0, 0);
                    for y in 0..limit {
                        for x in 0..10 {
                            if y >= columns[x] || board.occupied(x as i32, y) {
                                continue;
                            }
                            if (x >= 2 && columns[x - 1] < y && columns[x - 2] <= y)
                                || (x < 8 && columns[x + 1] < y && columns[x + 2] <= y)
                            {
                                expected.1 += 1;
                            } else {
                                expected.0 += 1;
                            }
                        }
                    }
                    assert_eq!(cavities_and_overhangs(&columns, &columns, limit), expected);
                }
            }
        }
    }

    #[test]
    fn t_double_slot_is_valued_by_locking_a_clone() {
        let mut b = board();
        b.rows[0] = 0x3ff ^ (1 << 4);
        b.rows[1] = 0x3ff ^ (0b111 << 3);
        b.rows[2] = 1 << 3;
        b.rows[3] = 1;
        let original = b;
        let (_, heights) = columns(&b);
        assert_eq!(
            scan_tspin(&b, &heights, 3),
            Some(TSpinHole {
                lines: 2,
                x: 3,
                y: 0,
                rotation: 2
            })
        );
        let w = Weights {
            t_hole: 200,
            ..Weights::zeroed()
        };
        assert_eq!(evaluate_board(&b, &w).value, [0, 400, 0, 0]);
        assert_eq!(b, original);
        assert_eq!(
            evaluate_board(
                &b,
                &Weights {
                    no_tspin: true,
                    ..w
                }
            )
            .value,
            [0; 4]
        );
    }

    #[test]
    fn t_triple_slot_geometry_golden() {
        let mut b = board();
        b.rows[0] = 0x3ff ^ (1 << 1);
        b.rows[1] = 0x3ff ^ (0b11 << 1);
        b.rows[2] = 0x3ff ^ (1 << 1);
        b.rows[4] = 1 << 1;
        b.rows[5] = 1;
        let (_, heights) = columns(&b);
        assert_eq!(
            scan_tst(&b, &heights, 5),
            Some(TSpinHole {
                lines: 3,
                x: 0,
                y: 0,
                rotation: 1
            })
        );
    }

    #[test]
    fn move_golden_keeps_line_delay_and_offset_combo_scoring() {
        let b = board();
        let piece = Piece::new(PieceKind::I, 3, 0, 1);
        let lock = LockResult {
            placement: 4,
            death: false,
            back_to_back: true,
            perfect_clear: false,
            ren: 4,
        };
        // (10 + 40) * -6 + 340 + 100 * combo_attack(3)^2 + 800.
        let r = evaluate_move(10, Some(piece), &b, &lock, true, &Weights::default());
        assert_eq!(r.evaluation, 940);
        assert_eq!(r.attack, 7);
        let pc = LockResult {
            perfect_clear: true,
            ..lock
        };
        assert_eq!(
            evaluate_move(10, Some(piece), &b, &pc, true, &Weights::default()),
            Reward {
                evaluation: 4080,
                attack: 10
            }
        );
    }

    #[test]
    fn hold_only_skips_move_cost_but_preserves_hold_t_reward() {
        let mut b = board();
        b.hold = Some(PieceKind::T);
        let lock = LockResult {
            placement: 0,
            death: false,
            back_to_back: false,
            perfect_clear: false,
            ren: 0,
        };
        assert_eq!(
            evaluate_move(999, None, &b, &lock, true, &Weights::default()),
            Reward {
                evaluation: 10,
                attack: -1
            }
        );
    }

    #[test]
    fn danger_uses_zero_based_center_height_and_nonclear_spins_are_continuous() {
        let mut b = board();
        b.rows[18] = 1 << 3;
        let piece = Piece::new(PieceKind::I, 3, 0, 1);
        let lock = LockResult {
            placement: 0,
            death: false,
            back_to_back: false,
            perfect_clear: false,
            ren: 0,
        };
        let w = Weights {
            move_time: -6,
            danger: -20,
            ..Weights::zeroed()
        };
        assert_eq!(
            evaluate_move(5, Some(piece), &b, &lock, false, &w).evaluation,
            -330
        );

        b.rows[18] = 0;
        b.hold = Some(PieceKind::T);
        let mini = LockResult {
            placement: 8,
            ..lock
        };
        let t_piece =
            Piece::new(PieceKind::T, piece.x, piece.y, piece.rotation()).with_tspin(piece.tspin());
        assert_eq!(
            evaluate_move(10, Some(t_piece), &b, &mini, false, &Weights::default()).evaluation,
            -360
        );
    }

    #[test]
    fn downstack_uses_original_floor_log2_including_zero() {
        let mut b = board();
        let w = Weights {
            downstack: 50,
            ..Weights::zeroed()
        };
        assert_eq!(evaluate_board(&b, &w).value, [0, -50, 0, 0]);
        b.ren = 1;
        assert_eq!(evaluate_board(&b, &w).value, [0; 4]);
        b.ren = 8;
        assert_eq!(evaluate_board(&b, &w).value, [0, 150, 0, 0]);
    }

    #[test]
    fn value_backup_retains_independent_best_spike_and_strict_score_ties() {
        let a = Value {
            value: [10, 20, 30, 40],
            spike: 2,
        };
        let b = Value {
            value: [25; 4],
            spike: 7,
        };
        assert_eq!(
            a.improve(b),
            Value {
                value: a.value,
                spike: 7
            }
        );
        assert_eq!((a / 2).spike, 0);
        assert_eq!((a * 2).spike, 0);
        assert_eq!(
            a.with_reward(Reward {
                evaluation: -10,
                attack: -1
            }),
            Value {
                value: [10, 20, 20, 40],
                spike: 0
            }
        );
        assert_eq!(
            a.with_reward(Reward {
                evaluation: 1,
                attack: 0
            })
            .spike,
            2
        );
        assert_eq!(floor_log2(0), -1);
        assert_eq!(floor_log2(8), 3);
    }
}
