//! Hikari's 24-row `SimpleBoard`, piece geometry, and SRS rules.
//!
//! Ported from Hikari commit 90aef085a53d6048cbd19de7f4c9c53a6f3c8960.
//! The combo and B2B values below intentionally follow `LockSelf`, including
//! its distinction from the Unity game's `Board.Lock`.
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

pub const WIDTH: usize = 10;
pub const HEIGHT: usize = 24;
pub const FILLED: u16 = (1 << WIDTH) - 1;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum PieceKind {
    #[default]
    I,
    O,
    T,
    J,
    L,
    S,
    Z,
}
impl PieceKind {
    pub const ALL: [Self; 7] = [
        Self::I,
        Self::O,
        Self::T,
        Self::J,
        Self::L,
        Self::S,
        Self::Z,
    ];
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }
    pub fn letter(self) -> char {
        ['I', 'O', 'T', 'J', 'L', 'S', 'Z'][self as usize]
    }
}
impl fmt::Display for PieceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.letter())
    }
}
impl FromStr for PieceKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|kind| s.eq_ignore_ascii_case(&kind.letter().to_string()))
            .ok_or_else(|| format!("invalid piece kind: {s}"))
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum TSpin {
    #[default]
    None,
    Mini,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "PieceWire", into = "PieceWire")]
pub struct Piece {
    pub kind: PieceKind,
    pub x: i8,
    pub y: i8,
    // Low two bits: rotation; next two bits: T-spin classification.
    state: u8,
}
impl Piece {
    pub const fn new(kind: PieceKind, x: i8, y: i8, rotation: u8) -> Self {
        Self {
            kind,
            x,
            y,
            state: rotation & 3,
        }
    }
    pub const fn spawn(kind: PieceKind) -> Self {
        Self::new(
            kind,
            3,
            if matches!(kind, PieceKind::I) { 17 } else { 18 },
            0,
        )
    }
    pub fn shifted(self, x: i8, y: i8) -> Self {
        Self {
            x: self.x + x,
            y: self.y + y,
            ..self
        }
    }
    pub fn with_rotation(self, rotation: u8) -> Self {
        Self {
            state: (self.state & !3) | (rotation & 3),
            ..self
        }
    }
    pub fn with_tspin(self, tspin: TSpin) -> Self {
        Self {
            state: (self.state & 3) | ((tspin as u8) << 2),
            ..self
        }
    }
    #[inline]
    pub const fn rotation(self) -> u8 {
        self.state & 3
    }
    #[inline]
    pub const fn tspin(self) -> TSpin {
        match self.state >> 2 {
            0 => TSpin::None,
            1 => TSpin::Mini,
            _ => TSpin::Full,
        }
    }
    pub fn shape(self) -> [u16; 4] {
        SHAPES[self.kind as usize][self.rotation() as usize]
    }
    pub fn cells(self) -> [(i32, i32); 4] {
        let shape = self.shape();
        let mut result = [(0, 0); 4];
        let mut next = 0;
        for (y, row) in shape.into_iter().enumerate() {
            for x in 0..4 {
                if row & (1 << x) != 0 {
                    result[next] = (self.x as i32 + x, self.y as i32 + y as i32);
                    next += 1;
                }
            }
        }
        result
    }
}
impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if self.kind == PieceKind::T {
            write!(f, "_{:?}", self.tspin())?;
        }
        write!(f, " ({},{}) {}", self.x, self.y, self.rotation())
    }
}

// Preserve the public serialized fields while packing their in-memory representation.
#[derive(Serialize, Deserialize)]
struct PieceWire {
    kind: PieceKind,
    x: i8,
    y: i8,
    rotation: u8,
    tspin: TSpin,
}
impl From<PieceWire> for Piece {
    fn from(p: PieceWire) -> Self {
        Self::new(p.kind, p.x, p.y, p.rotation).with_tspin(p.tspin)
    }
}
impl From<Piece> for PieceWire {
    fn from(p: Piece) -> Self {
        Self {
            kind: p.kind,
            x: p.x,
            y: p.y,
            rotation: p.rotation(),
            tspin: p.tspin(),
        }
    }
}

/// Rows of each rotation from bottom to top, exactly `Piece.NativeShapes`.
pub const SHAPES: [[[u16; 4]; 4]; 7] = [
    [[0, 0, 15, 0], [4, 4, 4, 4], [0, 15, 0, 0], [2, 2, 2, 2]],
    [[0, 6, 6, 0]; 4],
    [[0, 7, 2, 0], [2, 6, 2, 0], [2, 7, 0, 0], [2, 3, 2, 0]],
    [[0, 7, 1, 0], [2, 2, 6, 0], [4, 7, 0, 0], [3, 2, 2, 0]],
    [[0, 7, 4, 0], [6, 2, 2, 0], [1, 7, 0, 0], [2, 2, 3, 0]],
    [[0, 3, 6, 0], [4, 6, 2, 0], [3, 6, 0, 0], [2, 3, 1, 0]],
    [[0, 6, 3, 0], [2, 6, 4, 0], [6, 3, 0, 0], [1, 3, 2, 0]],
];

// Match the native ordering of four adjacent board rows in a u64 load.
const PACKED_SHAPES: [[u64; 4]; 7] = {
    let mut packed = [[0; 4]; 7];
    let mut kind = 0;
    while kind < 7 {
        let mut rotation = 0;
        while rotation < 4 {
            let mut row = 0;
            while row < 4 {
                let lane = if cfg!(target_endian = "little") {
                    row
                } else {
                    3 - row
                };
                packed[kind][rotation] |= (SHAPES[kind][rotation][row] as u64) << (16 * lane);
                row += 1;
            }
            rotation += 1;
        }
        kind += 1;
    }
    packed
};

// Four 16-bit rows, with the bottom row in the low lane. The packed lock
// path uses this table only on little-endian targets; other targets use
// the original row loop.
const PACKED_LOCK_SHAPES: [[u64; 4]; 7] = {
    let mut result = [[0; 4]; 7];
    let mut kind = 0;
    while kind < 7 {
        let mut rotation = 0;
        while rotation < 4 {
            let mut row = 0;
            while row < 4 {
                result[kind][rotation] |= (SHAPES[kind][rotation][row] as u64) << (row * 16);
                row += 1;
            }
            rotation += 1;
        }
        kind += 1;
    }
    result
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Board {
    pub rows: [u16; HEIGHT],
    pub back_to_back: bool,
    pub hold: Option<PieceKind>,
    /// Zero means the next `Take` starts a new full seven-bag.
    pub bag: u8,
    pub ren: i32,
}
impl Board {
    pub fn spawn(&self, kind: PieceKind) -> Option<Piece> {
        let mut piece = Piece::spawn(kind);
        if self.collides(piece) {
            piece = piece.shifted(0, 1);
        }
        (!self.collides(piece)).then_some(piece)
    }
    #[inline]
    pub fn occupied(&self, x: i32, y: i32) -> bool {
        if !(0..WIDTH as i32).contains(&x) || y < 0 {
            return true;
        }
        self.rows
            .get(y as usize)
            .is_some_and(|row| row & (1 << x) != 0)
    }
    #[inline(always)]
    pub fn collides(&self, piece: Piece) -> bool {
        // This early return is present in SimpleBoard.Collides.
        if piece.y > HEIGHT as i8 - 3 {
            return false;
        }
        if (0..=HEIGHT as i8 - 4).contains(&piece.y) && (-3..=9).contains(&piece.x) {
            // Four rows fit in this range. Their start may only be u16-aligned.
            // SAFETY: y is at most HEIGHT - 4, so all eight bytes are inside rows.
            let field = unsafe {
                self.rows
                    .as_ptr()
                    .add(piece.y as usize)
                    .cast::<u64>()
                    .read_unaligned()
            };
            // Keep each row within its own 16-bit lane, with three wall bits
            // on either side of the ten playable columns. The x range keeps
            // all four shape columns inside a lane even for invalid moves.
            let field = ((field & 0x03ff_03ff_03ff_03ff) << 3) | 0xe007_e007_e007_e007;
            let shape =
                PACKED_SHAPES[piece.kind as usize][piece.rotation() as usize] << (piece.x + 3);
            return field & shape != 0;
        }
        let shape = piece.shape();
        for (dy, row) in shape.into_iter().enumerate() {
            if row == 0 {
                continue;
            }
            let y = piece.y as i32 + dy as i32;
            if y < 0 {
                return true;
            }
            // Three sentinel columns on each side mirror Hikari's bitboard.
            let shifted = if piece.x >= 0 {
                (row as u32) << piece.x
            } else {
                (row as u32) >> (-piece.x)
            };
            // A right shift can hide blocks left of the board.
            if piece.x < 0 && row & ((1u16 << (-piece.x).min(15)) - 1) != 0 {
                return true;
            }
            if shifted & !(FILLED as u32) != 0 {
                return true;
            }
            if self
                .rows
                .get(y as usize)
                .is_some_and(|field| *field as u32 & shifted != 0)
            {
                return true;
            }
        }
        false
    }
    #[inline]
    pub fn sonic_drop(&self, piece: Piece) -> Option<Piece> {
        if self.collides(piece.shifted(0, -1)) {
            return Some(piece);
        }
        for y in (-3..=(piece.y as i32 - 2)).rev() {
            if self.collides(Piece {
                y: y as i8,
                ..piece
            }) {
                // The original creates a new Piece, clearing the spin status
                // only if the drop actually travels down at least one row.
                return Some(Piece::new(
                    piece.kind,
                    piece.x,
                    (y + 1) as i8,
                    piece.rotation(),
                ));
            }
        }
        None
    }
    pub fn max_height(&self) -> usize {
        self.rows
            .iter()
            .rposition(|row| *row != 0)
            .map_or(0, |row| row + 1)
    }
    /// Bit columns and highest occupied row (zero also represents empty).
    pub fn columns(&self) -> ([u32; WIDTH], [u8; WIDTH]) {
        let mut columns = [0; WIDTH];
        let mut heights = [0; WIDTH];
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                if self.rows[y] & (1 << x) != 0 {
                    columns[x] |= 1 << y;
                    heights[x] = y as u8;
                }
            }
        }
        (columns, heights)
    }
    pub fn check_tspin(&self, piece: Piece, kick: u8) -> TSpin {
        if piece.kind != PieceKind::T {
            return TSpin::None;
        }
        const CORNERS: [[(i32, i32); 2]; 4] = [
            [(0, 0), (2, 0)],
            [(0, 2), (0, 0)],
            [(2, 2), (0, 2)],
            [(2, 0), (2, 2)],
        ];
        let count = |r: usize| {
            CORNERS[r]
                .into_iter()
                .filter(|(x, y)| self.occupied(*x + piece.x as i32, *y + piece.y as i32))
                .count()
        };
        let front = count(piece.rotation() as usize);
        let back = count((piece.rotation() as usize + 2) & 3);
        if front + back < 3 {
            TSpin::None
        } else if kick == 4 || back == 2 {
            TSpin::Full
        } else {
            TSpin::Mini
        }
    }
    pub fn try_rotate(&self, piece: Piece, clockwise: bool) -> Option<(Piece, u8)> {
        if piece.kind == PieceKind::O {
            return Some((piece, 0));
        }
        let rotation = (piece.rotation() + if clockwise { 1 } else { 3 }) & 3;
        let direct = Piece::new(piece.kind, piece.x, piece.y, rotation);
        if !self.collides(direct) {
            return Some((direct, 0));
        }
        let table = if piece.kind == PieceKind::I {
            &KICKS_I
        } else {
            &KICKS
        };
        let index = (piece.rotation() as usize * 2 + if clockwise { 0 } else { 7 }) & 7;
        for (kick, &(x, y)) in table[index].iter().enumerate() {
            let result = piece.shifted(x, y).with_rotation(rotation);
            if !self.collides(result) {
                return Some((result, kick as u8 + 1));
            }
        }
        None
    }
    pub fn rotate(&self, piece: Piece, clockwise: bool) -> Option<Piece> {
        self.try_rotate(piece, clockwise)
            .map(|(piece, kick)| piece.with_tspin(self.check_tspin(piece, kick)))
    }
    pub fn take_from_bag(&mut self, kind: PieceKind) -> bool {
        if self.bag == 0 {
            self.bag = 0x7f;
        }
        let bit = 1 << kind as u8;
        if self.bag & bit == 0 {
            return false;
        }
        self.bag &= !bit;
        true
    }
    pub fn bag_candidates(&self) -> u8 {
        if self.bag == 0 { 0x7f } else { self.bag }
    }
    pub fn lock(&mut self, piece: Piece) -> LockResult {
        let mut cleared = [0usize; 4];
        let mut count = 0;
        let mut death = false;
        if cfg!(target_endian = "little")
            && (0..=HEIGHT as i8 - 4).contains(&piece.y)
            && (-3..=12).contains(&piece.x)
        {
            let shape = PACKED_LOCK_SHAPES[piece.kind as usize][piece.rotation() as usize];
            let shifted = if piece.x >= 0 {
                shape << piece.x
            } else {
                // Remove bits spilling from each lane into the previous one.
                (shape >> (-piece.x)) & 0x03ff_03ff_03ff_03ff
            };
            let y = piece.y as usize;
            // SAFETY: y <= 20 leaves four initialized u16 rows inside the
            // 24-row array; unaligned operations do not require u64 alignment.
            // The pointer does not outlive this exclusive borrow of the board.
            let placed = unsafe {
                let rows = self.rows.as_mut_ptr().add(y).cast::<u64>();
                let placed = u64::from_le(rows.read_unaligned()) | shifted;
                rows.write_unaligned(placed.to_le());
                placed
            };
            death = piece.y as i32 + (shape.trailing_zeros() >> 4) as i32 >= 20;
            for dy in 0..4 {
                // A pre-filled row is checked only if the original shape uses it,
                // even when an invalid x shift discards all its cells.
                if (shape >> (dy * 16)) as u16 != 0 && (placed >> (dy * 16)) as u16 == FILLED {
                    cleared[count] = y + dy;
                    count += 1;
                }
            }
        } else {
            let mut bottommost = true;
            for (dy, row) in piece.shape().into_iter().enumerate() {
                if row == 0 {
                    continue;
                }
                let y = piece.y as i32 + dy as i32;
                if bottommost {
                    death = y >= 20;
                    bottommost = false;
                }
                if y >= HEIGHT as i32 {
                    break;
                }
                // Legal placements never have an occupied cell below zero.
                if y < 0 {
                    death = true;
                    continue;
                }
                let shifted = (row as u32).rotate_left(piece.x.rem_euclid(32) as u32) as u16;
                self.rows[y as usize] |= shifted;
                if self.rows[y as usize] == FILLED {
                    cleared[count] = y as usize;
                    count += 1;
                }
            }
        }
        for (index, &row) in cleared[..count].iter().enumerate() {
            for target in row - index..HEIGHT - 1 {
                self.rows[target] = self.rows[target + 1];
            }
            self.rows[HEIGHT - 1 - index] = 0;
        }
        let perfect_clear = self.rows.iter().all(|row| *row == 0);
        let placement = (piece.tspin() as u8) * 8 + count as u8;
        if count != 0 {
            self.ren += 1;
            self.back_to_back = placement >= 4;
        } else {
            self.ren = 0;
        }
        death |= (self.rows[20] | self.rows[21]) & 0b0001111000 != 0;
        LockResult {
            placement,
            death,
            back_to_back: self.back_to_back,
            perfect_clear,
            ren: if count != 0 { self.ren + 1 } else { 0 },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockResult {
    pub placement: u8,
    pub death: bool,
    pub back_to_back: bool,
    pub perfect_clear: bool,
    pub ren: i32,
}
impl LockResult {
    pub fn lines(self) -> u8 {
        self.placement & 7
    }
    pub fn is_continuous(self) -> bool {
        self.placement >= 4
    }
    pub fn is_full_tspin_clear(self) -> bool {
        self.placement & 16 != 0 && self.lines() != 0
    }
    pub fn attack(self) -> i32 {
        if self.lines() == 0 {
            return -1;
        }
        if self.perfect_clear {
            return 10;
        }
        let garbage = match self.placement {
            2 | 10 => 1,
            3 | 17 => 2,
            4 | 18 => 4,
            19 => 6,
            _ => 0,
        };
        const REN_ATTACKS: [i32; 12] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5];
        garbage + i32::from(self.back_to_back) + REN_ATTACKS[self.ren.clamp(0, 11) as usize]
    }
}

const KICKS: [[(i8, i8); 4]; 8] = [
    [(-1, 0), (-1, 1), (0, -2), (-1, -2)],
    [(1, 0), (1, -1), (0, 2), (1, 2)],
    [(1, 0), (1, -1), (0, 2), (1, 2)],
    [(-1, 0), (-1, 1), (0, -2), (-1, -2)],
    [(1, 0), (1, 1), (0, -2), (1, -2)],
    [(-1, 0), (-1, -1), (0, 2), (-1, 2)],
    [(-1, 0), (-1, -1), (0, 2), (-1, 2)],
    [(1, 0), (1, 1), (0, -2), (1, -2)],
];
const KICKS_I: [[(i8, i8); 4]; 8] = [
    [(-2, 0), (1, 0), (-2, -1), (1, 2)],
    [(2, 0), (-1, 0), (2, 1), (-1, -2)],
    [(-1, 0), (2, 0), (-1, 2), (2, -1)],
    [(1, 0), (-2, 0), (1, -2), (-2, 1)],
    [(2, 0), (-1, 0), (2, 1), (-1, -2)],
    [(-2, 0), (1, 0), (-2, -1), (1, 2)],
    [(1, 0), (-2, 0), (1, -2), (-2, 1)],
    [(-1, 0), (2, 0), (-1, 2), (2, -1)],
];

#[cfg(test)]
mod tests {
    use super::*;

    // Keep the original row-based implementation as a differential oracle.
    fn scalar_collides(board: &Board, piece: Piece) -> bool {
        if piece.y > HEIGHT as i8 - 3 {
            return false;
        }
        for (dy, row) in piece.shape().into_iter().enumerate() {
            if row == 0 {
                continue;
            }
            let y = piece.y as i32 + dy as i32;
            if y < 0 {
                return true;
            }
            let shifted = if piece.x >= 0 {
                (row as u32) << piece.x
            } else {
                (row as u32) >> (-piece.x)
            };
            if piece.x < 0 && row & ((1u16 << (-piece.x).min(15)) - 1) != 0 {
                return true;
            }
            if shifted & !(FILLED as u32) != 0 {
                return true;
            }
            if board
                .rows
                .get(y as usize)
                .is_some_and(|field| *field as u32 & shifted != 0)
            {
                return true;
            }
        }
        false
    }

    #[test]
    fn packed_collides_matches_scalar_on_all_piece_geometry() {
        let mut state = 0x8f3d_2ca7_b519_64e1_u64;
        let mut random_row = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u16
        };
        for board_index in 0..99 {
            let mut board = Board::default();
            if board_index == 1 {
                board.rows.fill(FILLED);
            } else if board_index == 2 {
                // Out-of-board bits must not spill into the neighboring row.
                board.rows.fill(!FILLED);
            } else if board_index > 2 {
                for row in &mut board.rows {
                    *row = random_row();
                    // Exercise dense and sparse boards, including unused bits.
                    for _ in 0..board_index % 5 {
                        *row &= random_row();
                    }
                }
            }
            for kind in PieceKind::ALL {
                for rotation in 0..4 {
                    for tspin in [TSpin::None, TSpin::Mini, TSpin::Full] {
                        for x in -4..=12 {
                            for y in -4..=25 {
                                let piece = Piece::new(kind, x, y, rotation).with_tspin(tspin);
                                assert_eq!(
                                    board.collides(piece),
                                    scalar_collides(&board, piece),
                                    "board {board_index}, piece {piece:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spawn_and_drop_match_hikari_coordinates() {
        let board = Board::default();
        for kind in PieceKind::ALL {
            let spawn = board.spawn(kind).unwrap();
            assert_eq!(spawn.y, if kind == PieceKind::I { 17 } else { 18 });
            let dropped = board.sonic_drop(spawn).unwrap();
            assert_eq!(dropped.y, if kind == PieceKind::I { -2 } else { -1 });
            assert!(!board.collides(dropped));
            assert!(board.collides(dropped.shifted(0, -1)));
        }
    }
    #[test]
    fn seven_bag_restore_and_reject_repeat_match_original() {
        let mut board = Board::default();
        for (i, kind) in PieceKind::ALL.into_iter().enumerate() {
            assert!(board.take_from_bag(kind));
            if i != 6 {
                assert!(!board.take_from_bag(kind));
            }
        }
        assert_eq!(board.bag, 0);
        assert_eq!(board.bag_candidates(), 0x7f);
        assert!(board.take_from_bag(PieceKind::I));
        assert_eq!(board.bag_candidates(), 0x7e);
    }
    #[test]
    fn lockself_combo_b2b_and_perfect_clear_contract() {
        let mut board = Board::default();
        board.rows[..4].fill(FILLED & !(1 << 4));
        board.rows[4] = 1;
        let result = board.lock(Piece::new(PieceKind::I, 2, 0, 1));
        assert_eq!((result.lines(), board.ren, result.ren), (4, 1, 2));
        assert!(result.back_to_back);
        assert_eq!(result.attack(), 6); // 4 quad + current B2B + GetRenAttack(2)
        assert_eq!(board.rows[0], 1);
        let mut board = Board::default();
        board.rows[..4].fill(FILLED & !(1 << 4));
        let result = board.lock(Piece::new(PieceKind::I, 2, 0, 1));
        assert!(result.perfect_clear);
        assert_eq!(result.attack(), 10);
    }
    #[test]
    fn drop_clears_spin_status_only_when_it_moves() {
        let board = Board::default();
        let grounded = Piece::new(PieceKind::T, 3, -1, 0).with_tspin(TSpin::Full);
        assert_eq!(board.sonic_drop(grounded), Some(grounded));
        assert_eq!(
            board.sonic_drop(grounded.shifted(0, 1)).unwrap().tspin(),
            TSpin::None
        );
    }
    #[test]
    fn center_spawn_obstruction_is_death_after_lock() {
        let mut board = Board::default();
        let result = board.lock(Piece::new(PieceKind::O, 3, 19, 0));
        assert!(result.death);
    }
    #[test]
    fn original_srs_floor_kicks_for_t_and_i() {
        let board = Board::default();
        assert_eq!(
            board.try_rotate(Piece::new(PieceKind::T, 3, -1, 0), true),
            Some((Piece::new(PieceKind::T, 2, 0, 1), 2))
        );
        assert_eq!(
            board.try_rotate(Piece::new(PieceKind::I, 3, -2, 0), true),
            Some((Piece::new(PieceKind::I, 4, 0, 1), 4))
        );
    }
    #[test]
    fn original_corner_orientation_and_fifth_test_tspin_upgrade() {
        let piece = Piece::new(PieceKind::T, 3, 0, 0);
        let mut board = Board::default();
        board.rows[0] = 1 << 3;
        board.rows[2] = (1 << 3) | (1 << 5);
        assert_eq!(board.check_tspin(piece, 0), TSpin::Full);
        board.rows[0] = (1 << 3) | (1 << 5);
        board.rows[2] = 1 << 3;
        assert_eq!(board.check_tspin(piece, 0), TSpin::Mini);
        assert_eq!(board.check_tspin(piece, 4), TSpin::Full);
    }
    #[test]
    fn original_spawn_retries_one_row_up_then_fails() {
        let mut board = Board::default();
        board.rows[19] = 1 << 4;
        assert_eq!(
            board.spawn(PieceKind::O),
            Some(Piece::new(PieceKind::O, 3, 19, 0))
        );
        board.rows[21] = 1 << 4;
        assert_eq!(board.spawn(PieceKind::O), None);
    }
}

#[cfg(test)]
impl Board {
    fn lock_reference(&mut self, piece: Piece) -> LockResult {
        let mut cleared = [0usize; 4];
        let mut count = 0;
        let mut bottommost = true;
        let mut death = false;
        for (dy, row) in piece.shape().into_iter().enumerate() {
            if row == 0 {
                continue;
            }
            let y = piece.y as i32 + dy as i32;
            if bottommost {
                death = y >= 20;
                bottommost = false;
            }
            if y >= HEIGHT as i32 {
                break;
            }
            // Legal placements never have an occupied cell below zero.
            if y < 0 {
                death = true;
                continue;
            }
            let shifted = (row as u32).rotate_left(piece.x.rem_euclid(32) as u32) as u16;
            self.rows[y as usize] |= shifted;
            if self.rows[y as usize] == FILLED {
                cleared[count] = y as usize;
                count += 1;
            }
        }
        for (index, &row) in cleared[..count].iter().enumerate() {
            for target in row - index..HEIGHT - 1 {
                self.rows[target] = self.rows[target + 1];
            }
            self.rows[HEIGHT - 1 - index] = 0;
        }
        let perfect_clear = self.rows.iter().all(|row| *row == 0);
        let placement = (piece.tspin() as u8) * 8 + count as u8;
        if count != 0 {
            self.ren += 1;
            self.back_to_back = placement >= 4;
        } else {
            self.ren = 0;
        }
        death |= (self.rows[20] | self.rows[21]) & 0b0001111000 != 0;
        LockResult {
            placement,
            death,
            back_to_back: self.back_to_back,
            perfect_clear,
            ren: if count != 0 { self.ren + 1 } else { 0 },
        }
    }
}

#[cfg(test)]
mod packed_lock_tests {
    use super::*;

    fn compare(board: Board, piece: Piece) {
        let mut expected = board;
        let mut actual = board;
        let reference = expected.lock_reference(piece);
        let packed = actual.lock(piece);
        assert_eq!(
            packed, reference,
            "result mismatch: {piece:?}, board={board:?}"
        );
        assert_eq!(
            actual, expected,
            "board mismatch: {piece:?}, board={board:?}"
        );
    }

    #[test]
    fn packed_lock_matches_reference_for_every_i8_position() {
        let board = Board::default();
        for kind in PieceKind::ALL {
            for rotation in 0..4 {
                for y in i8::MIN..=i8::MAX {
                    for x in i8::MIN..=i8::MAX {
                        compare(board, Piece::new(kind, x, y, rotation));
                    }
                }
            }
        }
    }

    #[test]
    fn packed_lock_matches_reference_random_and_prefilled_rows() {
        let mut seed = 0x71ab_09cd_67ef_2345u64;
        for sample in 0..96 {
            let mut board = Board::default();
            for row in &mut board.rows {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *row = match sample % 4 {
                    0 => (seed as u16) & FILLED,
                    1 => FILLED,
                    2 => (seed as u16) | FILLED,
                    _ => (seed as u16) & FILLED & !(1 << (sample % 10)),
                };
            }
            board.back_to_back = sample % 2 == 0;
            board.ren = sample % 16;
            board.hold = PieceKind::from_index(sample as usize % 8);
            board.bag = sample as u8;
            for kind in PieceKind::ALL {
                for rotation in 0..4 {
                    for y in -4..=25 {
                        for x in -6..=16 {
                            for spin in [TSpin::None, TSpin::Mini, TSpin::Full] {
                                compare(board, Piece::new(kind, x, y, rotation).with_tspin(spin));
                            }
                        }
                    }
                }
            }
        }
    }
}
