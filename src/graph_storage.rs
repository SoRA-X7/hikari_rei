//! Exact board deduplication for graph registration.
//!
//! The probing array stores only a 32-bit fingerprint and a dense record index.
//! Boards are stored once, densely, and are compared in full on a fingerprint
//! match. Unlike a map keyed by `u64`, hash collisions cannot merge graph nodes.

use crate::board::Board;

const INITIAL_BUCKETS: usize = 16;

#[derive(Clone, Copy)]
struct Record {
    board: Board,
    node: u32,
}

/// Append-only within a graph layer; clearing retains the allocated buffers.
///
/// A bucket's low word is its record index plus one (zero denotes an empty
/// bucket), and its high word is the upper half of the board hash. Keeping the
/// probing array separate avoids loading a full board for unrelated probes and
/// moving board-sized buckets when the probing array grows.
#[derive(Default)]
pub(crate) struct BoardLookup {
    buckets: Vec<u64>,
    records: Vec<Record>,
}

/// Hash every field participating in `Board::eq`, including all 24 rows.
///
/// This deterministic, non-cryptographic hash is intended for internally
/// generated search positions. Four rows are combined at a time to avoid the
/// per-field hashing of `Board`'s derived `Hash` implementation.
#[inline]
pub(crate) fn board_hash(board: &Board) -> u64 {
    const MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;
    let mut hash = 0x243f_6a88_85a3_08d3_u64;
    for rows in board.rows.as_chunks::<4>().0 {
        let word = u64::from(rows[0])
            | (u64::from(rows[1]) << 16)
            | (u64::from(rows[2]) << 32)
            | (u64::from(rows[3]) << 48);
        hash = (hash.rotate_left(5) ^ word).wrapping_mul(MULTIPLIER);
    }
    let metadata = u64::from(board.ren as u32)
        | (u64::from(board.bag) << 32)
        | (u64::from(board.back_to_back) << 40)
        | (u64::from(board.hold.map_or(7, |kind| kind as u8)) << 41);
    hash = (hash.rotate_left(5) ^ metadata).wrapping_mul(MULTIPLIER);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

impl BoardLookup {
    pub(crate) fn clear(&mut self) {
        self.buckets.fill(0);
        self.records.clear();
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn get(&self, board: &Board) -> Option<&u32> {
        self.find(board, board_hash(board))
            .ok()
            .map(|record| &self.records[record].node)
    }

    #[inline]
    pub(crate) fn get_hashed(&self, board: &Board, hash: u64) -> Option<u32> {
        self.find(board, hash)
            .ok()
            .map(|record| self.records[record].node)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn contains_key(&self, board: &Board) -> bool {
        self.get(board).is_some()
    }

    #[cfg(test)]
    pub(crate) fn keys(&self) -> impl Iterator<Item = &Board> {
        self.records.iter().map(|record| &record.board)
    }

    #[inline]
    pub(crate) fn insert(&mut self, board: Board, node: u32) -> Option<u32> {
        self.insert_hashed(board, board_hash(&board), node)
    }

    /// Insert or replace a value using a hash returned by [`board_hash`].
    #[inline]
    pub(crate) fn insert_hashed(&mut self, board: Board, hash: u64, node: u32) -> Option<u32> {
        self.reserve_for_insert();
        match self.find(&board, hash) {
            Ok(record) => Some(std::mem::replace(&mut self.records[record].node, node)),
            Err(bucket) => {
                self.insert_at(bucket, board, hash, node);
                None
            }
        }
    }

    /// Resolve an existing node or register a new one with a single probe walk.
    ///
    /// `hash` must come from [`board_hash`]; growth recalculates hashes from the
    /// exact stored boards instead of adding another eight bytes to each key.
    /// The boolean reports whether a new board was inserted.
    #[inline]
    pub(crate) fn insert_if_absent_hashed(
        &mut self,
        board: Board,
        hash: u64,
        new_node: u32,
    ) -> (u32, bool) {
        self.reserve_for_insert();
        match self.find(&board, hash) {
            Ok(record) => (self.records[record].node, false),
            Err(bucket) => {
                self.insert_at(bucket, board, hash, new_node);
                (new_node, true)
            }
        }
    }

    /// Return either the exact record or the first vacant bucket.
    #[inline]
    fn find(&self, board: &Board, hash: u64) -> Result<usize, usize> {
        if self.buckets.is_empty() {
            return Err(0);
        }
        let mask = self.buckets.len() - 1;
        let mut bucket = hash as usize & mask;
        loop {
            let packed = self.buckets[bucket];
            if packed == 0 {
                return Err(bucket);
            }
            if packed >> 32 == hash >> 32 {
                let record = (packed as u32 - 1) as usize;
                if self.records[record].board == *board {
                    return Ok(record);
                }
            }
            bucket = (bucket + 1) & mask;
        }
    }

    #[inline]
    fn insert_at(&mut self, bucket: usize, board: Board, hash: u64, node: u32) {
        let index = u32::try_from(self.records.len())
            .expect("board table exceeds the u32 graph index limit");
        let encoded = index.checked_add(1).expect("board table index overflow");
        self.records.push(Record { board, node });
        self.buckets[bucket] = (hash & 0xffff_ffff_0000_0000) | u64::from(encoded);
    }

    #[inline]
    fn reserve_for_insert(&mut self) {
        // Linear probing is intentionally limited to 75% load. At higher loads
        // a missing board would inspect long runs of otherwise compact buckets.
        if self.records.len() >= self.buckets.len() / 4 * 3 {
            let capacity = if self.buckets.is_empty() {
                INITIAL_BUCKETS
            } else {
                self.buckets
                    .len()
                    .checked_mul(2)
                    .expect("board table capacity overflow")
            };
            self.resize_buckets(capacity);
        }
    }

    fn resize_buckets(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        self.buckets.resize(capacity, 0);
        self.buckets.fill(0);
        self.records
            .reserve((capacity / 4 * 3).saturating_sub(self.records.len()));
        let mask = capacity - 1;
        for (index, record) in self.records.iter().enumerate() {
            let hash = board_hash(&record.board);
            let mut bucket = hash as usize & mask;
            while self.buckets[bucket] != 0 {
                bucket = (bucket + 1) & mask;
            }
            self.buckets[bucket] = (hash & 0xffff_ffff_0000_0000) | (index as u64 + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PieceKind;
    use std::collections::HashMap;

    fn board(index: u32) -> Board {
        let mut board = Board {
            ren: index as i32,
            bag: (index % 128) as u8,
            hold: PieceKind::from_index(index as usize % 8),
            back_to_back: index & 1 != 0,
            ..Board::default()
        };
        for (row, value) in board.rows.iter_mut().enumerate() {
            *value = index.rotate_left(row as u32) as u16;
        }
        board
    }

    #[test]
    fn equal_hashes_still_require_full_board_equality() {
        let mut lookup = BoardLookup::default();
        lookup.resize_buckets(64);
        // Force the entire hash to collide, not merely the 32-bit fingerprint.
        // Capacity stays fixed so these deliberately artificial hashes are not
        // replaced by the production board_hash during growth.
        let hash = 0x1234_5678_ffff_ffff;
        for index in 0..32 {
            assert_eq!(
                lookup.insert_if_absent_hashed(board(index), hash, index),
                (index, true)
            );
        }
        for index in 0..32 {
            assert_eq!(lookup.get_hashed(&board(index), hash), Some(index));
            assert_eq!(
                lookup.insert_if_absent_hashed(board(index), hash, 999),
                (index, false)
            );
        }
        assert_eq!(lookup.get_hashed(&board(32), hash), None);
        assert_eq!(lookup.records.len(), 32);
    }

    #[test]
    fn growth_and_replacement_match_exact_key_hash_map() {
        let mut lookup = BoardLookup::default();
        let mut reference = HashMap::new();
        for index in 0..10_000 {
            let key = board(index);
            assert_eq!(lookup.insert(key, index), reference.insert(key, index));
            if index.is_power_of_two() {
                for (key, value) in &reference {
                    assert_eq!(lookup.get(key), Some(value));
                }
            }
        }
        for index in (0..10_000).rev() {
            let key = board(index);
            assert_eq!(
                lookup.insert_hashed(key, board_hash(&key), index + 1),
                reference.insert(key, index + 1)
            );
        }
        assert_eq!(lookup.keys().count(), reference.len());
        for (key, value) in &reference {
            assert!(lookup.contains_key(key));
            assert_eq!(lookup.get_hashed(key, board_hash(key)), Some(*value));
        }
    }

    #[test]
    fn clear_removes_old_keys_and_reuses_allocations() {
        let mut lookup = BoardLookup::default();
        for index in 0..1000 {
            lookup.insert(board(index), index);
        }
        let bucket_capacity = lookup.buckets.capacity();
        let record_capacity = lookup.records.capacity();
        lookup.clear();
        assert_eq!(lookup.keys().count(), 0);
        for index in 0..1000 {
            assert_eq!(lookup.get(&board(index)), None);
            assert_eq!(lookup.insert(board(index), index + 7), None);
        }
        assert_eq!(lookup.buckets.capacity(), bucket_capacity);
        assert_eq!(lookup.records.capacity(), record_capacity);
        assert_eq!(lookup.get(&board(999)), Some(&1006));
    }

    #[test]
    fn upper_rows_and_all_metadata_remain_distinct() {
        let mut keys = vec![Board::default()];
        for row in 0..24 {
            let mut key = Board::default();
            key.rows[row] = 1;
            keys.push(key);
        }
        for kind in PieceKind::ALL {
            keys.push(Board {
                hold: Some(kind),
                ..Board::default()
            });
        }
        for bag in [1, 0x7f, 0x80, 0xff] {
            keys.push(Board {
                bag,
                ..Board::default()
            });
        }
        for ren in [1, -1, i32::MIN, i32::MAX] {
            keys.push(Board {
                ren,
                ..Board::default()
            });
        }
        keys.push(Board {
            back_to_back: true,
            ..Board::default()
        });
        let mut lookup = BoardLookup::default();
        for (index, key) in keys.iter().copied().enumerate() {
            assert_eq!(lookup.insert(key, index as u32), None);
            if index != 0 {
                assert_ne!(board_hash(&keys[0]), board_hash(&key));
            }
        }
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(lookup.get(key), Some(&(index as u32)));
        }
    }
}
