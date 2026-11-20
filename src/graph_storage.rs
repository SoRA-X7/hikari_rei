//! Exact board deduplication for graph registration.
//!
//! The probing arrays store a seven-bit fingerprint and dense record index.
//! Eight fingerprints are inspected in one word, keeping unrelated board data
//! out of the probe walk. Every fingerprint hit requires complete Board
//! equality: hash collisions cannot merge graph nodes.

use crate::board::Board;

const INITIAL_BUCKETS: usize = 16;
const GROUP_WIDTH: usize = 8;
const EMPTY: u8 = 0x80;
const HIGH_BYTES: u64 = 0x8080_8080_8080_8080;
const LOW_BYTES: u64 = 0x0101_0101_0101_0101;

/// Identify zero bytes. Borrow propagation can flag an adjacent 0x01 byte too,
/// so fingerprint hits are candidates only; exact key equality is authoritative.
#[inline(always)]
fn zero_bytes(word: u64) -> u64 {
    word.wrapping_sub(LOW_BYTES) & !word & HIGH_BYTES
}

#[inline(always)]
fn fingerprint(hash: u64) -> u8 {
    (hash >> 57) as u8
}

// Keep the complete exact key within an aligned 64-byte record.
#[derive(Clone, Copy)]
#[repr(align(64))]
struct Record {
    board: Board,
    node: u32,
}

/// Append-only within a graph layer; clearing retains the allocated buffers.
///
/// Control bytes carry either EMPTY or a seven-bit fingerprint. The first seven
/// bytes are mirrored at the end, allowing unaligned eight-byte groups across
/// the power-of-two table's boundary. Record indices are stored separately and
/// are read only for matching fingerprints; boards never move during probing.
#[derive(Default)]
pub(crate) struct BoardLookup {
    buckets: Vec<u32>,
    control: Vec<u8>,
    records: Vec<Record>,
    // Growth touches only this compact array, without reading and hashing each
    // full board again. Keeping hashes separate avoids padding every Record.
    hashes: Vec<u64>,
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

#[inline(always)]
fn cache_hint<T>(address: *const T) {
    // SAFETY: Callers derive the address from a live, in-bounds allocation.
    // These read-only hints retain no pointer and do not read a Rust value.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{address}]", address = in(reg) address,
            options(nostack, preserves_flags, readonly));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: The same live-allocation guarantee applies to the x86 hint.
    unsafe {
        core::arch::x86_64::_mm_prefetch(address.cast(), core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = address;
}

impl BoardLookup {
    #[inline(always)]
    pub(crate) fn prefetch(&self, hash: u64) {
        if self.buckets.is_empty() {
            return;
        }
        let bucket = hash as usize & (self.buckets.len() - 1);
        cache_hint(self.control.as_ptr().wrapping_add(bucket));
        cache_hint(self.buckets.as_ptr().wrapping_add(bucket));
    }

    #[inline(always)]
    pub(crate) fn prefetch_record(&self, hash: u64) {
        if !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            return;
        }
        if self.buckets.is_empty() {
            return;
        }
        let mask = self.buckets.len() - 1;
        let bucket = hash as usize & mask;
        let control = self.control_group(bucket);
        let mut matches = zero_bytes(control ^ (u64::from(fingerprint(hash)) * LOW_BYTES));
        while matches != 0 {
            let slot = (bucket + matches.trailing_zeros() as usize / 8) & mask;
            let record = (self.buckets[slot] - 1) as usize;
            cache_hint(&self.records[record]);
            matches &= matches - 1;
        }
    }

    /// Keep records backed by a retained graph node and remap their IDs.
    /// The old-to-new map uses `u32::MAX` for a removed node.
    pub(crate) fn remap_nodes(&mut self, remap: &[u32]) {
        let mut retained = 0;
        for old in 0..self.records.len() {
            let node = remap[self.records[old].node as usize];
            if node == u32::MAX {
                continue;
            }
            if old != retained {
                self.records[retained] = self.records[old];
                self.hashes[retained] = self.hashes[old];
            }
            self.records[retained].node = node;
            retained += 1;
        }
        if retained < self.records.len() {
            self.records.truncate(retained);
            self.hashes.truncate(retained);
            self.control.fill(EMPTY);
            for index in 0..retained {
                let hash = self.hashes[index];
                let bucket = self.first_empty(hash);
                self.set_bucket(bucket, hash, index as u32 + 1);
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.control.fill(EMPTY);
        self.records.clear();
        self.hashes.clear();
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
        match self.find(&board, hash) {
            Ok(record) => Some(std::mem::replace(&mut self.records[record].node, node)),
            Err(bucket) => {
                let bucket = self.vacant_bucket_after_reserve(bucket, hash);
                self.insert_at(bucket, board, hash, node);
                None
            }
        }
    }

    /// Resolve an existing node or register a new one with a single probe walk.
    ///
    /// `hash` must come from [`board_hash`]. The supplied hash is retained for
    /// table growth, so a board is hashed only once by its expansion worker.
    /// The boolean reports whether a new board was inserted.
    #[inline(always)]
    pub(crate) fn insert_if_absent_hashed(
        &mut self,
        board: Board,
        hash: u64,
        new_node: u32,
    ) -> (u32, bool) {
        self.insert_if_absent_hashed_ref(&board, hash, new_node)
    }

    /// Probe the expansion's stable board storage directly. A duplicate needs
    /// no owned board temporary; only a new exact key is copied into a record.
    #[inline(always)]
    pub(crate) fn insert_if_absent_hashed_ref(
        &mut self,
        board: &Board,
        hash: u64,
        new_node: u32,
    ) -> (u32, bool) {
        match self.find(board, hash) {
            Ok(record) => (self.records[record].node, false),
            Err(bucket) => {
                let bucket = self.vacant_bucket_after_reserve(bucket, hash);
                self.insert_at(bucket, *board, hash, new_node);
                (new_node, true)
            }
        }
    }

    /// Return either the exact record or a vacant bucket in the first group
    /// containing an empty byte. No removals occur between layer clears.
    #[inline(always)]
    fn find(&self, board: &Board, hash: u64) -> Result<usize, usize> {
        if self.buckets.is_empty() {
            return Err(0);
        }
        let mask = self.buckets.len() - 1;
        let mut bucket = hash as usize & mask;
        let repeated = u64::from(fingerprint(hash)) * LOW_BYTES;
        loop {
            let control = self.control_group(bucket);
            let mut matches = zero_bytes(control ^ repeated);
            while matches != 0 {
                let offset = matches.trailing_zeros() as usize / 8;
                let slot = (bucket + offset) & mask;
                let record = (self.buckets[slot] - 1) as usize;
                if self.records[record].board == *board {
                    return Ok(record);
                }
                matches &= matches - 1;
            }
            let empty = control & HIGH_BYTES;
            if empty != 0 {
                return Err((bucket + empty.trailing_zeros() as usize / 8) & mask);
            }
            bucket = (bucket + GROUP_WIDTH) & mask;
        }
    }

    #[inline(always)]
    fn control_group(&self, bucket: usize) -> u64 {
        // The seven mirrored bytes make every starting bucket's complete group
        // available, including groups crossing the table's last real bucket.
        let mut bytes = [0_u8; GROUP_WIDTH];
        bytes.copy_from_slice(&self.control[bucket..bucket + GROUP_WIDTH]);
        u64::from_le_bytes(bytes)
    }

    #[inline(always)]
    fn insert_at(&mut self, bucket: usize, board: Board, hash: u64, node: u32) {
        let index = u32::try_from(self.records.len())
            .expect("board table exceeds the u32 graph index limit");
        let encoded = index.checked_add(1).expect("board table index overflow");
        self.records.push(Record { board, node });
        self.hashes.push(hash);
        self.set_bucket(bucket, hash, encoded);
    }

    #[inline(always)]
    fn set_bucket(&mut self, bucket: usize, hash: u64, encoded: u32) {
        self.buckets[bucket] = encoded;
        let control = fingerprint(hash);
        self.control[bucket] = control;
        if bucket < GROUP_WIDTH - 1 {
            self.control[bucket + self.buckets.len()] = control;
        }
    }

    #[inline(always)]
    fn first_empty(&self, hash: u64) -> usize {
        let mask = self.buckets.len() - 1;
        let mut bucket = hash as usize & mask;
        loop {
            let empty = self.control_group(bucket) & HIGH_BYTES;
            if empty != 0 {
                return (bucket + empty.trailing_zeros() as usize / 8) & mask;
            }
            bucket = (bucket + GROUP_WIDTH) & mask;
        }
    }

    /// Reserve an upper bound on new exact keys for a complete expansion.
    /// Callers may reserve once before appending a batch of candidates.
    pub(crate) fn reserve(&mut self, additional: usize) {
        let required = self
            .records
            .len()
            .checked_add(additional)
            .expect("board table capacity overflow");
        // Eight-wide probing stays compact at 75% load. Even an unsuccessful
        // probe usually finds an empty slot in its first group.
        if required <= self.buckets.len() / 4 * 3 {
            return;
        }
        let capacity = required
            .checked_mul(4)
            .and_then(|required| required.checked_add(2))
            .map(|required| required / 3)
            .and_then(usize::checked_next_power_of_two)
            .expect("board table capacity overflow")
            .max(INITIAL_BUCKETS);
        self.resize_buckets(capacity);
    }

    #[inline(always)]
    fn vacant_bucket_after_reserve(&mut self, bucket: usize, hash: u64) -> usize {
        if self.records.len() < self.buckets.len() / 4 * 3 {
            return bucket;
        }
        // Probe before reserving: a hit at the load limit never grows the table.
        self.reserve(1);
        self.first_empty(hash)
    }

    #[cold]
    #[inline(never)]
    fn resize_buckets(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        debug_assert!(capacity >= GROUP_WIDTH);
        self.buckets.resize(capacity, 0);
        self.control.resize(capacity + GROUP_WIDTH - 1, EMPTY);
        self.control.fill(EMPTY);
        let additional = (capacity / 4 * 3).saturating_sub(self.records.len());
        self.records.reserve(additional);
        self.hashes.reserve(additional);
        for index in 0..self.hashes.len() {
            let hash = self.hashes[index];
            let bucket = self.first_empty(hash);
            self.set_bucket(bucket, hash, index as u32 + 1);
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

    fn assert_live_probe_indices(lookup: &BoardLookup) {
        assert_eq!(lookup.records.len(), lookup.hashes.len());
        if lookup.buckets.is_empty() {
            assert!(lookup.control.is_empty());
            assert!(lookup.records.is_empty());
            return;
        }
        let capacity = lookup.buckets.len();
        assert!(capacity.is_power_of_two() && capacity >= GROUP_WIDTH);
        assert_eq!(lookup.control.len(), capacity + GROUP_WIDTH - 1);
        assert_eq!(
            &lookup.control[capacity..],
            &lookup.control[..GROUP_WIDTH - 1]
        );
        let mut seen = vec![false; lookup.records.len()];
        for (slot, &control) in lookup.control[..capacity].iter().enumerate() {
            if control == EMPTY {
                // Empty controls may hide stale indices retained by clear().
                continue;
            }
            assert!(control < EMPTY);
            let encoded = lookup.buckets[slot] as usize;
            assert!(encoded > 0 && encoded <= lookup.records.len());
            let index = encoded - 1;
            assert!(!std::mem::replace(&mut seen[index], true));
            assert_eq!(control, fingerprint(lookup.hashes[index]));
        }
        assert!(seen.into_iter().all(|occupied| occupied));
    }

    fn assert_prefetch_is_neutral(lookup: &BoardLookup) {
        let buckets = lookup.buckets.clone();
        let control = lookup.control.clone();
        let hashes = lookup.hashes.clone();
        let records: Vec<_> = lookup.records.iter().map(|r| (r.board, r.node)).collect();
        // Query every fingerprint at the last eight starting slots. These
        // groups cross the mirrored boundary and include absent fingerprints
        // adjacent to occupied ones, exercising zero_bytes' false positives.
        let capacity = lookup.buckets.len().max(GROUP_WIDTH);
        for fingerprint in 0_u64..128 {
            for slot in capacity - GROUP_WIDTH..capacity {
                let hash = (fingerprint << 57) | slot as u64;
                lookup.prefetch(hash);
                lookup.prefetch_record(hash);
            }
        }
        for (record, &hash) in lookup.records.iter().zip(&lookup.hashes) {
            lookup.prefetch(hash);
            lookup.prefetch_record(hash);
            assert_eq!(lookup.get_hashed(&record.board, hash), Some(record.node));
        }
        assert_eq!(lookup.buckets, buckets);
        assert_eq!(lookup.control, control);
        assert_eq!(lookup.hashes, hashes);
        assert_eq!(
            lookup
                .records
                .iter()
                .map(|r| (r.board, r.node))
                .collect::<Vec<_>>(),
            records
        );
        assert_live_probe_indices(lookup);
    }

    #[test]
    fn prefetch_is_neutral_before_allocation_after_growth_and_after_clear() {
        let mut lookup = BoardLookup::default();
        assert_prefetch_is_neutral(&lookup);
        lookup.reserve(12);
        assert_prefetch_is_neutral(&lookup);
        for index in 0..12 {
            let hash = (u64::from(index & 1) << 57) | 15;
            lookup.insert_if_absent_hashed(board(index), hash, index);
        }
        assert_prefetch_is_neutral(&lookup);
        let old_capacity = lookup.buckets.len();
        lookup.reserve(128);
        assert!(lookup.buckets.len() > old_capacity);
        // Place every fingerprint at the same wraparound probe start.
        let start = lookup.buckets.len() as u64 - 1;
        for index in 0..128 {
            let hash = (u64::from(index) << 57) | start;
            lookup.insert_if_absent_hashed(board(index + 1000), hash, index + 1000);
        }
        assert_prefetch_is_neutral(&lookup);
        lookup.clear();
        assert!(lookup.buckets.iter().any(|&index| index != 0));
        assert_prefetch_is_neutral(&lookup);
        // Reuse fewer records: stale indices now also exceed records.len().
        lookup.insert_if_absent_hashed(board(2000), 15, 2000);
        assert_prefetch_is_neutral(&lookup);
    }

    #[test]
    fn mixed_collisions_reservation_clear_and_prefetch_match_exact_reference() {
        let mut lookup = BoardLookup::default();
        let mut reference = HashMap::new();
        for step in 0_u32..384 {
            if step != 0 && step % 97 == 0 {
                lookup.clear();
                reference.clear();
                assert_prefetch_is_neutral(&lookup);
            }
            if step % 29 == 0 {
                lookup.reserve((step % 47 + 1) as usize);
            }
            let index = (step * 17 + step / 7) % 48;
            let key = board(index);
            // Two adjacent fingerprints and one shared probe start deliberately
            // create complete hash collisions and borrow-related candidates.
            let hash = (u64::from(index & 1) << 57) | 15;
            lookup.prefetch(hash);
            lookup.prefetch_record(hash);
            if step % 3 == 0 {
                assert_eq!(
                    lookup.insert_hashed(key, hash, step),
                    reference.insert(key, step)
                );
            } else {
                let fresh = !reference.contains_key(&key);
                let expected = *reference.entry(key).or_insert(step);
                let actual = if step & 1 == 0 {
                    lookup.insert_if_absent_hashed(key, hash, step)
                } else {
                    lookup.insert_if_absent_hashed_ref(&key, hash, step)
                };
                assert_eq!(actual, (expected, fresh));
            }
            assert_live_probe_indices(&lookup);
            assert_eq!(lookup.records.len(), reference.len());
            for (key, &node) in &reference {
                let hash = (u64::from(key.ren as u32 & 1) << 57) | 15;
                assert_eq!(lookup.get_hashed(key, hash), Some(node));
            }
        }
        assert_prefetch_is_neutral(&lookup);
    }

    #[test]
    fn equal_hashes_still_require_full_board_equality() {
        let mut lookup = BoardLookup::default();
        lookup.resize_buckets(64);
        // Force the entire hash to collide, not merely the seven-bit fingerprint.
        // Growth must also preserve these deliberately artificial hashes.
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
    fn full_hash_collisions_survive_growth_and_reservation() {
        let mut lookup = BoardLookup::default();
        let hash = 0x1234_5678_ffff_ffff;
        for index in 0..1000 {
            if index % 100 == 0 {
                lookup.reserve(100);
            }
            assert_eq!(
                lookup.insert_if_absent_hashed(board(index), hash, index),
                (index, true)
            );
        }
        for index in 0..1000 {
            assert_eq!(lookup.get_hashed(&board(index), hash), Some(index));
        }
        assert_eq!(lookup.get_hashed(&board(1000), hash), None);
    }

    #[test]
    fn duplicate_at_load_limit_does_not_grow() {
        let mut lookup = BoardLookup::default();
        lookup.reserve(12);
        for index in 0..12 {
            lookup.insert(board(index), index);
        }
        assert_eq!(lookup.buckets.len(), INITIAL_BUCKETS);
        assert_eq!(lookup.insert(board(0), 999), Some(0));
        assert_eq!(lookup.buckets.len(), INITIAL_BUCKETS);
        lookup.insert(board(12), 12);
        assert_eq!(lookup.buckets.len(), INITIAL_BUCKETS * 2);
    }

    #[test]
    fn wrapping_groups_and_adjacent_fingerprint_candidates_stay_exact() {
        let mut lookup = BoardLookup::default();
        lookup.reserve(12);
        // All start at the final bucket; consecutive fingerprints exercise the
        // zero-byte detector's borrow-propagation false positive as well.
        for index in 0..12 {
            let hash = (u64::from(index & 1) << 57) | 15;
            assert_eq!(
                lookup.insert_if_absent_hashed(board(index), hash, index),
                (index, true)
            );
        }
        for index in 0..12 {
            let hash = (u64::from(index & 1) << 57) | 15;
            assert_eq!(lookup.get_hashed(&board(index), hash), Some(index));
        }
        lookup.reserve(100);
        for index in 0..12 {
            let hash = (u64::from(index & 1) << 57) | 15;
            assert_eq!(lookup.get_hashed(&board(index), hash), Some(index));
        }
        lookup.clear();
        for index in 0..12 {
            let hash = (u64::from(index & 1) << 57) | 15;
            assert_eq!(lookup.get_hashed(&board(index), hash), None);
            assert_eq!(
                lookup.insert_if_absent_hashed(board(index), hash, index),
                (index, true)
            );
        }
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
    fn clear_removes_old_keys_and_allows_reinsertion() {
        let mut lookup = BoardLookup::default();
        for index in 0..1000 {
            lookup.insert(board(index), index);
        }
        lookup.clear();
        assert_eq!(lookup.keys().count(), 0);
        for index in 0..1000 {
            assert_eq!(lookup.get(&board(index)), None);
            assert_eq!(lookup.insert(board(index), index + 7), None);
        }
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
        }
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(lookup.get(key), Some(&(index as u32)));
        }
    }
}
