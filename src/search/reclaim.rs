//! Reuse graph capacity held by former root siblings.

use super::{Layer, NONE, ParentLink, Search};

impl Search {
    /// Only the TBP controller's idle path enables reclamation. Limit attempts
    /// to one per root and leave small custom limits to the normal cap rule.
    pub(crate) fn reclaim_at_cap(&mut self, last_epoch: &mut Option<u64>) -> bool {
        const MIN_REMOVED: usize = 750_000;
        if !self.counters.memory_limited
            || self.config.max_nodes < MIN_REMOVED * 2
            || self.layers[0].nodes.len() < 2
            || self.layers[0].nodes[self.root as usize].dead
            || *last_epoch == Some(self.epoch)
        {
            return false;
        }
        *last_epoch = Some(self.epoch);
        let retained = self
            .layers
            .iter()
            .map(|layer| layer.nodes.len())
            .sum::<usize>();
        let reachable = count_reachable(self.layers.make_contiguous(), self.root);
        if retained - reachable < MIN_REMOVED.max(retained / 2) {
            return false;
        }
        // Old worker completions carry the pre-compaction generation and are
        // discarded before they can try to register against remapped IDs.
        self.cancel_stream();
        compact(self.layers.make_contiguous(), self.root);
        self.root = 0;
        self.counters.memory_limited = false;
        debug_assert_eq!(self.stats().nodes, reachable);
        true
    }
}

/// Follow every materialized draw group. Inactive groups remain, along with
/// permanently dead nodes, so this does not change the root's stored DAG.
fn count_reachable(layers: &[Layer], root: u32) -> usize {
    let mut current = vec![false; layers[0].nodes.len()];
    current[root as usize] = true;
    let mut next = Vec::new();
    let mut count = 1;
    for pair in layers.windows(2) {
        next.resize(pair[1].nodes.len(), false);
        for (&marked, node) in current.iter().zip(&pair[0].nodes) {
            if !marked {
                continue;
            }
            if let Some(children) = node.children {
                let start = children.start as usize;
                let length = children.lengths.iter().map(|&n| n as usize).sum::<usize>();
                for edge in &pair[0].edges[start..start + length] {
                    let mark = &mut next[edge.node as usize];
                    count += usize::from(!*mark);
                    *mark = true;
                }
            }
        }
        std::mem::swap(&mut current, &mut next);
        next.clear();
    }
    count
}

fn compact(layers: &mut [Layer], root: u32) {
    let mut remap = vec![NONE; layers[0].nodes.len()];
    remap[root as usize] = 0;
    layers[0].nodes[root as usize].parent_head = NONE;
    layers[0].parents.clear();
    let mut next_remap = Vec::new();
    let mut ranges = Vec::new();
    for depth in 0..layers.len() {
        let (before, after) = layers.split_at_mut(depth + 1);
        let layer = &mut before[depth];
        ranges.clear();
        let mut new_len = 0;
        for (old, &new) in remap.iter().enumerate() {
            if new == NONE {
                continue;
            }
            let node = layer.nodes[old];
            layer.nodes[new as usize] = node;
            new_len += 1;
            if let Some(children) = node.children {
                let length = children.lengths.iter().map(|&n| n as usize).sum::<usize>();
                ranges.push((children.start as usize, length, new));
            }
        }
        layer.nodes.truncate(new_len);
        #[cfg(feature = "fast-registration")]
        layer.lookup.remap_nodes(&remap);
        #[cfg(not(feature = "fast-registration"))]
        layer.lookup.retain(|_, node| {
            *node = remap[*node as usize];
            *node != NONE
        });
        let Some(child) = after.first_mut() else {
            debug_assert!(ranges.is_empty());
            break;
        };
        next_remap.resize(child.nodes.len(), NONE);
        for &(start, length, _) in &ranges {
            for edge in &layer.edges[start..start + length] {
                next_remap[edge.node as usize] = 0;
            }
        }
        let mut new_child = 0;
        for (node, index) in child.nodes.iter_mut().zip(&mut next_remap) {
            if *index != NONE {
                *index = new_child;
                node.parent_head = NONE;
                new_child += 1;
            }
        }
        child.parents.clear();
        // Move ranges toward the front in their original order. An empty
        // range at the same offset as a populated one consumes no space.
        ranges.sort_unstable();
        let mut end = 0;
        for &(start, length, parent) in &ranges {
            let children = layer.nodes[parent as usize].children.as_mut().unwrap();
            children.start = end as u32;
            for offset in 0..length {
                let mut edge = layer.edges[start + offset];
                let child_node = &mut child.nodes[edge.node as usize];
                let next =
                    std::mem::replace(&mut child_node.parent_head, child.parents.len() as u32);
                child.parents.push(ParentLink { parent, next });
                edge.node = next_remap[edge.node as usize];
                layer.edges[end + offset] = edge;
            }
            end += length;
        }
        layer.edges.truncate(end);
        std::mem::swap(&mut remap, &mut next_remap);
        next_remap.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::eval::Weights;
    use crate::search::{Config, KINDS};
    use std::collections::{HashMap, HashSet};

    fn fixture() -> Search {
        let mut search = Search::new(Config::default(), Weights::default()).unwrap();
        search.start(Board::default(), KINDS.to_vec());
        for _ in 0..3 {
            for _ in 0..200 {
                search.search_step();
            }
            let piece = search.current_plan().into_iter().flatten().next().unwrap();
            search.commit_physical(piece).unwrap();
            search.add_next_piece(piece.kind).unwrap();
        }
        for _ in 0..150 {
            search.search_step();
        }
        search
    }

    fn assert_same_reachable_graph(before: &Search, after: &Search) {
        let mut nodes = HashMap::from([(before.root, after.root)]);
        let mut original_nodes = 0;
        for (depth, (original, swept)) in before.layers.iter().zip(&after.layers).enumerate() {
            assert_eq!(nodes.len(), swept.nodes.len(), "layer {depth}");
            let mut seen = HashSet::new();
            let mut kept_boards = 0;
            for board in original.lookup.keys() {
                let old = original.lookup.get(board).expect("lookup has its own key");
                let new = nodes.get(old);
                assert_eq!(swept.lookup.get(board), new);
                kept_boards += usize::from(new.is_some());
            }
            assert_eq!(swept.lookup.keys().count(), kept_boards);
            let mut next_nodes = HashMap::new();
            let mut next_parents: HashMap<u32, Vec<u32>> = HashMap::new();
            let mut kept_edges = 0;
            for (old, new) in nodes.drain() {
                assert!(seen.insert(new));
                original_nodes += 1;
                let left = original.nodes[old as usize];
                let right = swept.nodes[new as usize];
                assert_eq!(left.value, right.value);
                assert_eq!(left.dead, right.dead);
                match (left.children, right.children) {
                    (Some(left), Some(right)) => {
                        assert_eq!(left.lengths, right.lengths);
                        assert_eq!(left.possibilities, right.possibilities);
                        for kind in KINDS {
                            for (l, r) in original.edges[left.range(kind)]
                                .iter()
                                .zip(&swept.edges[right.range(kind)])
                            {
                                assert_eq!(
                                    (l.piece, l.reward, l.time),
                                    (r.piece, r.reward, r.time)
                                );
                                kept_edges += 1;
                                assert_eq!(*next_nodes.entry(l.node).or_insert(r.node), r.node);
                                next_parents.entry(r.node).or_default().push(new);
                            }
                        }
                    }
                    (None, None) => {}
                    _ => panic!("compaction changed whether a reachable node was expanded"),
                }
            }
            assert_eq!(kept_edges, swept.edges.len());
            if let Some(child) = after.layers.get(depth + 1) {
                assert_eq!(child.parents.len(), kept_edges);
                for (id, mut expected) in next_parents {
                    let mut actual = Vec::new();
                    let mut link = child.nodes[id as usize].parent_head;
                    while link != NONE {
                        let parent = child.parents[link as usize];
                        actual.push(parent.parent);
                        link = parent.next;
                    }
                    expected.sort_unstable();
                    actual.sort_unstable();
                    assert_eq!(expected, actual);
                }
            }
            nodes = next_nodes;
        }
        assert_eq!(original_nodes, after.stats().nodes);
        assert!(nodes.is_empty());
        assert!(original_nodes < before.stats().nodes);
    }

    #[test]
    fn compact_preserves_reachable_dag_and_allows_prior_moves_to_commit() {
        let mut before = fixture();
        let mut after = fixture();
        let chosen = before.best_move().unwrap();
        let original_plan = before.current_plan();
        let original_ranked = serde_json::to_value(before.ranked_moves()).unwrap();
        let retained = count_reachable(after.layers.make_contiguous(), after.root);
        compact(after.layers.make_contiguous(), after.root);
        after.root = 0;
        assert_eq!(after.stats().nodes, retained);
        assert_same_reachable_graph(&before, &after);
        assert_eq!(after.current_plan(), original_plan);
        assert_eq!(
            serde_json::to_value(after.ranked_moves()).unwrap(),
            original_ranked
        );
        before.advance(&chosen).unwrap();
        after.advance(&chosen).unwrap();
        assert_eq!(before.board(), after.board());
        assert_eq!(before.queue(), after.queue());
        before.add_next_piece(KINDS[2]).unwrap();
        after.add_next_piece(KINDS[2]).unwrap();
        assert_eq!(before.current_plan(), after.current_plan());
        for _ in 0..20 {
            assert!(after.search_step() > 0);
        }
    }
}
