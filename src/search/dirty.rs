//! Back up completed work while stopping at unchanged ancestor values.

use super::{Layer, NONE, Value, propagate_layers};

#[derive(Default)]
pub(super) struct DirtyScratch {
    forced: Vec<Vec<u32>>,
    previous: Vec<(u32, Value, bool)>,
}

#[cfg(test)]
pub(super) fn propagate_changed(
    layers: &mut [Layer],
    updates: &mut [Vec<u32>],
    pool: Option<&rayon::ThreadPool>,
) {
    propagate_changed_with_scratch(layers, updates, pool, &mut DirtyScratch::default());
}

/// Propagate a set of completed updates to every affected ancestor.
///
/// The caller may already have changed a seed's value, children, or death flag,
/// so every original seed reaches all of its parents even when its backup does
/// not change its current state. Later ancestors only propagate when their full
/// value or death flag changes. A layer is joined before the next one starts.
pub(super) fn propagate_changed_with_scratch(
    layers: &mut [Layer],
    updates: &mut [Vec<u32>],
    pool: Option<&rayon::ThreadPool>,
    scratch: &mut DirtyScratch,
) {
    let DirtyScratch { forced, previous } = scratch;
    if forced.len() < updates.len() {
        forced.resize_with(updates.len(), Vec::new);
    }
    for (seeds, updates) in forced.iter_mut().zip(updates.iter()) {
        seeds.clear();
        seeds.extend_from_slice(updates);
        seeds.sort_unstable();
        seeds.dedup();
    }

    for depth in (0..updates.len()).rev() {
        let current_updates = &mut updates[depth];
        current_updates.sort_unstable();
        current_updates.dedup();
        if current_updates.is_empty() {
            continue;
        }
        previous.clear();
        previous.extend(current_updates.iter().map(|&index| {
            let node = layers[depth].nodes[index as usize];
            (index, node.value, node.dead)
        }));

        // A one-layer update window makes this layer relative depth zero:
        // reuse exactly the existing serial/parallel backup, without its
        // unconditional parent gathering. Child layers remain available.
        propagate_layers(
            &mut layers[depth..],
            std::slice::from_mut(current_updates),
            pool,
        );

        if depth == 0 {
            continue;
        }
        let current = &layers[depth];
        for &(index, value, dead) in previous.iter() {
            let node = current.nodes[index as usize];
            if forced[depth].binary_search(&index).is_err()
                && node.value == value
                && node.dead == dead
            {
                continue;
            }
            let mut link = node.parent_head;
            while link != NONE {
                let parent = current.parents[link as usize];
                updates[depth - 1].push(parent.parent);
                link = parent.next;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PieceKind;
    use crate::eval::{Reward, Value};
    use crate::search::{Children, Edge, KINDS, Node, ParentLink};

    // Keep the original serial algorithm independent of the production backup
    // helpers: the comparison must detect errors in either update traversal.
    fn serial_oracle(layers: &mut [Layer], mut updates: Vec<Vec<u32>>) {
        for depth in (0..updates.len()).rev() {
            updates[depth].sort_unstable();
            updates[depth].dedup();
            for &index in &updates[depth] {
                let node = layers[depth].nodes[index as usize];
                if !node.dead
                    && let Some(children) = node.children
                {
                    let (before, after) = layers.split_at_mut(depth + 1);
                    let current = &mut before[depth];
                    let child_nodes = &after[0].nodes;
                    let mut sum = [0_i32; 4];
                    let mut spike = 0;
                    let mut count = 0;
                    let mut dead = false;
                    for kind in KINDS {
                        if children.possibilities & (1 << kind as u8) == 0 {
                            continue;
                        }
                        let edges = &mut current.edges[children.range(kind)];
                        edges.sort_unstable_by(|a, b| {
                            let a_node = child_nodes[a.node as usize];
                            let b_node = child_nodes[b.node as usize];
                            a_node.dead.cmp(&b_node.dead).then_with(|| {
                                b_node
                                    .value
                                    .with_reward(b.reward)
                                    .score()
                                    .cmp(&a_node.value.with_reward(a.reward).score())
                            })
                        });
                        let mut best: Option<Value> = None;
                        for edge in edges {
                            let child = child_nodes[edge.node as usize];
                            if !child.dead {
                                let value = child.value.with_reward(edge.reward);
                                best = Some(best.map_or(value, |old| old.improve(value)));
                            }
                        }
                        if let Some(value) = best {
                            for (total, component) in sum.iter_mut().zip(value.value) {
                                *total = total.wrapping_add(component);
                            }
                            spike = spike.max(value.spike);
                            count += 1;
                        } else {
                            dead = true;
                        }
                    }
                    if dead || count == 0 {
                        current.nodes[index as usize].dead = true;
                    } else {
                        current.nodes[index as usize].value = Value {
                            value: sum.map(|component| component / count),
                            spike,
                        };
                    }
                }
            }
            if depth > 0 {
                for offset in 0..updates[depth].len() {
                    let index = updates[depth][offset];
                    let current = &layers[depth];
                    let mut link = current.nodes[index as usize].parent_head;
                    while link != NONE {
                        let parent = current.parents[link as usize];
                        updates[depth - 1].push(parent.parent);
                        link = parent.next;
                    }
                }
            }
        }
    }

    fn clone_layers(layers: &[Layer]) -> Vec<Layer> {
        layers
            .iter()
            .map(|layer| Layer {
                nodes: layer.nodes.clone(),
                edges: layer.edges.clone(),
                parents: layer.parents.clone(),
                ..Layer::default()
            })
            .collect()
    }

    fn empty_layers(widths: &[usize]) -> Vec<Layer> {
        widths
            .iter()
            .map(|&width| Layer {
                nodes: vec![Node::new(Value::default()); width],
                ..Layer::default()
            })
            .collect()
    }

    fn connect(layers: &mut [Layer], depth: usize, parent: u32, children: &[u32]) {
        let (before, after) = layers.split_at_mut(depth + 1);
        let current = &mut before[depth];
        let next = &mut after[0];
        let mut lengths = [0; 7];
        lengths[PieceKind::T as usize] = children.len() as u16;
        current.nodes[parent as usize].children = Some(Children {
            start: current.edges.len() as u32,
            lengths,
            possibilities: 1 << PieceKind::T as u8,
        });
        for &child in children {
            current.edges.push(Edge {
                node: child,
                piece: None,
                reward: Reward::default(),
                time: 0,
            });
            let next_link = next.nodes[child as usize].parent_head;
            next.nodes[child as usize].parent_head = next.parents.len() as u32;
            next.parents.push(ParentLink {
                parent,
                next: next_link,
            });
        }
    }

    fn settle(layers: &mut [Layer]) {
        let updates = layers
            .iter()
            .map(|layer| (0..layer.nodes.len() as u32).collect())
            .collect();
        serial_oracle(layers, updates);
    }

    fn fixture() -> Vec<Layer> {
        let mut layers = empty_layers(&[3, 4, 4, 4]);
        for (index, node) in layers[3].nodes.iter_mut().enumerate() {
            node.value.value[0] = (index as i32 + 1) * 10;
        }
        for depth in [2, 1] {
            // Append order deliberately differs from node order.
            for (parent, children) in [(2, [2, 3]), (0, [0, 1]), (3, [0, 3]), (1, [1, 2])] {
                connect(&mut layers, depth, parent, &children);
            }
        }
        for parent in 0..3 {
            connect(&mut layers, 0, parent, &[parent, parent + 1]);
        }
        settle(&mut layers);
        layers
    }

    fn assert_same_graph(actual: &[Layer], expected: &[Layer]) {
        assert_eq!(actual.len(), expected.len());
        for (depth, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.nodes.len(), expected.nodes.len());
            assert_eq!(actual.edges.len(), expected.edges.len());
            assert_eq!(actual.parents.len(), expected.parents.len());
            for (index, (actual, expected)) in actual.nodes.iter().zip(&expected.nodes).enumerate()
            {
                assert_eq!(actual.value, expected.value, "value at {depth}/{index}");
                assert_eq!(actual.dead, expected.dead, "death at {depth}/{index}");
                assert_eq!(actual.parent_head, expected.parent_head);
                assert_eq!(
                    actual
                        .children
                        .map(|c| (c.start, c.lengths, c.possibilities)),
                    expected
                        .children
                        .map(|c| (c.start, c.lengths, c.possibilities))
                );
            }
            for (actual, expected) in actual.edges.iter().zip(&expected.edges) {
                assert_eq!(actual.node, expected.node);
                assert_eq!(actual.piece, expected.piece);
                assert_eq!(actual.reward, expected.reward);
                assert_eq!(actual.time, expected.time);
            }
            for (actual, expected) in actual.parents.iter().zip(&expected.parents) {
                assert_eq!(actual.parent, expected.parent);
                assert_eq!(actual.next, expected.next);
            }
        }
    }

    fn check_against_oracle(layers: &mut [Layer], mut updates: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let mut expected = clone_layers(layers);
        serial_oracle(&mut expected, updates.clone());
        propagate_changed(layers, &mut updates, None);
        assert_same_graph(layers, &expected);
        updates
    }

    #[test]
    fn changed_leaf_reaches_all_transposition_parents() {
        let mut layers = fixture();
        layers[3].nodes[0].value.value[0] = 100;
        let updates = check_against_oracle(&mut layers, vec![vec![], vec![], vec![], vec![0, 0]]);
        assert_eq!(updates[2], [0, 3]);
        assert_eq!(layers[0].nodes[0].value.value[0], 100);
        assert_eq!(layers[0].nodes[2].value.value[0], 100);
    }

    #[test]
    fn unchanged_ancestors_stop_but_seed_still_reaches_parents() {
        let mut layers = fixture();
        let updates = check_against_oracle(&mut layers, vec![vec![], vec![], vec![], vec![0]]);
        assert_eq!(updates[2], [0, 3]);
        assert!(updates[1].is_empty());
        assert!(updates[0].is_empty());
    }

    #[test]
    fn externally_dead_seed_reaches_parents_despite_unchanged_backup() {
        let mut layers = fixture();
        layers[2].nodes[1].dead = true;
        let updates = check_against_oracle(&mut layers, vec![vec![], vec![], vec![1], vec![]]);
        assert_eq!(updates[1], [0, 1]);
        assert_eq!(layers[1].nodes[0].value.value[0], 20);
    }

    #[test]
    fn unchanged_dead_intermediate_does_not_block_other_parent_paths() {
        let mut layers = fixture();
        layers[2].nodes[3].dead = true;
        settle(&mut layers);
        layers[3].nodes[0].value.value[0] = 100;
        check_against_oracle(&mut layers, vec![vec![], vec![], vec![], vec![0]]);
        assert!(layers[2].nodes[3].dead);
        assert_eq!(layers[0].nodes[0].value.value[0], 100);
    }

    #[test]
    fn equal_score_with_different_components_and_spike_still_propagates() {
        let mut layers = fixture();
        layers[3].nodes[3].value = Value {
            value: [10, 30, 0, 0],
            spike: 9,
        };
        check_against_oracle(&mut layers, vec![vec![], vec![], vec![], vec![3]]);
        assert_eq!(layers[0].nodes[2].value.value, [10, 30, 0, 0]);
        assert_eq!(layers[0].nodes[2].value.spike, 9);
    }

    #[test]
    fn newly_dead_empty_expansion_propagates_to_every_parent() {
        let mut layers = fixture();
        // The registered expansion has no surviving response for its draw.
        connect(&mut layers, 2, 1, &[]);
        check_against_oracle(&mut layers, vec![vec![], vec![], vec![1], vec![]]);
        assert!(layers[2].nodes[1].dead);
        assert_eq!(layers[1].nodes[0].value.value[0], 20);
    }

    #[test]
    fn seed_at_shallower_layer_remains_forced_when_deeper_updates_join_it() {
        let mut layers = fixture();
        layers[2].nodes[0].value.value[0] = -500;
        // Its backup restores the old value, while the independent leaf seed
        // does not change the rest of the graph. The seed must still reach up.
        let updates = check_against_oracle(&mut layers, vec![vec![], vec![], vec![0, 0], vec![3]]);
        assert!(updates[1].contains(&0));
        assert!(updates[1].contains(&3));
    }

    #[test]
    fn large_layer_uses_existing_parallel_backup_and_matches_serial_oracle() {
        let mut layers = empty_layers(&[3, 20, 40, 64]);
        for (index, node) in layers[3].nodes.iter_mut().enumerate() {
            node.value.value[0] = index as i32;
        }
        for parent in (0..40).rev() {
            let children: Vec<_> = (0..256).map(|offset| (parent + offset) % 64).collect();
            connect(&mut layers, 2, parent, &children);
        }
        let children: Vec<_> = (0..40).collect();
        for parent in (0..20).rev() {
            connect(&mut layers, 1, parent, &children);
        }
        let children: Vec<_> = (0..20).collect();
        for parent in 0..3 {
            connect(&mut layers, 0, parent, &children);
        }
        settle(&mut layers);
        layers[3].nodes[3].value = Value {
            value: [100, 2, 3, 4],
            spike: 7,
        };
        let seeds = vec![vec![], vec![], vec![], vec![3, 3]];
        let mut expected = clone_layers(&layers);
        serial_oracle(&mut expected, seeds.clone());
        assert!(layers[2].nodes.len() >= super::super::PARALLEL_BACKUP_MIN_NODES);
        assert!(layers[2].edges.len() >= super::super::PARALLEL_BACKUP_MIN_EDGES);
        for threads in [2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut actual = clone_layers(&layers);
            let mut updates = seeds.clone();
            propagate_changed(&mut actual, &mut updates, Some(&pool));
            assert_eq!(updates[2].len(), 40);
            assert_same_graph(&actual, &expected);
        }
    }
}
