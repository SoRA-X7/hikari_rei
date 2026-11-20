//! Reuse each edge's score and weight during a ranked random walk.

use super::{Edge, Layer, Random};

/// Match the reference walk while calculating each score and weight once.
/// Expansion limits each draw kind to 255 edges, including speculation.
pub(super) fn pick_ranked_cached(
    edges: &[Edge],
    children: &Layer,
    rng: &mut Random,
    scratch: &mut [f32; 255],
) -> usize {
    debug_assert!(edges.len() <= scratch.len());
    let weights = &mut scratch[..edges.len()];
    let mut minimum = f32::INFINITY;
    for (edge, score) in edges.iter().zip(weights.iter_mut()) {
        *score = children.nodes[edge.node as usize]
            .value
            .with_reward(edge.reward)
            .score() as f32;
        // Dead children still participate in the reference minimum.
        minimum = minimum.min(*score);
    }
    let sum: f32 = edges
        .iter()
        .zip(weights.iter_mut())
        .enumerate()
        .map(|(rank, (edge, weight))| {
            *weight = if children.nodes[edge.node as usize].dead {
                0.0
            } else {
                let distance = *weight - minimum + 1.0;
                let rank = (rank + 1) as f32;
                distance * distance / (rank * rank) + 40.0
            };
            *weight
        })
        .sum();
    if sum <= f32::EPSILON {
        return 0;
    }
    let random = rng.fraction() * sum;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative > random {
            return index;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Reward, Value};
    use crate::search::{Node, pick_ranked};

    fn edge(node: u32, reward: i32) -> Edge {
        Edge {
            node,
            piece: None,
            reward: Reward {
                evaluation: reward,
                attack: 0,
            },
            time: 0,
        }
    }

    fn compare(edges: &[Edge], layer: &Layer, seed: u64) -> usize {
        let mut reference = Random::new(seed);
        let mut actual = Random::new(seed);
        // Reused storage must not depend on values left by a previous walk.
        let mut scratch = [f32::NAN; 255];
        let expected = pick_ranked(edges, layer, &mut reference);
        let result = pick_ranked_cached(edges, layer, &mut actual, &mut scratch);
        assert_eq!(result, expected);
        assert_eq!(actual.0, reference.0, "random consumption changed");
        result
    }

    #[test]
    fn randomized_ranked_walk_matches_reference_index_and_rng() {
        let mut source = Random::new(0x5412_720f_6ab1_348e);
        let mut scratch = [f32::NAN; 255];
        for case in 0..512 {
            let node_count = (source.next() % 64 + 1) as usize;
            let edge_count = (source.next() % 256) as usize;
            let mut layer = Layer::default();
            for index in 0..node_count {
                let components = match case % 5 {
                    0 => [0; 4],
                    1 => [source.next() as i32 % 7; 4],
                    2 => [i32::MAX, i32::MIN, index as i32, source.next() as i32],
                    _ => std::array::from_fn(|_| source.next() as i32),
                };
                let mut node = Node::new(Value {
                    value: components,
                    spike: source.next() as i32,
                });
                node.dead = match case % 7 {
                    0 => true,
                    1 => false,
                    2 => index % 2 == 0,
                    _ => source.next() & 3 == 0,
                };
                layer.nodes.push(node);
            }
            let edges: Vec<_> = (0..edge_count)
                .map(|_| {
                    let index = (source.next() % node_count as u64) as u32;
                    let reward = if case % 5 == 0 {
                        0
                    } else {
                        source.next() as i32
                    };
                    edge(index, reward)
                })
                .collect();
            for _ in 0..16 {
                let seed = source.next();
                let mut expected = Random::new(seed);
                let mut actual = Random::new(seed);
                let reference = pick_ranked(&edges, &layer, &mut expected);
                let result = pick_ranked_cached(&edges, &layer, &mut actual, &mut scratch);
                assert_eq!(result, reference, "case {case}, seed {seed}");
                assert_eq!(actual.0, expected.0, "case {case}, seed {seed}");
            }
        }
    }

    #[test]
    fn empty_and_all_dead_edges_do_not_consume_rng() {
        let mut layer = Layer::default();
        let mut dead = Node::new(Value {
            value: [i32::MIN, i32::MAX, 0, 0],
            spike: 0,
        });
        dead.dead = true;
        layer.nodes.push(dead);
        for edges in [vec![], vec![edge(0, i32::MAX); 255]] {
            let mut rng = Random::new(37);
            let before = rng.0;
            let mut scratch = [1.0; 255];
            assert_eq!(
                pick_ranked_cached(&edges, &layer, &mut rng, &mut scratch),
                0
            );
            assert_eq!(rng.0, before);
            compare(&edges, &layer, 37);
        }
    }

    #[test]
    fn zero_random_draw_uses_strict_cdf_comparison() {
        let mut layer = Layer::default();
        let mut dead = Node::new(Value::default());
        dead.dead = true;
        layer.nodes = vec![dead, Node::new(Value::default())];
        // SplitMix's next state and output are both zero for this state.
        let seed = 0_u64.wrapping_sub(0x9e3779b97f4a7c15);
        assert_eq!(compare(&[edge(0, 0), edge(1, 0)], &layer, seed), 1);
    }

    #[test]
    fn dead_minimum_ties_and_wrapping_rewards_match_reference() {
        let mut layer = Layer::default();
        let mut dead = Node::new(Value {
            value: [-1_000_000, 0, 0, 0],
            spike: 0,
        });
        dead.dead = true;
        layer.nodes = vec![
            dead,
            Node::new(Value::default()),
            Node::new(Value::default()),
            Node::new(Value {
                value: [i32::MAX, i32::MAX, 10, -100],
                spike: i32::MAX,
            }),
        ];
        let edges = [edge(0, 0), edge(1, 0), edge(2, 0), edge(3, i32::MAX)];
        for seed in 0..256 {
            compare(&edges, &layer, seed);
        }
    }
}
