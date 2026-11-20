//! Compare the streamlined backup scan with the original full improvement loop.

use super::{Children, Edge, KINDS, Node, Random, Value, backup_edges};
use crate::board::Piece;
use crate::eval::Reward;

// Deliberately keep the old scan and comparator independent of backup_edges.
fn original_backup(children: Children, edges: &mut [Edge], child_nodes: &[Node]) -> Option<Value> {
    let mut sum = [0_i32; 4];
    let mut spike = 0;
    let mut count = 0_i32;
    let mut dead = false;
    for kind in KINDS {
        if children.possibilities & (1 << kind as u8) == 0 {
            continue;
        }
        let range = children.range(kind);
        let edges =
            &mut edges[range.start - children.start as usize..range.end - children.start as usize];
        edges.sort_unstable_by(|a, b| {
            let a_node = &child_nodes[a.node as usize];
            let b_node = &child_nodes[b.node as usize];
            a_node.dead.cmp(&b_node.dead).then_with(|| {
                b_node
                    .value
                    .with_reward(b.reward)
                    .score()
                    .cmp(&a_node.value.with_reward(a.reward).score())
            })
        });
        let mut best: Option<Value> = None;
        for edge in edges.iter() {
            let child = child_nodes[edge.node as usize];
            if !child.dead {
                let value = child.value.with_reward(edge.reward);
                best = Some(best.map_or(value, |previous| previous.improve(value)));
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
    (!dead && count != 0).then(|| Value {
        value: sum.map(|component| component / count),
        spike,
    })
}

fn edge(node: usize, order: usize, reward: Reward) -> Edge {
    Edge {
        node: node as u32,
        piece: Some(Piece::new(
            KINDS[order % 7],
            (order % 11) as i8 - 1,
            (order % 24) as i8,
            (order % 4) as u8,
        )),
        reward,
        time: order as i32,
    }
}

fn compare(children: Children, edges: Vec<Edge>, nodes: &[Node]) -> Option<Value> {
    let mut expected_edges = edges.clone();
    let mut actual_edges = edges;
    let expected = original_backup(children, &mut expected_edges, nodes);
    let actual = backup_edges(children, &mut actual_edges, nodes);
    assert_eq!(actual, expected, "full Value or speculative death changed");
    assert_eq!(actual_edges.len(), expected_edges.len());
    for (actual, expected) in actual_edges.iter().zip(&expected_edges) {
        assert_eq!(
            (actual.node, actual.piece, actual.reward, actual.time),
            (
                expected.node,
                expected.piece,
                expected.reward,
                expected.time
            ),
            "the unstable sort's exact tie order changed"
        );
    }
    actual
}

#[test]
fn randomized_known_and_speculative_backups_match_original_full_values_and_edge_order() {
    let mut rng = Random::new(0x54bc_cda7_9621_008f);
    for case in 0..768 {
        let node_count = (rng.next() % 97 + 1) as usize;
        let mut nodes = Vec::with_capacity(node_count);
        for index in 0..node_count {
            let value = match case % 6 {
                0 => [0; 4],
                1 => [index as i32, -(index as i32), 17, 0],
                2 => [i32::MAX, i32::MIN, rng.next() as i32, i32::MAX],
                3 => std::array::from_fn(|_| (rng.next() % 13) as i32 - 6),
                _ => std::array::from_fn(|_| rng.next() as i32),
            };
            let mut node = Node::new(Value {
                value,
                spike: rng.next() as i32,
            });
            node.dead = match case % 5 {
                0 => false,
                1 => true,
                _ => rng.next() & 3 == 0,
            };
            nodes.push(node);
        }
        let mut lengths = [0_u16; 7];
        for (kind, length) in lengths.iter_mut().enumerate() {
            *length = if case % 23 == kind {
                255
            } else {
                (rng.next() % 19) as u16
            };
        }
        let possibilities = match case % 4 {
            0 => 1 << (case % 7),
            1 => 127,
            _ => (rng.next() & 127) as u8,
        };
        let children = Children {
            start: (rng.next() % 1000) as u32,
            lengths,
            possibilities,
        };
        let count: usize = lengths.iter().map(|&n| n as usize).sum();
        let edges = (0..count)
            .map(|order| {
                edge(
                    (rng.next() % node_count as u64) as usize,
                    order,
                    Reward {
                        evaluation: if case % 6 <= 1 { 0 } else { rng.next() as i32 },
                        attack: match order % 5 {
                            0 => -1,
                            1 => 0,
                            2 => i32::MAX,
                            _ => rng.next() as i32,
                        },
                    },
                )
            })
            .collect();
        compare(children, edges, &nodes);
    }
}

#[test]
fn the_first_sorted_tie_keeps_its_components_while_another_edge_supplies_spike() {
    let nodes = [
        Node::new(Value {
            value: [31, -11, 0, 0],
            spike: -20,
        }),
        Node::new(Value {
            value: [5, 15, 0, 0],
            spike: -30,
        }),
        Node::new(Value {
            value: [-40, 0, 0, 0],
            spike: 900,
        }),
        Node::new(Value {
            value: [19, 0, 0, 0],
            spike: i32::MAX,
        }),
    ];
    let edges = (0..63)
        .map(|order| {
            edge(
                order % 4,
                order,
                Reward {
                    evaluation: 0,
                    attack: if order % 4 == 3 { 1 } else { 0 },
                },
            )
        })
        .collect();
    let value = compare(
        Children {
            start: 49,
            lengths: [63, 0, 0, 0, 0, 0, 0],
            possibilities: 1,
        },
        edges,
        &nodes,
    )
    .unwrap();
    assert_eq!(value.score(), 20);
    assert_eq!(value.spike, 900);
    assert!(matches!(value.value, [31, -11, 0, 0] | [5, 15, 0, 0]));
}

#[test]
fn speculative_death_still_sorts_later_active_kinds_and_leaves_inactive_ranges_alone() {
    let nodes: Vec<_> = [2, 3, 1, 4, 0]
        .into_iter()
        .map(|score| {
            Node::new(Value {
                value: [score, 0, 0, 0],
                spike: 0,
            })
        })
        .collect();
    let edges = (0..5)
        .map(|order| edge(order, order, Reward::default()))
        .collect();
    assert!(
        compare(
            Children {
                start: 17,
                lengths: [0, 3, 2, 0, 0, 0, 0],
                possibilities: 3
            },
            edges,
            &nodes
        )
        .is_none()
    );
}

#[test]
fn component_averages_and_attack_spikes_keep_wrapping_and_negative_behavior() {
    let nodes = [
        Node::new(Value {
            value: [i32::MAX, i32::MIN, 9, -19],
            spike: i32::MAX,
        }),
        Node::new(Value {
            value: [i32::MAX, i32::MIN, -9, 19],
            spike: -800,
        }),
    ];
    let edges = vec![
        edge(
            0,
            0,
            Reward {
                evaluation: i32::MAX,
                attack: 1,
            },
        ),
        edge(
            1,
            1,
            Reward {
                evaluation: i32::MIN,
                attack: -1,
            },
        ),
    ];
    let value = compare(
        Children {
            start: 100,
            lengths: [1, 1, 0, 0, 0, 0, 0],
            possibilities: 3,
        },
        edges,
        &nodes,
    )
    .unwrap();
    assert_eq!(value.value[0], -1);
    assert_eq!(value.value[1], 0);
    assert_eq!(value.spike, 0);
}
