use super::*;
use serde_json::{Value as JsonValue, json};

fn search(threads: usize, max_nodes: usize) -> Search {
    Search::new(
        Config {
            batch_size: 32,
            max_depth: 8,
            max_nodes,
            previews: 4,
            use_hold: false,
            ..Config::for_threads(threads)
        },
        Weights::default(),
    )
    .unwrap()
}

fn queue() -> Vec<PieceKind> {
    vec![PieceKind::T, PieceKind::I, PieceKind::O, PieceKind::J]
}

fn selected(search: &Search) -> Selected {
    search.select(&mut Random::new(7121), false).unwrap()
}

fn generation(search: &Search) -> u64 {
    search.stream.generation.load(Ordering::Relaxed)
}

fn inject(search: &mut Search, generation: u64, expansion: Expansion) {
    let key = (expansion.selected.depth, expansion.selected.node);
    // Queue the completion first: even a failing assertion can safely drop
    // Search, whose pause must receive a completion for every physical credit.
    assert!(
        search
            .stream
            .sender
            .send(Completion {
                generation,
                key,
                expansion: Some(expansion),
            })
            .is_ok()
    );
    search.stream.reserved.insert(key);
    search.stream.outstanding += 1;
}

fn graph(search: &Search) -> JsonValue {
    let layers: Vec<_> = search
        .layers
        .iter()
        .map(|layer| {
            let nodes: Vec<_> = layer
                .nodes
                .iter()
                .map(|node| {
                    (
                        node.value,
                        node.dead,
                        node.parent_head,
                        node.children.map(|c| (c.start, c.lengths, c.possibilities)),
                    )
                })
                .collect();
            let edges: Vec<_> = layer
                .edges
                .iter()
                .map(|edge| (edge.node, edge.piece, edge.reward, edge.time))
                .collect();
            let parents: Vec<_> = layer
                .parents
                .iter()
                .map(|link| (link.parent, link.next))
                .collect();
            json!({"nodes": nodes, "edges": edges, "parents": parents})
        })
        .collect();
    json!({
        "board": search.board(), "queue": search.queue(), "stats": search.stats(),
        "best": search.best_move(), "plan": search.current_plan(), "layers": layers,
    })
}

fn assert_parent_links(search: &Search) {
    assert!(search.stats().nodes <= search.config.max_nodes);
    for depth in 0..search.layers.len().saturating_sub(1) {
        let current = &search.layers[depth];
        let next = &search.layers[depth + 1];
        for (parent, node) in current.nodes.iter().enumerate() {
            let Some(children) = node.children else {
                continue;
            };
            let length: usize = children.lengths.iter().map(|&n| n as usize).sum();
            let edges = &current.edges[children.start as usize..children.start as usize + length];
            for edge in edges {
                let child = &next.nodes[edge.node as usize];
                let mut link = child.parent_head;
                let mut found = false;
                let mut remaining = next.parents.len() + 1;
                while link != NONE {
                    assert!(remaining > 0, "parent links contain a cycle");
                    remaining -= 1;
                    let entry = next.parents[link as usize];
                    found |= entry.parent as usize == parent;
                    link = entry.next;
                }
                assert!(found, "missing parent at depth {depth}, node {parent}");
            }
        }
    }
}

#[test]
fn old_generation_completion_cannot_release_a_new_reservation() {
    let mut stream = Stream::new(Weights::default());
    let key = (2, 7);
    let old_generation = stream.generation.load(Ordering::Relaxed);
    stream.reserved.insert(key);
    stream.outstanding = 2;
    stream.cancel();
    stream.reserved.insert(key);

    assert!(
        stream
            .take(Completion {
                generation: old_generation,
                key,
                expansion: None,
            })
            .is_none()
    );
    assert_eq!(stream.outstanding, 1);
    assert!(stream.reserved.contains(&key));
    assert!(
        stream
            .take(Completion {
                generation: stream.generation.load(Ordering::Relaxed),
                key,
                expansion: None,
            })
            .is_none()
    );
    assert_eq!(stream.outstanding, 0);
    assert!(stream.reserved.is_empty());
}

#[test]
fn start_and_reset_discard_results_even_when_root_ids_are_reused() {
    for restart in [false, true] {
        let mut search = search(2, 50_000);
        search.start(Board::default(), queue());
        let old_generation = generation(&search);
        let expansion = prepare(
            selected(&search),
            false,
            &search.weights,
            search.cavity_mode,
        );
        inject(&mut search, old_generation, expansion);

        let mut replacement = Board::default();
        replacement.rows[0] = 0x3ff ^ (1 << 4);
        if restart {
            search.start(replacement, queue());
        } else {
            search.reset(replacement);
        }
        assert_ne!(generation(&search), old_generation);
        assert_eq!(search.stream.outstanding, 1);
        search.stream.reserved.insert((0, 0));
        let before = graph(&search);

        assert!(search.receive_ready(None).is_empty());
        assert_eq!(search.stream.outstanding, 0);
        assert!(search.stream.reserved.contains(&(0, 0)));
        assert_eq!(graph(&search), before);
        assert!(search.layers[0].nodes[0].children.is_none());
    }
}

#[test]
fn preview_revelation_discards_speculative_work_without_invalidating_chosen_moves() {
    let mut search = search(2, 50_000);
    search.start(Board::default(), vec![PieceKind::T]);
    assert_eq!(search.search_batch(), 1);
    let old_epoch = search.epoch;
    let old_generation = generation(&search);
    let selection = selected(&search);
    assert_eq!(selection.depth, 1);
    assert!(selection.candidates.count_ones() > 1);
    let key = (selection.depth, selection.node);
    let expansion = prepare(selection, false, &search.weights, search.cavity_mode);
    inject(&mut search, old_generation, expansion);

    search.add_next_piece(PieceKind::I).unwrap();
    assert_eq!(search.epoch, old_epoch);
    assert_ne!(generation(&search), old_generation);
    search.stream.reserved.insert(key);
    let before = graph(&search);
    assert!(search.receive_ready(None).is_empty());
    assert_eq!(graph(&search), before);
    assert!(search.stream.reserved.contains(&key));
    assert_eq!(selected(&search).candidates, 1 << PieceKind::I as u8);
}

#[test]
fn commit_discards_pending_work_from_the_previous_layer_numbers() {
    let mut search = search(2, 50_000);
    search.start(Board::default(), queue());
    assert_eq!(search.search_batch(), 1);
    let chosen = search.best_move().unwrap();
    let old_generation = generation(&search);
    let expansion = prepare(
        selected(&search),
        false,
        &search.weights,
        search.cavity_mode,
    );
    inject(&mut search, old_generation, expansion);

    search.advance(&chosen).unwrap();
    assert_ne!(generation(&search), old_generation);
    let before = graph(&search);
    assert!(search.receive_ready(None).is_empty());
    assert_eq!(search.stream.outstanding, 0);
    assert_eq!(graph(&search), before);
    assert_parent_links(&search);
}

#[test]
fn repeated_resets_preserve_physical_credits_and_bound_the_queue() {
    for threads in [2, 4] {
        let mut search = search(threads, 50_000);
        search.start(Board::default(), queue());
        search.dispatch();
        assert_eq!(search.stream.outstanding, 1);
        let limit = threads * 2;
        for _ in 0..limit + 4 {
            let outstanding = search.stream.outstanding;
            search.reset(Board::default());
            assert_eq!(search.stream.outstanding, outstanding);
            assert!(search.stream.reserved.is_empty());
            search.dispatch();
            assert!(search.stream.outstanding <= limit);
        }
        assert_eq!(search.stream.outstanding, limit);
        let before = graph(&search);
        search.pause();
        assert_eq!(search.stream.outstanding, 0);
        assert!(search.stream.reserved.is_empty());
        assert_eq!(graph(&search), before);
    }
}

#[test]
fn pause_drains_real_workers_without_publishing_more_results_and_can_resume() {
    for threads in [2, 4] {
        let mut search = search(threads, 50_000);
        search.start(Board::default(), queue());
        for _ in 0..6 {
            assert!(search.search_step() > 0);
            assert!(search.stream.outstanding <= threads * 2);
        }
        assert!(search.stream.outstanding > 0);
        let before = graph(&search);
        search.pause();
        assert_eq!(search.stream.outstanding, 0);
        assert!(search.stream.reserved.is_empty());
        assert_eq!(graph(&search), before);
        assert!(search.receive_ready(None).is_empty());
        search.pause();
        assert_eq!(graph(&search), before);
        assert!(search.search_step() > 0);
        search.pause();
        assert_parent_links(&search);
        assert_eq!(search.stats().batches, 0);
    }
}

#[test]
fn node_cap_rejects_an_entire_expansion_and_never_publishes_partial_children() {
    for threads in [2, 4] {
        let mut tiny = search(threads, 2);
        tiny.start(Board::default(), queue());
        assert_eq!(tiny.search_step(), 0);
        tiny.pause();
        assert_eq!(tiny.stats().nodes, 1);
        assert_eq!(tiny.stats().edges, 0);
        assert!(tiny.stats().memory_limited);
        assert!(!tiny.stats().dead);
        assert!(tiny.layers[0].nodes[0].children.is_none());

        let mut search = search(threads, 50_000);
        search.start(Board::default(), queue());
        let expansion = prepare(
            selected(&search),
            false,
            &search.weights,
            search.cavity_mode,
        );
        assert!(!expansion.dead);
        let expected_edges = expansion.edges.len();
        let fresh: HashSet<_> = expansion.edges.iter().map(|edge| edge.board).collect();
        search.config.max_nodes = fresh.len() + 1;
        assert_eq!(search.search_step(), 1);
        assert_eq!(search.stats().nodes, search.config.max_nodes);
        assert_eq!(search.stats().edges, expected_edges);
        assert!(search.layers[0].nodes[0].children.is_some());

        assert_eq!(search.search_step(), 0);
        search.pause();
        assert!(search.stats().memory_limited);
        assert!(!search.stats().dead);
        assert_eq!(search.stats().nodes, fresh.len() + 1);
        assert_eq!(search.stats().edges, expected_edges);
        assert!(
            search.layers[1]
                .nodes
                .iter()
                .all(|node| node.children.is_none())
        );
        assert_parent_links(&search);
    }
}

#[test]
fn reference_batch_cancels_stream_work_before_expanding_the_same_leaf() {
    let mut reference = search(2, 50_000);
    let mut actual = search(2, 50_000);
    reference.start(Board::default(), queue());
    actual.start(Board::default(), queue());
    let old_generation = generation(&actual);
    let expansion = prepare(
        selected(&actual),
        false,
        &actual.weights,
        actual.cavity_mode,
    );
    inject(&mut actual, old_generation, expansion);

    assert_eq!(actual.search_batch(), reference.search_batch());
    assert_eq!(actual.stream.outstanding, 0);
    assert!(actual.stream.reserved.is_empty());
    assert_eq!(graph(&actual), graph(&reference));
    assert_eq!(actual.stats().batches, 1);

    assert!(actual.search_step() > 0);
    assert!(actual.stream.outstanding > 0);
    actual.search_batch();
    assert_eq!(actual.stream.outstanding, 0);
    assert!(actual.stream.reserved.is_empty());
    assert_eq!(actual.stats().batches, 2);
    assert_parent_links(&actual);
}

#[test]
fn published_children_already_have_their_board_evaluations() {
    for threads in [1, 2, 4] {
        let mut search = search(threads, 50_000);
        search.start(Board::default(), queue());
        let expansion = prepare(
            selected(&search),
            false,
            &search.weights,
            search.cavity_mode,
        );
        assert!(!expansion.dead);
        let expected: Vec<_> = expansion
            .edges
            .iter()
            .map(|edge| (edge.board, evaluate_board(&edge.board, &search.weights)))
            .collect();
        let before = graph(&search);
        search.dispatch();
        assert_eq!(graph(&search), before);
        assert!(search.best_move().is_none());

        assert_eq!(search.search_step(), 1);
        for (board, expected_value) in expected {
            let node = *search.layers[1].lookup.get(&board).unwrap();
            assert_eq!(search.layers[1].nodes[node as usize].value, expected_value);
            assert!(search.layers[1].nodes[node as usize].children.is_none());
        }
        assert!(search.best_move().is_some());
        assert_eq!(search.stats().batches, 0);
        search.pause();
        assert_parent_links(&search);
    }
}

#[test]
fn a_late_result_for_an_expanded_node_cannot_append_its_edges_twice() {
    let mut search = search(2, 50_000);
    search.start(Board::default(), queue());
    let selection = selected(&search);
    assert_eq!(search.search_step(), 1);
    search.pause();
    let current_generation = generation(&search);
    let expansion = prepare(selection, false, &search.weights, search.cavity_mode);
    inject(&mut search, current_generation, expansion);
    let before = graph(&search);

    assert!(search.receive_ready(None).is_empty());
    assert_eq!(search.stream.outstanding, 0);
    assert!(search.stream.reserved.is_empty());
    assert_eq!(graph(&search), before);
}
