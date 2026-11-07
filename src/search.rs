//! Hikari's layered transposition graph.
//!
//! This follows `HikariAI2`, `SelectJob`, `ExpandJob`, and
//! `BackpropagateJob` in the C# reference. It is deliberately not an MCTS:
//! children are ranked by their backed-up integer evaluation, selection is a
//! weighted random walk, and a transposition has every incoming parent.

#[cfg(not(feature = "fast-registration"))]
use std::collections::HashMap;
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::board::{Board, LockResult, Piece, PieceKind};
use crate::eval::{Reward, Value, Weights, evaluate_board, evaluate_move};
#[cfg(feature = "fast-registration")]
use crate::graph_storage::{BoardLookup, board_hash};
use crate::movegen;

const NONE: u32 = u32::MAX;
const KINDS: [PieceKind; 7] = [
    PieceKind::I,
    PieceKind::O,
    PieceKind::T,
    PieceKind::J,
    PieceKind::L,
    PieceKind::S,
    PieceKind::Z,
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Maximum number of queue transitions beyond the root, including empty hold.
    pub max_depth: usize,
    pub use_hold: bool,
    pub use_speculation: bool,
    pub previews: usize,
    /// Independent random selections in a batch (Hikari's ParallelCount).
    pub batch_size: usize,
    /// Native expansion/evaluation workers, independent of batch_size.
    pub threads: usize,
    pub seed: u64,
    /// Retained unique nodes, including currently unreachable siblings.
    /// Reaching this cap stops expansion; committing a move frees the old layer.
    pub max_nodes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 32,
            use_hold: true,
            use_speculation: true,
            previews: 12,
            batch_size: 128,
            threads: 1,
            seed: 1,
            max_nodes: 2_000_000,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_depth < 2 {
            return Err("max_depth must allow at least two queue transitions".into());
        }
        if self.previews == 0 || self.batch_size == 0 || self.threads == 0 {
            return Err("previews, batch_size, and threads must be positive".into());
        }
        if self.max_nodes < 2 || self.max_nodes >= NONE as usize {
            return Err("max_nodes must be in 2..u32::MAX".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct SearchStats {
    /// Actual retained graph nodes, not random-walk visits or generated moves.
    pub nodes: usize,
    pub edges: usize,
    /// Greatest populated layer index; empty-hold transitions count as an edge.
    pub max_depth: usize,
    /// Leaf nodes expanded since start, including proven dead leaves.
    pub expanded: u64,
    /// Legal non-death candidate edges generated before transposition merging.
    pub generated: u64,
    pub transpositions: u64,
    pub batches: u64,
    pub memory_limited: bool,
    pub dead: bool,
    /// Number of nodes kept immediately after the most recent commit.
    pub retained_after_advance: usize,
}

/// Optional profiling counters, accumulated since `start`.
///
/// The five phase wall times are sequential, non-overlapping controller times.
/// Worker times instead sum individual tasks and may exceed wall time when
/// several threads run concurrently. CPU times exclude descheduling and are
/// zero on platforms without the optional thread CPU clock.
#[cfg(feature = "profile-search")]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ProfileStats {
    pub selection_wall_ns: u64,
    pub expansion_wall_ns: u64,
    pub registration_wall_ns: u64,
    pub evaluation_wall_ns: u64,
    pub backpropagation_wall_ns: u64,
    /// Controller thread only: excludes parallel selection workers.
    pub selection_cpu_ns: u64,
    pub registration_cpu_ns: u64,
    /// Controller thread only; covers the full evaluation stage with one worker.
    pub evaluation_cpu_ns: u64,
    pub backpropagation_cpu_ns: u64,
    pub expansion_worker_wall_ns: u64,
    pub movegen_worker_wall_ns: u64,
    pub expansion_worker_cpu_ns: u64,
    pub movegen_worker_cpu_ns: u64,
    pub expansion_count: u64,
    pub movegen_calls: u64,
    pub evaluated_boards: u64,
    pub batches: u64,
}

#[cfg(feature = "profile-search")]
#[derive(Default)]
struct ExpansionProfile {
    expansion_wall_ns: u64,
    expansion_cpu_ns: u64,
    movegen_wall_ns: u64,
    movegen_cpu_ns: u64,
    movegen_calls: u64,
}

#[cfg(feature = "profile-search")]
struct ProfileClock {
    wall: Instant,
    cpu: Option<u64>,
}

#[cfg(feature = "profile-search")]
impl ProfileClock {
    fn start() -> Self {
        Self {
            wall: Instant::now(),
            cpu: crate::profile_clock::thread_cpu_ns(),
        }
    }

    fn elapsed(&self) -> (u64, u64) {
        let cpu = match (self.cpu, crate::profile_clock::thread_cpu_ns()) {
            (Some(start), Some(end)) => end.saturating_sub(start),
            _ => 0,
        };
        (self.wall.elapsed().as_nanos() as u64, cpu)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ChosenMove {
    /// None is the original Hikari empty-hold-only transition.
    pub piece: Option<Piece>,
    pub hold: bool,
    pub value: Value,
    /// Mirai movement cost, including one unit for swapping an occupied hold.
    pub time: i32,
    #[serde(skip)]
    epoch: u64,
}

#[derive(Clone, Copy, Debug)]
struct Children {
    start: u32,
    lengths: [u16; 7],
    possibilities: u8,
}

impl Children {
    fn range(&self, kind: PieceKind) -> std::ops::Range<usize> {
        let index = kind as usize;
        let start = self.start as usize
            + self.lengths[..index]
                .iter()
                .map(|&n| n as usize)
                .sum::<usize>();
        start..start + self.lengths[index] as usize
    }

    fn known(&self) -> Option<PieceKind> {
        (self.possibilities.count_ones() == 1)
            .then(|| KINDS[self.possibilities.trailing_zeros() as usize])
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    node: u32,
    piece: Option<Piece>,
    reward: Reward,
    time: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Node {
    value: Value,
    children: Option<Children>,
    dead: bool,
    parent_head: u32,
}

impl Node {
    fn new(value: Value) -> Self {
        Self {
            value,
            children: None,
            dead: false,
            parent_head: NONE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ParentLink {
    parent: u32,
    next: u32,
}

#[derive(Default)]
struct Layer {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    #[cfg(not(feature = "fast-registration"))]
    lookup: HashMap<Board, u32>,
    #[cfg(feature = "fast-registration")]
    lookup: BoardLookup,
    parents: Vec<ParentLink>,
}

impl Layer {
    fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.lookup.clear();
        self.parents.clear();
    }

    /// Empty-hold children bypass deduplication, just like GraphGroup.AddNode:
    /// their next selection must have holding disabled.
    #[cfg(not(feature = "fast-registration"))]
    fn add_node(&mut self, key: Board, parent: u32, bypass: bool) -> (u32, bool) {
        let existing = if bypass {
            None
        } else {
            self.lookup.get(&key).copied()
        };
        let fresh = existing.is_none();
        let index = existing.unwrap_or_else(|| {
            let index = self.nodes.len() as u32;
            self.nodes.push(Node::new(Value::default()));
            if !bypass {
                self.lookup.insert(key, index);
            }
            index
        });
        let next = self.nodes[index as usize].parent_head;
        self.nodes[index as usize].parent_head = self.parents.len() as u32;
        self.parents.push(ParentLink { parent, next });
        (index, fresh)
    }

    #[cfg(all(feature = "fast-registration", test))]
    fn add_node(&mut self, key: Board, parent: u32, bypass: bool) -> (u32, bool) {
        self.add_node_hashed(key, board_hash(&key), parent, bypass)
    }

    #[cfg(feature = "fast-registration")]
    fn add_node_hashed(&mut self, key: Board, hash: u64, parent: u32, bypass: bool) -> (u32, bool) {
        let next_index = self.nodes.len() as u32;
        let (index, fresh) = if bypass {
            (next_index, true)
        } else {
            self.lookup.insert_if_absent_hashed(key, hash, next_index)
        };
        if fresh {
            self.nodes.push(Node::new(Value::default()));
        }
        let next = self.nodes[index as usize].parent_head;
        self.nodes[index as usize].parent_head = self.parents.len() as u32;
        self.parents.push(ParentLink { parent, next });
        (index, fresh)
    }
}

#[derive(Clone, Copy)]
struct Selected {
    depth: usize,
    node: u32,
    board: Board,
    can_hold: bool,
    candidates: u8,
}

struct PlannedEdge {
    // Intentionally before bag.Take(current): original ExpandJob keys and
    // evaluates this board. Selection reconstructs the consumed bag separately.
    board: Board,
    /// Computed by the expansion worker once, then reused by registration.
    #[cfg(feature = "fast-registration")]
    hash: u64,
    piece: Option<Piece>,
    reward: Reward,
    time: i32,
}

struct Expansion {
    selected: Selected,
    edges: Vec<PlannedEdge>,
    lengths: [u16; 7],
    dead: bool,
    #[cfg(feature = "profile-search")]
    profile: ExpansionProfile,
}

#[derive(Clone, Copy)]
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        // SplitMix64 accepts every seed, including zero.
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn fraction(&mut self) -> f32 {
        (self.next() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    fn piece(&mut self, possibilities: u8) -> PieceKind {
        let mut rank = (self.next() % u64::from(possibilities.count_ones())) as u32;
        for kind in KINDS {
            if possibilities & (1 << kind as u8) != 0 {
                if rank == 0 {
                    return kind;
                }
                rank -= 1;
            }
        }
        unreachable!("selection requires at least one possible piece")
    }
}

pub struct Search {
    config: Config,
    weights: Weights,
    board: Board,
    queue: Vec<PieceKind>,
    pending: VecDeque<PieceKind>,
    layers: VecDeque<Layer>,
    root: u32,
    rng: Random,
    pool: Option<rayon::ThreadPool>,
    counters: SearchStats,
    #[cfg(feature = "profile-search")]
    profile: ProfileStats,
    epoch: u64,
    started: bool,
    root_can_hold: bool,
    select_guidance: Vec<Option<Piece>>,
}

impl Search {
    pub fn new(config: Config, weights: Weights) -> Result<Self, String> {
        config.validate()?;
        let pool = if config.threads == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.threads)
                    .thread_name(|i| format!("hikari-search-{i}"))
                    .build()
                    .map_err(|error| error.to_string())?,
            )
        };
        let rng = Random::new(config.seed);
        Ok(Self {
            config,
            weights,
            board: Board::default(),
            queue: Vec::new(),
            pending: VecDeque::new(),
            layers: VecDeque::new(),
            root: 0,
            rng,
            pool,
            counters: SearchStats::default(),
            #[cfg(feature = "profile-search")]
            profile: ProfileStats::default(),
            epoch: 0,
            started: false,
            root_can_hold: true,
            select_guidance: Vec::new(),
        })
    }

    /// Explicitly replace the position. Ordinary committed moves use commit,
    /// which rotates one layer and preserves all the deeper layers.
    pub fn start(&mut self, board: Board, queue: Vec<PieceKind>) {
        self.board = board;
        self.queue.clear();
        self.pending = queue.into();
        self.layers.clear();
        self.layers.push_back(Layer::default());
        self.layers[0].nodes.push(Node::new(Value::default()));
        self.layers[0].lookup.insert(board, 0);
        self.root = 0;
        self.rng = Random::new(self.config.seed);
        self.counters = SearchStats::default();
        #[cfg(feature = "profile-search")]
        {
            self.profile = ProfileStats::default();
        }
        self.epoch = self.epoch.wrapping_add(1);
        self.started = true;
        self.root_can_hold = true;
        self.select_guidance.clear();
        self.refill_previews();
    }

    /// Apply an externally observed board change, retaining the known queue and
    /// bag. Pure added garbage translates the previous plan into selectGuidance,
    /// as in ResetInternal; other board edits discard the guidance.
    pub fn reset(&mut self, mut board: Board) {
        let previous_height = self.board.max_height();
        let next_height = board.max_height();
        let previous_plan = self.current_plan();
        self.select_guidance.clear();
        // ResetInternal preserves the AI's inferred bag rather than adopting
        // the observer's potentially stale randomizer state.
        board.bag = self.board.bag;
        // The reference checks the previous board here, before assigning the
        // newly observed board. Preserve this order; ordinary expansion then
        // determines whether the new board has a surviving placement.
        let dead = self
            .queue
            .first()
            .is_some_and(|&kind| self.board.spawn(kind).is_none());
        if !dead && next_height > previous_height {
            let difference = next_height - previous_height;
            let garbage = board.rows[..difference]
                .iter()
                .all(|row| row.count_ones() == 9)
                && (0..previous_height).all(|y| self.board.rows[y] == board.rows[y + difference]);
            if garbage {
                self.select_guidance = previous_plan
                    .into_iter()
                    .map(|piece| {
                        piece.map(|piece| Piece {
                            y: piece.y + difference as i8,
                            ..piece
                        })
                    })
                    .collect();
            }
        }
        self.board = board;
        if self.layers.is_empty() {
            self.layers.push_back(Layer::default());
        }
        for layer in &mut self.layers {
            layer.clear();
        }
        let mut root = Node::new(Value::default());
        root.dead = dead;
        self.layers[0].nodes.push(root);
        self.layers[0].lookup.insert(board, 0);
        self.root = 0;
        self.epoch = self.epoch.wrapping_add(1);
        self.counters.memory_limited = false;
        self.counters.retained_after_advance = 0;
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn queue(&self) -> &[PieceKind] {
        &self.queue
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Revealing a piece resolves the corresponding speculative layer and
    /// propagates its changed value through every transposition parent.
    pub fn add_next_piece(&mut self, kind: PieceKind) -> Result<(), String> {
        if !self.started {
            return Err("start must be called before adding pieces".into());
        }
        self.pending.push_back(kind);
        self.refill_previews();
        Ok(())
    }

    fn refill_previews(&mut self) {
        while self.queue.len() < self.config.previews {
            let Some(kind) = self.pending.pop_front() else {
                break;
            };
            let depth = self.queue.len();
            self.queue.push(kind);
            self.resolve(depth, kind);
        }
    }

    fn resolve(&mut self, depth: usize, kind: PieceKind) {
        if depth >= self.layers.len() {
            return;
        }
        let mut updates = vec![Vec::new(); self.layers.len()];
        let bit = 1 << kind as u8;
        for (index, node) in self.layers[depth].nodes.iter_mut().enumerate() {
            if let Some(children) = node.children.as_mut() {
                if children.possibilities & bit == 0 {
                    node.dead = true;
                } else {
                    children.possibilities = bit;
                }
                updates[depth].push(index as u32);
            }
        }
        self.propagate(updates);
    }

    /// A batch takes independent random walks through the current graph,
    /// deduplicates selected leaves, expands them, then backs up all parents.
    /// Returns the number of leaves actually expanded.
    pub fn search_batch(&mut self) -> usize {
        if !self.started
            || self.queue.is_empty()
            || self.counters.memory_limited
            || self.layers[0].nodes[self.root as usize].dead
        {
            return 0;
        }

        #[cfg(feature = "profile-search")]
        let selection_clock = ProfileClock::start();

        // Hikari's MaxDepth counts populated layers, including the root. Once
        // the rebuilt graph catches the old plan, the guidance has done its job.
        if !self.select_guidance.is_empty()
            && self.stats().max_depth + 1 >= self.select_guidance.len()
        {
            self.select_guidance.clear();
        }

        // Draw seeds serially, then run independent walks through one graph
        // snapshot, as SelectJob does. Indexed parallel collection keeps the
        // original order for deterministic SelectionValidate deduplication.
        let seeds: Vec<u64> = (0..self.config.batch_size)
            .map(|_| self.rng.next())
            .collect();
        let select = |(index, &seed): (usize, &u64)| self.select(&mut Random(seed), index == 0);
        let selected: Vec<Option<Selected>> = if let Some(pool) = &self.pool {
            pool.install(|| seeds.par_iter().enumerate().map(select).collect())
        } else {
            seeds.iter().enumerate().map(select).collect()
        };
        let mut selections = Vec::with_capacity(self.config.batch_size);
        let mut selected_keys = HashSet::with_capacity(self.config.batch_size);
        for selection in selected.into_iter().flatten() {
            if selected_keys.insert((selection.depth, selection.node)) {
                selections.push(selection);
            }
        }
        self.counters.batches += 1;
        #[cfg(feature = "profile-search")]
        {
            let (wall, cpu) = selection_clock.elapsed();
            self.profile.selection_wall_ns += wall;
            self.profile.selection_cpu_ns += cpu;
            self.profile.batches += 1;
        }
        if selections.is_empty() {
            return 0;
        }
        self.expand_selections(selections)
    }

    fn expand_selections(&mut self, selections: Vec<Selected>) -> usize {
        #[cfg(feature = "profile-search")]
        let expansion_clock = Instant::now();
        let weights = &self.weights;
        let use_hold = self.config.use_hold;
        let expand = |selected: &Selected| make_expansion(*selected, use_hold, weights);
        let expansions: Vec<Expansion> = if let Some(pool) = &self.pool {
            pool.install(|| selections.par_iter().map(expand).collect())
        } else {
            selections.iter().map(expand).collect()
        };
        #[cfg(feature = "profile-search")]
        {
            self.profile.expansion_wall_ns += expansion_clock.elapsed().as_nanos() as u64;
        }
        #[cfg(feature = "profile-search")]
        let registration_clock = ProfileClock::start();
        #[cfg(feature = "profile-search")]
        for expansion in &expansions {
            self.profile.expansion_worker_wall_ns += expansion.profile.expansion_wall_ns;
            self.profile.movegen_worker_wall_ns += expansion.profile.movegen_wall_ns;
            self.profile.expansion_worker_cpu_ns += expansion.profile.expansion_cpu_ns;
            self.profile.movegen_worker_cpu_ns += expansion.profile.movegen_cpu_ns;
            self.profile.movegen_calls += expansion.profile.movegen_calls;
            self.profile.expansion_count += 1;
        }
        self.counters.generated += expansions
            .iter()
            .map(|expansion| expansion.edges.len() as u64)
            .sum::<u64>();

        let required_layers = selections
            .iter()
            .map(|selected| selected.depth + 2)
            .max()
            .unwrap_or(1);
        while self.layers.len() < required_layers {
            self.layers.push_back(Layer::default());
        }
        let mut updates = vec![Vec::new(); self.layers.len()];
        let mut evaluations = Vec::new();
        let mut node_count = self
            .layers
            .iter()
            .map(|layer| layer.nodes.len())
            .sum::<usize>();
        let mut expanded = 0;
        #[cfg(feature = "fast-registration")]
        let mut capacity_keys = BoardLookup::default();

        for expansion in expansions {
            let selected = expansion.selected;
            if expansion.dead {
                self.layers[selected.depth].nodes[selected.node as usize].dead = true;
                updates[selected.depth].push(selected.node);
                expanded += 1;
                continue;
            }

            // Reserve a complete expansion. A cap never silently truncates a
            // move set, marks a viable position dead, or clears a subtree.
            #[cfg(not(feature = "fast-registration"))]
            let fits = {
                let mut fresh_keys = HashSet::new();
                let child_layer = &self.layers[selected.depth + 1];
                let needed = expansion
                    .edges
                    .iter()
                    .filter(|edge| {
                        edge.piece.is_none()
                            || (!child_layer.lookup.contains_key(&edge.board)
                                && fresh_keys.insert(edge.board))
                    })
                    .count();
                node_count + needed <= self.config.max_nodes
            };
            #[cfg(feature = "fast-registration")]
            let fits = if expansion.edges.len() <= self.config.max_nodes - node_count {
                // Every candidate adds at most one node. In the common case
                // this bound proves the whole expansion fits, without probing
                // the graph twice or allocating a per-expansion dedup set.
                true
            } else {
                // Near the cap retain the exact reservation rule, including
                // bypassed empty-hold nodes and duplicates within this batch.
                capacity_keys.clear();
                let child_layer = &self.layers[selected.depth + 1];
                let needed = expansion
                    .edges
                    .iter()
                    .filter(|edge| {
                        edge.piece.is_none()
                            || (child_layer
                                .lookup
                                .get_hashed(&edge.board, edge.hash)
                                .is_none()
                                && capacity_keys
                                    .insert_if_absent_hashed(edge.board, edge.hash, 0)
                                    .1)
                    })
                    .count();
                node_count + needed <= self.config.max_nodes
            };
            if !fits {
                self.counters.memory_limited = true;
                break;
            }

            let start = self.layers[selected.depth].edges.len() as u32;
            #[cfg(not(feature = "fast-registration"))]
            {
                let mut edges = Vec::with_capacity(expansion.edges.len());
                for planned in expansion.edges {
                    let child_layer = &mut self.layers[selected.depth + 1];
                    let (node, fresh) =
                        child_layer.add_node(planned.board, selected.node, planned.piece.is_none());
                    if fresh {
                        evaluations.push((selected.depth + 1, node, planned.board));
                        node_count += 1;
                    } else {
                        self.counters.transpositions += 1;
                    }
                    edges.push(Edge {
                        node,
                        piece: planned.piece,
                        reward: planned.reward,
                        time: planned.time,
                    });
                }
                self.layers[selected.depth].edges.extend(edges);
            }
            #[cfg(feature = "fast-registration")]
            {
                // The accepted move set can append directly to its parent
                // layer: each child and parent link is registered in the same
                // order, while avoiding a temporary edge allocation and copy.
                let layers = self.layers.make_contiguous();
                let (before, after) = layers.split_at_mut(selected.depth + 1);
                let current = &mut before[selected.depth];
                let child_layer = &mut after[0];
                current.edges.reserve(expansion.edges.len());
                for planned in expansion.edges {
                    let (node, fresh) = child_layer.add_node_hashed(
                        planned.board,
                        planned.hash,
                        selected.node,
                        planned.piece.is_none(),
                    );
                    if fresh {
                        evaluations.push((selected.depth + 1, node, planned.board));
                        node_count += 1;
                    } else {
                        self.counters.transpositions += 1;
                    }
                    current.edges.push(Edge {
                        node,
                        piece: planned.piece,
                        reward: planned.reward,
                        time: planned.time,
                    });
                }
            }
            self.layers[selected.depth].nodes[selected.node as usize].children = Some(Children {
                start,
                lengths: expansion.lengths,
                possibilities: selected.candidates,
            });
            updates[selected.depth].push(selected.node);
            expanded += 1;
        }

        #[cfg(feature = "profile-search")]
        {
            let (wall, cpu) = registration_clock.elapsed();
            self.profile.registration_wall_ns += wall;
            self.profile.registration_cpu_ns += cpu;
            self.profile.evaluated_boards += evaluations.len() as u64;
        }
        #[cfg(feature = "profile-search")]
        let evaluation_clock = ProfileClock::start();

        // The graph has already merged transpositions, so each newly inserted
        // board is evaluated once, even when many expansion parents found it.
        let evaluate = |&(depth, node, board): &(usize, u32, Board)| {
            (depth, node, evaluate_board(&board, &self.weights))
        };
        let values: Vec<_> = if let Some(pool) = &self.pool {
            pool.install(|| evaluations.par_iter().map(evaluate).collect())
        } else {
            evaluations.iter().map(evaluate).collect()
        };
        for (depth, node, value) in values {
            self.layers[depth].nodes[node as usize].value = value;
        }
        #[cfg(feature = "profile-search")]
        {
            let (wall, cpu) = evaluation_clock.elapsed();
            self.profile.evaluation_wall_ns += wall;
            self.profile.evaluation_cpu_ns += cpu;
        }
        #[cfg(feature = "profile-search")]
        let backpropagation_clock = ProfileClock::start();
        self.counters.expanded += expanded as u64;
        self.propagate(updates);
        #[cfg(feature = "profile-search")]
        {
            let (wall, cpu) = backpropagation_clock.elapsed();
            self.profile.backpropagation_wall_ns += wall;
            self.profile.backpropagation_cpu_ns += cpu;
        }
        expanded
    }

    pub fn think(&mut self, duration: Duration) {
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            if self.search_batch() == 0
                && (!self.started
                    || self.queue.is_empty()
                    || self.counters.memory_limited
                    || self.layers[0].nodes[self.root as usize].dead)
            {
                break;
            }
        }
    }

    fn select(&self, rng: &mut Random, guided: bool) -> Option<Selected> {
        let mut board = self.board;
        let mut index = self.root;
        let mut can_hold = self.root_can_hold;
        for depth in 0..self.layers.len() {
            let node = &self.layers[depth].nodes[index as usize];
            if node.dead {
                return None;
            }
            let Some(children) = node.children else {
                if depth >= self.config.max_depth {
                    return None;
                }
                let candidates = if let Some(&known) = self.queue.get(depth) {
                    1 << known as u8
                } else if self.config.use_speculation {
                    board.bag_candidates()
                } else {
                    return None;
                };
                return Some(Selected {
                    depth,
                    node: index,
                    board,
                    can_hold,
                    candidates,
                });
            };
            let known = children.known();
            let kind = known.unwrap_or_else(|| rng.piece(children.possibilities));
            let range = children.range(kind);
            let edges = &self.layers[depth].edges[range];
            let child_layer = &self.layers[depth + 1];
            // None itself is an empty-hold instruction, so distinguish missing
            // guidance from a guided hold-only transition.
            let picked = if guided && known.is_some() {
                self.select_guidance
                    .get(depth)
                    .and_then(|&target| edges.iter().position(|edge| edge.piece == target))
            } else {
                None
            }
            .unwrap_or_else(|| pick_ranked(edges, child_layer, rng));
            let child = edges.get(picked)?;
            // Keep original semantics: draw consumption happens on the path,
            // not in the dedup/evaluation key produced by ExpandJob.
            apply_transition(&mut board, kind, child.piece);
            index = child.node;
            can_hold = child.piece.is_some();
        }
        None
    }

    fn propagate(&mut self, mut updates: Vec<Vec<u32>>) {
        for depth in (0..updates.len()).rev() {
            updates[depth].sort_unstable();
            updates[depth].dedup();
            for offset in 0..updates[depth].len() {
                let index = updates[depth][offset];
                self.backup(depth, index);
                if depth > 0 {
                    let layer = &self.layers[depth];
                    let mut link = layer.nodes[index as usize].parent_head;
                    while link != NONE {
                        let parent = layer.parents[link as usize];
                        updates[depth - 1].push(parent.parent);
                        link = parent.next;
                    }
                }
            }
        }
    }

    fn backup(&mut self, depth: usize, index: u32) {
        let node = self.layers[depth].nodes[index as usize];
        if node.dead {
            return;
        }
        let Some(children) = node.children else {
            return;
        };
        let mut sum = [0_i32; 4];
        let mut spike = 0;
        let mut count = 0_i32;
        let mut dead = false;
        // VecDeque's backing storage may wrap after an advance. Using separate
        // slices here keeps the two mutable/immutable layer borrows disjoint.
        let layers = self.layers.make_contiguous();
        let (before, after) = layers.split_at_mut(depth + 1);
        let current = &mut before[depth];
        let child_layer = &after[0];
        for kind in KINDS {
            if children.possibilities & (1 << kind as u8) == 0 {
                continue;
            }
            let edges = &mut current.edges[children.range(kind)];
            edges.sort_unstable_by(|a, b| {
                let a_node = &child_layer.nodes[a.node as usize];
                let b_node = &child_layer.nodes[b.node as usize];
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
                let child = child_layer.nodes[edge.node as usize];
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
                // Hikari rejects a speculative node if any possible draw has
                // no surviving response, rather than averaging away death.
                dead = true;
            }
        }
        let node = &mut current.nodes[index as usize];
        if dead || count == 0 {
            node.dead = true;
        } else {
            node.value = Value {
                value: sum.map(|component| component / count),
                spike,
            };
        }
    }

    pub fn best_move(&self) -> Option<ChosenMove> {
        if !self.started {
            return None;
        }
        let &current = self.queue.first()?;
        let node = &self.layers[0].nodes[self.root as usize];
        if node.dead {
            return None;
        }
        let children = node.children?;
        if children.known()? != current {
            return None;
        }
        let edge = self.layers[0].edges.get(children.range(current).start)?;
        let child = &self.layers[1].nodes[edge.node as usize];
        if child.dead {
            return None;
        }
        Some(ChosenMove {
            piece: edge.piece,
            hold: edge.piece.is_none_or(|piece| piece.kind != current),
            value: child.value.with_reward(edge.reward),
            time: edge.time,
            epoch: self.epoch,
        })
    }

    /// Alive root moves in Hikari evaluation order. A protocol adapter can use
    /// this to choose a representable alternative when its wire format cannot
    /// express the best empty-hold transition.
    pub fn ranked_moves(&self) -> Vec<ChosenMove> {
        if !self.started {
            return Vec::new();
        }
        let Some(&current) = self.queue.first() else {
            return Vec::new();
        };
        let node = &self.layers[0].nodes[self.root as usize];
        if node.dead {
            return Vec::new();
        }
        let Some(children) = node.children else {
            return Vec::new();
        };
        if children.known() != Some(current) {
            return Vec::new();
        }
        self.layers[0].edges[children.range(current)]
            .iter()
            .filter_map(|edge| {
                let child = &self.layers[1].nodes[edge.node as usize];
                (!child.dead).then_some(ChosenMove {
                    piece: edge.piece,
                    hold: edge.piece.is_none_or(|piece| piece.kind != current),
                    value: child.value.with_reward(edge.reward),
                    time: edge.time,
                    epoch: self.epoch,
                })
            })
            .collect()
    }

    pub fn current_plan(&self) -> Vec<Option<Piece>> {
        let mut plan = Vec::new();
        if !self.started {
            return plan;
        }
        let mut index = self.root;
        for depth in 0..self.layers.len().saturating_sub(1) {
            let node = &self.layers[depth].nodes[index as usize];
            if node.dead {
                break;
            }
            let Some(children) = node.children else {
                break;
            };
            let Some(kind) = children.known() else {
                break;
            };
            let Some(edge) = self.layers[depth].edges.get(children.range(kind).start) else {
                break;
            };
            if self.layers[depth + 1].nodes[edge.node as usize].dead {
                break;
            }
            plan.push(edge.piece);
            index = edge.node;
        }
        plan
    }

    /// Confirm a previously returned choice. Continued pondering may change
    /// ranks without invalidating a choice for this same root position.
    pub fn advance(&mut self, chosen: &ChosenMove) -> Result<(), String> {
        if chosen.epoch != self.epoch {
            return Err("the chosen move belongs to an earlier root position".into());
        }
        self.commit(chosen.piece)
    }

    /// Confirm a complete physical placement, including its implicit empty
    /// hold. Both transitions are validated before changing board or queue.
    /// Failed validation may expand the graph, but cannot partly commit a hold.
    pub fn commit_physical(&mut self, piece: Piece) -> Result<(), String> {
        if !self.started {
            return Err("start must be called before committing a move".into());
        }
        let current = *self
            .queue
            .first()
            .ok_or("cannot commit without a known current piece")?;
        if piece.kind == current || self.board.hold.is_some() {
            return self.commit(Some(piece));
        }
        if !self.root_can_hold || self.queue.get(1) != Some(&piece.kind) {
            return Err("piece is not available after an empty hold".into());
        }
        if self.layers[0].nodes[self.root as usize].children.is_none() {
            self.search_batch();
        }
        let children = self.layers[0].nodes[self.root as usize]
            .children
            .ok_or("the root has not been expanded")?;
        let hold = self.layers[0].edges[children.range(current)]
            .iter()
            .find(|edge| edge.piece.is_none())
            .copied()
            .ok_or("the root has no empty-hold transition")?;
        if self.layers[1].nodes[hold.node as usize].children.is_none() {
            if self.counters.memory_limited {
                return Err("node limit prevents validating the empty-hold placement".into());
            }
            let mut board = self.board;
            apply_transition(&mut board, current, None);
            self.expand_selections(vec![Selected {
                depth: 1,
                node: hold.node,
                board,
                can_hold: false,
                candidates: 1 << piece.kind as u8,
            }]);
        }
        let held_children = self.layers[1].nodes[hold.node as usize]
            .children
            .ok_or("the empty-hold child could not be expanded")?;
        if held_children.possibilities & (1 << piece.kind as u8) == 0
            || !self.layers[1].edges[held_children.range(piece.kind)]
                .iter()
                .any(|edge| edge.piece == Some(piece))
        {
            return Err("the requested placement is absent after empty hold".into());
        }
        self.commit(None)?;
        self.commit(Some(piece))
    }

    /// Commit any retained legal root move, including an earlier suggestion.
    /// Only the consumed layer is cleared; descendants and transpositions keep
    /// their indices, evaluations, children, and multiple parent links.
    pub fn commit(&mut self, piece: Option<Piece>) -> Result<(), String> {
        if !self.started {
            return Err("start must be called before committing a move".into());
        }
        // A protocol client can confirm the second half of an empty-hold move
        // immediately. Expand that new root before looking up the placement.
        if self.layers[0].nodes[self.root as usize].children.is_none() {
            self.search_batch();
        }
        let &kind = self
            .queue
            .first()
            .ok_or("cannot commit without a known current piece")?;
        let children = self.layers[0].nodes[self.root as usize]
            .children
            .ok_or("the root has not been expanded")?;
        if children.possibilities & (1 << kind as u8) == 0 {
            return Err("the current piece is absent from the root speculation".into());
        }
        let child = self.layers[0].edges[children.range(kind)]
            .iter()
            .find(|child| child.piece == piece)
            .copied()
            .ok_or("the requested move is absent from the retained root children")?;

        apply_transition(&mut self.board, kind, piece);
        self.queue.remove(0);
        let mut old = self
            .layers
            .pop_front()
            .expect("started search has a root layer");
        old.clear();
        self.layers.push_back(old);
        self.root = child.node;
        self.root_can_hold = piece.is_some();
        self.epoch = self.epoch.wrapping_add(1);
        self.counters.memory_limited = false;
        self.counters.retained_after_advance =
            self.layers.iter().map(|layer| layer.nodes.len()).sum();
        self.refill_previews();
        Ok(())
    }

    pub fn stats(&self) -> SearchStats {
        let mut result = self.counters;
        result.nodes = 0;
        result.edges = 0;
        result.max_depth = 0;
        for (depth, layer) in self.layers.iter().enumerate() {
            result.nodes += layer.nodes.len();
            result.edges += layer.edges.len();
            if !layer.nodes.is_empty() {
                result.max_depth = depth;
            }
        }
        result.dead = self.started && self.layers[0].nodes[self.root as usize].dead;
        result
    }

    /// Cumulative optional instrumentation; absent from normal builds.
    #[cfg(feature = "profile-search")]
    pub fn profile(&self) -> ProfileStats {
        self.profile
    }
}

fn apply_transition(board: &mut Board, current: PieceKind, piece: Option<Piece>) {
    // Bag.Take returning false is intentionally ignored, as in SelectJob and
    // AdvanceGraph. The public queue is authoritative even for a mid-bag reset.
    board.take_from_bag(current);
    match piece {
        None => {
            debug_assert!(
                board.hold.is_none(),
                "empty hold-only transition needs an empty hold"
            );
            board.hold = Some(current);
        }
        Some(piece) => {
            if piece.kind != current {
                debug_assert_eq!(board.hold, Some(piece.kind));
                board.hold = Some(current);
            }
            board.lock(piece);
        }
    }
}

fn make_expansion(selected: Selected, use_hold: bool, weights: &Weights) -> Expansion {
    #[cfg(feature = "profile-search")]
    let clock = ProfileClock::start();
    let mut output = Expansion {
        selected,
        edges: Vec::new(),
        lengths: [0; 7],
        dead: false,
        #[cfg(feature = "profile-search")]
        profile: ExpansionProfile::default(),
    };
    // Mirai caches movements for both current and held piece in one instance.
    // Here each distinct kind is generated once and shared among draw cases.
    let mut moves: [Option<Vec<movegen::Placement>>; 7] = std::array::from_fn(|_| None);
    for kind in KINDS {
        if selected.candidates & (1 << kind as u8) == 0 {
            continue;
        }
        let start = output.edges.len();
        append_placements(
            &mut output.edges,
            selected.board,
            kind,
            false,
            weights,
            &mut moves,
            #[cfg(feature = "profile-search")]
            &mut output.profile,
        );
        if use_hold && selected.can_hold {
            if let Some(held) = selected.board.hold {
                if held != kind {
                    let mut after_hold = selected.board;
                    after_hold.hold = Some(kind);
                    append_placements(
                        &mut output.edges,
                        after_hold,
                        held,
                        true,
                        weights,
                        &mut moves,
                        #[cfg(feature = "profile-search")]
                        &mut output.profile,
                    );
                }
            } else {
                let mut board = selected.board;
                board.hold = Some(kind);
                let placement = LockResult::default();
                output.edges.push(PlannedEdge {
                    board,
                    #[cfg(feature = "fast-registration")]
                    hash: board_hash(&board),
                    piece: None,
                    reward: evaluate_move(
                        3,
                        None,
                        &board,
                        &placement,
                        selected.board.back_to_back,
                        weights,
                    ),
                    time: 3,
                });
            }
        }
        // SpeculationInfo stores a byte per kind. Keep the original 255-edge
        // limit even though native Rust ranges have room for larger move sets.
        output.edges.truncate((start + 255).min(output.edges.len()));
        let count = output.edges.len() - start;
        output.lengths[kind as usize] = count as u16;
        if count == 0 {
            output.dead = true;
            break;
        }
    }
    // Include cache destruction in expansion work, as it also happens in
    // ordinary builds before returning the expansion result to the controller.
    drop(moves);
    #[cfg(feature = "profile-search")]
    {
        let (wall, cpu) = clock.elapsed();
        output.profile.expansion_wall_ns = wall;
        output.profile.expansion_cpu_ns = cpu;
    }
    output
}

#[cfg_attr(feature = "profile-search", allow(clippy::too_many_arguments))]
fn append_placements(
    output: &mut Vec<PlannedEdge>,
    board: Board,
    kind: PieceKind,
    hold: bool,
    weights: &Weights,
    moves: &mut [Option<Vec<movegen::Placement>>; 7],
    #[cfg(feature = "profile-search")] profile: &mut ExpansionProfile,
) {
    let placements = moves[kind as usize].get_or_insert_with(|| {
        #[cfg(feature = "profile-search")]
        let clock = ProfileClock::start();
        let placements = movegen::generate(&board, kind);
        #[cfg(feature = "profile-search")]
        {
            let (wall, cpu) = clock.elapsed();
            profile.movegen_wall_ns += wall;
            profile.movegen_cpu_ns += cpu;
            profile.movegen_calls += 1;
        }
        placements
    });
    for placement in placements.iter() {
        let mut after = board;
        let result = after.lock(placement.piece);
        if result.death {
            continue;
        }
        let time = placement.time + i32::from(hold);
        output.push(PlannedEdge {
            board: after,
            #[cfg(feature = "fast-registration")]
            hash: board_hash(&after),
            piece: Some(placement.piece),
            reward: evaluate_move(
                time,
                Some(placement.piece),
                &after,
                &result,
                board.back_to_back,
                weights,
            ),
            time,
        });
    }
}

fn pick_ranked(edges: &[Edge], children: &Layer, rng: &mut Random) -> usize {
    let minimum = edges
        .iter()
        .map(|edge| {
            children.nodes[edge.node as usize]
                .value
                .with_reward(edge.reward)
                .score() as f32
        })
        .fold(f32::INFINITY, f32::min);
    let weight = |(rank, edge): (usize, &Edge)| {
        let child = &children.nodes[edge.node as usize];
        if child.dead {
            return 0.0;
        }
        let score = child.value.with_reward(edge.reward).score() as f32;
        let distance = score - minimum + 1.0;
        let rank = (rank + 1) as f32;
        distance * distance / (rank * rank) + 40.0
    };
    let sum: f32 = edges.iter().enumerate().map(weight).sum();
    if sum <= f32::EPSILON {
        return 0;
    }
    let random = rng.fraction() * sum;
    let mut cumulative = 0.0;
    for item in edges.iter().enumerate() {
        cumulative += weight(item);
        if cumulative > random {
            return item.0;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search() -> Search {
        Search::new(
            Config {
                batch_size: 32,
                max_nodes: 100_000,
                ..Config::default()
            },
            Weights::default(),
        )
        .unwrap()
    }

    fn value(score: i32, spike: i32) -> Value {
        Value {
            value: [score, 0, 0, 0],
            spike,
        }
    }

    fn edge(node: u32) -> Edge {
        Edge {
            node,
            piece: None,
            reward: Reward::default(),
            time: 0,
        }
    }

    #[cfg(feature = "profile-search")]
    #[test]
    fn profiling_accounts_for_completed_work_and_resets_on_start() {
        let mut search = search();
        let queue = vec![PieceKind::T, PieceKind::I, PieceKind::O];
        search.start(Board::default(), queue.clone());
        for _ in 0..8 {
            search.search_batch();
        }
        let profile = search.profile();
        let stats = search.stats();
        assert_eq!(profile.batches, stats.batches);
        assert_eq!(profile.expansion_count, stats.expanded);
        assert_eq!(profile.evaluated_boards as usize, stats.nodes - 1);
        assert!(profile.movegen_calls > 0);
        assert!(profile.movegen_worker_wall_ns > 0);
        assert!(profile.movegen_worker_wall_ns <= profile.expansion_worker_wall_ns);
        assert!(profile.movegen_worker_cpu_ns <= profile.expansion_worker_cpu_ns);
        assert!(profile.selection_wall_ns > 0);
        assert!(profile.expansion_wall_ns > 0);
        assert!(profile.registration_wall_ns > 0);
        assert!(profile.evaluation_wall_ns > 0);
        assert!(profile.backpropagation_wall_ns > 0);
        search.start(Board::default(), queue);
        assert_eq!(search.profile().batches, 0);
        assert_eq!(search.profile().movegen_calls, 0);
        assert_eq!(search.profile().movegen_worker_wall_ns, 0);
    }

    #[test]
    fn transposition_retains_and_updates_every_parent() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I]);
        search.layers[0].nodes.push(Node::new(Value::default()));
        search.layers.push_back(Layer::default());
        let (child, fresh) = search.layers[1].add_node(Board::default(), 0, false);
        assert!(fresh);
        let (same, fresh) = search.layers[1].add_node(Board::default(), 1, false);
        assert!(!fresh);
        assert_eq!(child, same);
        let children = Children {
            start: 0,
            lengths: [1, 0, 0, 0, 0, 0, 0],
            possibilities: 1,
        };
        search.layers[0].nodes[0].children = Some(children);
        search.layers[0].nodes[1].children = Some(Children {
            start: 1,
            ..children
        });
        search.layers[0].edges = vec![edge(child), edge(child)];
        search.layers[1].nodes[child as usize].value = value(712, 9);
        search.propagate(vec![Vec::new(), vec![child]]);
        assert_eq!(search.layers[0].nodes[0].value, value(712, 9));
        assert_eq!(search.layers[0].nodes[1].value, value(712, 9));
    }

    #[test]
    fn empty_hold_bypasses_lookup_without_replacing_a_shared_node() {
        let mut layer = Layer::default();
        let board = Board::default();
        let (shared, fresh) = layer.add_node(board, 11, false);
        assert!(fresh);
        let (held, fresh) = layer.add_node(board, 22, true);
        assert!(fresh);
        assert_ne!(shared, held);
        let (same, fresh) = layer.add_node(board, 33, false);
        assert!(!fresh);
        assert_eq!(same, shared);
        assert_eq!(layer.nodes.len(), 2);
        assert_eq!(layer.parents.len(), 3);
        assert_eq!(layer.lookup.get(&board), Some(&shared));

        let newest = layer.parents[layer.nodes[shared as usize].parent_head as usize];
        assert_eq!(newest.parent, 33);
        let oldest = layer.parents[newest.next as usize];
        assert_eq!(oldest.parent, 11);
        assert_eq!(oldest.next, NONE);
        let held_parent = layer.parents[layer.nodes[held as usize].parent_head as usize];
        assert_eq!(held_parent.parent, 22);
        assert_eq!(held_parent.next, NONE);
    }

    #[test]
    fn a_complete_duplicate_expansion_fits_at_the_node_cap() {
        let mut search = search();
        search.config.use_hold = false;
        search.start(Board::default(), vec![PieceKind::I]);
        let selected = Selected {
            depth: 0,
            node: 0,
            board: *search.board(),
            can_hold: true,
            candidates: 1 << PieceKind::I as u8,
        };
        let expansion = make_expansion(selected, false, &search.weights);
        let mut children = Layer::default();
        for planned in expansion.edges {
            children.add_node(planned.board, 0, false);
        }
        search.layers.push_back(children);
        let before = search.stats().nodes;
        search.config.max_nodes = before;

        assert_eq!(search.search_batch(), 1);
        assert_eq!(search.stats().nodes, before);
        assert!(!search.stats().memory_limited);
        assert!(search.layers[0].nodes[0].children.is_some());
        assert!(search.stats().transpositions > 0);
    }

    #[test]
    fn rejected_partial_duplicate_expansion_does_not_modify_the_graph() {
        let mut search = search();
        search.config.use_hold = false;
        search.start(Board::default(), vec![PieceKind::I]);
        let selected = Selected {
            depth: 0,
            node: 0,
            board: *search.board(),
            can_hold: true,
            candidates: 1 << PieceKind::I as u8,
        };
        let expansion = make_expansion(selected, false, &search.weights);
        let unique = expansion
            .edges
            .iter()
            .map(|planned| planned.board)
            .collect::<HashSet<_>>()
            .len();
        assert!(unique > 1);
        let mut children = Layer::default();
        children.add_node(expansion.edges[0].board, 0, false);
        search.layers.push_back(children);
        // Existing root + all unique children would require unique + 1 nodes.
        search.config.max_nodes = unique;
        let before = search.stats().nodes;
        let parent_count = search.layers[1].parents.len();
        let lookup_count = search.layers[1].lookup.keys().count();

        assert_eq!(search.search_batch(), 0);
        assert!(search.stats().memory_limited);
        assert_eq!(search.stats().nodes, before);
        assert_eq!(search.layers[1].parents.len(), parent_count);
        assert_eq!(search.layers[1].lookup.keys().count(), lookup_count);
        assert!(search.layers[0].edges.is_empty());
        assert!(search.layers[0].nodes[0].children.is_none());
        assert_eq!(search.stats().transpositions, 0);
    }

    #[test]
    fn unknown_draws_average_their_best_values_and_maximize_spike() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I]);
        search.layers.push_back(Layer::default());
        search.layers[1].nodes = vec![
            Node::new(value(100, 3)),
            Node::new(value(300, 11)),
            Node::new(value(50, 17)),
        ];
        search.layers[0].edges = vec![edge(0), edge(1), edge(2)];
        search.layers[0].nodes[0].children = Some(Children {
            start: 0,
            lengths: [1, 2, 0, 0, 0, 0, 0],
            possibilities: 0b11,
        });
        search.propagate(vec![vec![0], Vec::new()]);
        // Best I = 100, best O = 300; spike is separately max across O moves.
        assert_eq!(search.layers[0].nodes[0].value, value(200, 17));
        search.resolve(0, PieceKind::O);
        assert_eq!(search.layers[0].nodes[0].value, value(300, 17));
        assert_eq!(
            search.layers[0].nodes[0].children.unwrap().known(),
            Some(PieceKind::O)
        );
    }

    #[test]
    fn any_unavoidable_speculative_death_marks_the_node_dead() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I]);
        search.layers.push_back(Layer::default());
        search.layers[1].nodes = vec![Node::new(value(100, 3)), Node::new(value(300, 11))];
        search.layers[1].nodes[1].dead = true;
        search.layers[0].edges = vec![edge(0), edge(1)];
        search.layers[0].nodes[0].children = Some(Children {
            start: 0,
            lengths: [1, 1, 0, 0, 0, 0, 0],
            possibilities: 0b11,
        });
        search.propagate(vec![vec![0], Vec::new()]);
        assert!(search.layers[0].nodes[0].dead);
    }

    #[test]
    fn advancing_preserves_deeper_nodes_and_the_chosen_child_index() {
        let mut search = search();
        search.start(
            Board::default(),
            vec![PieceKind::I, PieceKind::O, PieceKind::T, PieceKind::J],
        );
        for _ in 0..16 {
            search.search_batch();
        }
        let chosen = search.best_move().expect("empty board has a move");
        let children = search.layers[0].nodes[search.root as usize]
            .children
            .unwrap();
        let original = search.layers[0].edges[children.range(PieceKind::I)]
            .iter()
            .find(|edge| edge.piece == chosen.piece)
            .unwrap()
            .node;
        let descendants = search
            .layers
            .iter()
            .skip(1)
            .map(|layer| layer.nodes.len())
            .collect::<Vec<_>>();
        let expected = descendants.iter().sum::<usize>();
        let original_value = search.layers[1].nodes[original as usize].value;
        search.advance(&chosen).unwrap();
        assert_eq!(search.root, original);
        assert_eq!(
            search.layers[0].nodes[search.root as usize].value,
            original_value
        );
        assert_eq!(search.stats().nodes, expected);
        assert_eq!(search.stats().retained_after_advance, expected);
        for (depth, count) in descendants.into_iter().enumerate() {
            assert_eq!(search.layers[depth].nodes.len(), count);
        }
        assert_eq!(search.queue()[0], PieceKind::O);
        assert!(search.advance(&chosen).is_err());
    }

    #[test]
    fn committing_a_previous_suggestion_does_not_require_it_to_stay_first() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I, PieceKind::O]);
        search.search_batch();
        let suggested = search.best_move().unwrap();
        for _ in 0..4 {
            search.search_batch();
        }
        search.commit(suggested.piece).unwrap();
        assert_eq!(search.queue(), &[PieceKind::O]);
    }

    #[test]
    fn hold_only_is_a_separate_transition_and_cannot_hold_twice() {
        let mut search = search();
        search.start(
            Board::default(),
            vec![PieceKind::I, PieceKind::O, PieceKind::T],
        );
        search.search_batch();
        search.commit(None).unwrap();
        assert_eq!(search.board().hold, Some(PieceKind::I));
        assert_eq!(search.board().rows, [0; 24]);
        assert_eq!(search.queue()[0], PieceKind::O);
        search.search_batch();
        let root = search.layers[0].nodes[search.root as usize];
        let children = root.children.unwrap();
        let options = &search.layers[0].edges[children.range(PieceKind::O)];
        assert!(
            options
                .iter()
                .all(|edge| edge.piece.is_some_and(|piece| piece.kind == PieceKind::O))
        );
    }

    #[test]
    fn node_limit_leaves_the_unexpanded_root_viable_and_never_clears_it() {
        let mut search = Search::new(
            Config {
                max_nodes: 2,
                ..Config::default()
            },
            Weights::default(),
        )
        .unwrap();
        search.start(Board::default(), vec![PieceKind::I]);
        assert_eq!(search.search_batch(), 0);
        assert!(search.stats().memory_limited);
        assert!(!search.stats().dead);
        assert_eq!(search.stats().nodes, 1);
        assert!(search.layers[0].nodes[0].children.is_none());
    }

    #[test]
    fn seeded_serial_and_parallel_batches_preserve_search_semantics() {
        let mut serial = search();
        let mut parallel = Search::new(
            Config {
                threads: 2,
                ..serial.config.clone()
            },
            Weights::default(),
        )
        .unwrap();
        let queue = vec![PieceKind::I, PieceKind::O, PieceKind::T, PieceKind::J];
        serial.start(Board::default(), queue.clone());
        parallel.start(Board::default(), queue);
        for _ in 0..8 {
            assert_eq!(serial.search_batch(), parallel.search_batch());
        }
        assert_eq!(serial.stats().nodes, parallel.stats().nodes);
        assert_eq!(
            serial.stats().transpositions,
            parallel.stats().transpositions
        );
        let a = serial.best_move().unwrap();
        let b = parallel.best_move().unwrap();
        assert_eq!(a.piece, b.piece);
        assert_eq!(a.value, b.value);
    }

    #[test]
    fn expansion_key_keeps_pre_draw_bag_but_commit_consumes_the_draw() {
        let mut search = search();
        let board = Board {
            bag: 0x7f,
            ..Board::default()
        };
        search.start(board, vec![PieceKind::I, PieceKind::O]);
        search.search_batch();
        assert!(search.layers[1].lookup.keys().all(|key| key.bag == 0x7f));
        let chosen = search.best_move().unwrap();
        search.advance(&chosen).unwrap();
        assert_eq!(search.board().bag & 1, 0);
    }

    #[test]
    fn revealing_a_preview_resolves_existing_layers_without_throwing_them_away() {
        let mut search = Search::new(
            Config {
                use_hold: false,
                ..search().config.clone()
            },
            Weights::default(),
        )
        .unwrap();
        search.start(Board::default(), vec![PieceKind::I]);
        for _ in 0..6 {
            search.search_batch();
        }
        assert!(search.layers[1].nodes.iter().any(|node| {
            node.children
                .is_some_and(|children| children.possibilities.count_ones() > 1)
        }));
        let before = search.stats().nodes;
        search.add_next_piece(PieceKind::O).unwrap();
        assert_eq!(search.stats().nodes, before);
        for node in &search.layers[1].nodes {
            if let Some(children) = node.children {
                assert_eq!(children.known(), Some(PieceKind::O));
            }
        }
        assert_eq!(search.queue(), &[PieceKind::I, PieceKind::O]);
    }

    #[test]
    fn empty_hold_and_placement_can_commit_without_intervening_ponder() {
        let mut search = search();
        search.start(
            Board::default(),
            vec![PieceKind::I, PieceKind::O, PieceKind::T],
        );
        search.search_batch();
        let placement = movegen::generate(search.board(), PieceKind::O)[0].piece;
        search.commit(None).unwrap();
        // commit lazily expands the unexpanded hold-only root.
        search.commit(Some(placement)).unwrap();
        assert_eq!(search.queue(), &[PieceKind::T]);
        assert_eq!(search.board().hold, Some(PieceKind::I));
        assert!(search.board().rows.iter().any(|&row| row != 0));
    }

    #[test]
    fn ranked_root_moves_begin_with_the_best_and_keep_descending_scores() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I, PieceKind::O]);
        search.search_batch();
        let best = search.best_move().unwrap();
        let ranked = search.ranked_moves();
        assert_eq!(ranked[0].piece, best.piece);
        assert_eq!(ranked[0].value, best.value);
        assert!(
            ranked
                .windows(2)
                .all(|pair| pair[0].value.score() >= pair[1].value.score())
        );
        assert!(ranked.iter().any(|chosen| chosen.piece.is_some()));
    }

    #[test]
    fn physical_empty_hold_validation_is_atomic_on_an_invalid_placement() {
        let mut search = search();
        search.start(
            Board::default(),
            vec![PieceKind::I, PieceKind::O, PieceKind::T],
        );
        search.search_batch();
        let before = *search.board();
        let queue = search.queue().to_vec();
        let invalid = Piece::new(PieceKind::O, -3, -3, 0);
        assert!(search.commit_physical(invalid).is_err());
        assert_eq!(*search.board(), before);
        assert_eq!(search.queue(), queue);
        let placement = movegen::generate(search.board(), PieceKind::O)[0].piece;
        search.commit_physical(placement).unwrap();
        assert_eq!(search.queue(), &[PieceKind::T]);
        assert_eq!(search.board().hold, Some(PieceKind::I));
    }

    #[test]
    fn reset_preserves_queue_bag_rng_and_counters_and_shifts_a_garbage_plan() {
        let mut search = Search::new(
            Config {
                use_hold: false,
                ..search().config.clone()
            },
            Weights::default(),
        )
        .unwrap();
        let mut board = Board {
            bag: 0x7f,
            ..Board::default()
        };
        board.rows[0] = 0b0000000011;
        board.rows[1] = 0b0000000001;
        search.start(
            board,
            vec![
                PieceKind::I,
                PieceKind::O,
                PieceKind::T,
                PieceKind::J,
                PieceKind::L,
            ],
        );
        for _ in 0..8 {
            search.search_batch();
        }
        let old_plan = search.current_plan();
        assert!(!old_plan.is_empty());
        let queue = search.queue().to_vec();
        let batches = search.stats().batches;
        let expanded = search.stats().expanded;
        let rng = search.rng.0;
        let mut raised = board;
        raised.rows = [0; 24];
        raised.rows[0] = 0b1111111110;
        raised.rows[1] = 0b1111111101;
        for y in 0..board.max_height() {
            raised.rows[y + 2] = board.rows[y];
        }
        raised.bag = 0b10; // Observer state must not replace the inferred bag.
        search.reset(raised);
        assert_eq!(search.queue(), queue);
        assert_eq!(search.board().bag, 0x7f);
        assert_eq!(search.rng.0, rng);
        assert_eq!(search.stats().batches, batches);
        assert_eq!(search.stats().expanded, expanded);
        assert_eq!(search.stats().nodes, 1);
        assert_eq!(search.stats().max_depth, 0);
        let expected: Vec<_> = old_plan
            .into_iter()
            .map(|piece| piece.map(|piece| piece.shifted(0, 2)))
            .collect();
        assert_eq!(search.select_guidance, expected);
    }

    #[test]
    fn reset_discards_guidance_for_a_non_garbage_board_edit() {
        let mut search = search();
        search.start(
            Board::default(),
            vec![PieceKind::I, PieceKind::O, PieceKind::T],
        );
        for _ in 0..3 {
            search.search_batch();
        }
        assert!(!search.current_plan().is_empty());
        let mut board = *search.board();
        board.rows[0] = 0b1111111100; // Eight occupied cells is not garbage.
        board.rows[1] = 0b1111111110;
        search.reset(board);
        assert!(search.select_guidance.is_empty());
        assert_eq!(search.stats().nodes, 1);
    }

    #[test]
    fn first_walk_follows_a_known_guided_child_and_other_walks_remain_random() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I, PieceKind::O]);
        search.search_batch();
        let children = search.layers[0].nodes[0].children.unwrap();
        let target = *search.layers[0].edges[children.range(PieceKind::I)]
            .last()
            .unwrap();
        search.select_guidance = vec![target.piece];
        for seed in 0..16 {
            let selected = search.select(&mut Random::new(seed), true).unwrap();
            assert_eq!(selected.depth, 1);
            assert_eq!(selected.node, target.node);
        }
        let random_nodes = (0..32)
            .map(|seed| search.select(&mut Random::new(seed), false).unwrap().node)
            .collect::<HashSet<_>>();
        assert!(random_nodes.len() > 1);
        // Missing guidance falls back to the normal ranked roulette.
        search.select_guidance = vec![Some(Piece::new(PieceKind::I, -3, -3, 0))];
        assert!(search.select(&mut Random::new(17), true).is_some());
    }

    #[test]
    fn guidance_clears_when_populated_layer_count_catches_the_previous_plan() {
        let mut search = search();
        search.start(Board::default(), vec![PieceKind::I, PieceKind::O]);
        search.search_batch();
        assert_eq!(search.stats().max_depth, 1);
        search.select_guidance = vec![None, None];
        search.search_batch();
        assert!(search.select_guidance.is_empty());
    }
}
