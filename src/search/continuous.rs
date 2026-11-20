//! Owned expansion jobs keep running while the controller publishes results.
//! Workers never borrow the graph, so registration can grow or rotate layers.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

const DISPATCH_GROUP_SIZE: usize = 16;

pub(super) struct Completion {
    pub(super) generation: u64,
    pub(super) key: (usize, u32),
    pub(super) expansion: Option<Expansion>,
}

pub(super) struct Stream {
    pub(super) generation: Arc<AtomicU64>,
    pub(super) sender: mpsc::Sender<Completion>,
    pub(super) receiver: Mutex<mpsc::Receiver<Completion>>,
    pub(super) reserved: HashSet<(usize, u32)>,
    pub(super) outstanding: usize,
    weights: Arc<Weights>,
    pub(super) dirty: dirty::DirtyScratch,
}

impl Stream {
    pub(super) fn new(weights: Weights) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            generation: Arc::new(AtomicU64::new(0)),
            sender,
            receiver: Mutex::new(receiver),
            reserved: HashSet::new(),
            outstanding: 0,
            weights: Arc::new(weights),
            dirty: dirty::DirtyScratch::default(),
        }
    }

    fn cancel(&mut self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.reserved.clear();
        // Outstanding credits include old generations until their completion
        // arrives. Repeated resets therefore cannot grow the work queue.
    }

    fn take(&mut self, completed: Completion) -> Option<Expansion> {
        debug_assert!(self.outstanding > 0);
        self.outstanding -= 1;
        if completed.generation != self.generation.load(Ordering::Relaxed) {
            return None;
        }
        if completed.expansion.is_none() {
            self.reserved.remove(&completed.key);
        }
        // Keep a completed leaf reserved until it has actually been registered.
        // Other leaves can be dispatched while these results are published.
        completed.expansion
    }
}

pub(super) fn prepare(
    selected: Selected,
    use_hold: bool,
    weights: &Weights,
    cavity_mode: CavityMode,
) -> Expansion {
    let mut held_copies = Vec::new();
    let mut expansion =
        make_expansion_with_held_copies(selected, use_hold, weights, |first, start, count| {
            held_copies.push((first, start, count));
        });
    #[cfg(feature = "profile-search")]
    let clock = ProfileClock::start();
    if !expansion.dead {
        let evaluate =
            |edge: &PlannedEdge| evaluate_board_with_cavities(&edge.board, weights, cavity_mode);
        let edges = &expansion.edges;
        if held_copies.is_empty() {
            expansion.evaluations = Some(edges.iter().map(evaluate).collect());
        } else {
            let mut evaluations = Vec::with_capacity(edges.len());
            let mut start = 0;
            for (source, target, count) in held_copies {
                evaluations.extend(edges[start..target].iter().map(evaluate));
                // Board evaluation only uses hold to count whether T is available.
                // Both ranges contain matching geometry and non-T hold pieces.
                evaluations.extend_from_within(source..source + count);
                start = target + count;
            }
            evaluations.extend(edges[start..].iter().map(evaluate));
            expansion.evaluations = Some(evaluations);
        }
    }
    #[cfg(feature = "profile-search")]
    {
        let (wall, cpu) = clock.elapsed();
        expansion.profile.evaluation_wall_ns = wall;
        expansion.profile.evaluation_cpu_ns = cpu;
    }
    expansion
}

impl Search {
    pub(super) fn cancel_stream(&mut self) {
        self.stream.cancel();
    }

    /// Stop outstanding expansion work without publishing any more results.
    /// A subsequent `search_step` resumes from the retained graph.
    pub fn pause(&mut self) {
        self.stream.cancel();
        while self.stream.outstanding != 0 {
            let completed = self
                .stream
                .receiver
                .get_mut()
                .unwrap()
                .recv()
                .expect("expansion completion channel");
            self.stream.take(completed);
        }
        debug_assert!(self.stream.reserved.is_empty());
    }

    fn can_search(&self) -> bool {
        self.started
            && !self.queue.is_empty()
            && !self.counters.memory_limited
            && !self.layers[0].nodes[self.root as usize].dead
    }

    fn dispatch_group(&mut self, selected: Vec<Selected>) {
        if selected.is_empty() {
            return;
        }
        debug_assert!(selected.len() <= DISPATCH_GROUP_SIZE);
        #[cfg(feature = "profile-search")]
        let clock = Instant::now();
        let generation = self.stream.generation.load(Ordering::Relaxed);
        let revision = Arc::clone(&self.stream.generation);
        let sender = self.stream.sender.clone();
        let weights = Arc::clone(&self.stream.weights);
        let use_hold = self.config.use_hold;
        let cavity_mode = self.cavity_mode;
        let jobs = selected.len();
        // Credits count leaves, including cancelled generations. Pending
        // selection buffers acquire credits only when submitted to the pool.
        self.stream.outstanding += jobs;
        self.pool
            .as_ref()
            .expect("dispatch requires workers")
            .spawn(move || {
                if revision.load(Ordering::Relaxed) != generation {
                    for selected in selected {
                        let _ = sender.send(Completion {
                            generation,
                            key: (selected.depth, selected.node),
                            expansion: None,
                        });
                    }
                    return;
                }
                let revision = revision.as_ref();
                let weights = weights.as_ref();
                let sender = &sender;
                // This closure now runs on a Rayon worker. Individual leaves
                // enter worker-local FIFO deques and can be stolen; only one
                // external injection and set of shared handles is needed.
                rayon::scope_fifo(move |scope| {
                    for selected in selected {
                        scope.spawn_fifo(move |_| {
                            let expansion = (revision.load(Ordering::Relaxed) == generation)
                                .then(|| prepare(selected, use_hold, weights, cavity_mode));
                            // Publish each result immediately. The controller
                            // never waits for the rest of this dispatch group.
                            let _ = sender.send(Completion {
                                generation,
                                key: (selected.depth, selected.node),
                                expansion,
                            });
                        });
                    }
                });
            });
        #[cfg(feature = "profile-search")]
        {
            self.profile.dispatch_wall_ns += clock.elapsed().as_nanos() as u64;
            self.profile.dispatch_groups += 1;
            self.profile.dispatch_jobs += jobs as u64;
        }
    }

    fn dispatch(&mut self) {
        if self.pool.is_none() {
            return;
        }
        let limit = self.config.threads.saturating_mul(2);
        if self.stream.outstanding >= limit {
            return;
        }
        #[cfg(feature = "profile-search")]
        let clock = ProfileClock::start();
        // Bound duplicate attempts when all reachable leaves are reserved or a
        // random walk reaches the depth limit. Never wait for a full cohort.
        let attempts = limit.saturating_mul(8).saturating_add(16);
        let mut selection_weights = [0.0; 255];
        let mut pending = Vec::new();
        for attempt in 0..attempts {
            if self.stream.outstanding.saturating_add(pending.len()) >= limit {
                break;
            }
            let seed = self.rng.next();
            #[cfg(feature = "profile-search")]
            let walk_clock = Instant::now();
            let selected =
                self.select_cached(&mut Random(seed), attempt == 0, &mut selection_weights);
            #[cfg(feature = "profile-search")]
            {
                self.profile.selection_walk_wall_ns += walk_clock.elapsed().as_nanos() as u64;
            }
            let Some(selected) = selected else {
                continue;
            };
            let key = (selected.depth, selected.node);
            if !self.stream.reserved.insert(key) {
                continue;
            }
            if pending.is_empty() {
                pending.reserve(DISPATCH_GROUP_SIZE);
            }
            pending.push(selected);
            if pending.len() == DISPATCH_GROUP_SIZE {
                self.dispatch_group(std::mem::take(&mut pending));
            }
        }
        // A root or narrow frontier may yield only one leaf. Always submit the
        // remainder now rather than waiting for a complete dispatch group.
        self.dispatch_group(pending);
        #[cfg(feature = "profile-search")]
        {
            let (wall, cpu) = clock.elapsed();
            self.profile.selection_wall_ns += wall;
            self.profile.selection_cpu_ns += cpu;
        }
    }

    fn receive_ready(&mut self, first: Option<Completion>) -> Vec<Expansion> {
        let mut ready = Vec::new();
        let mut next = first;
        loop {
            let completed = next
                .take()
                .or_else(|| self.stream.receiver.get_mut().unwrap().try_recv().ok());
            let Some(completed) = completed else {
                break;
            };
            if let Some(expansion) = self.stream.take(completed) {
                let selected = expansion.selected;
                let valid = self
                    .layers
                    .get(selected.depth)
                    .and_then(|layer| layer.nodes.get(selected.node as usize))
                    .is_some_and(|node| !node.dead && node.children.is_none());
                if valid {
                    ready.push(expansion);
                } else {
                    self.stream
                        .reserved
                        .remove(&(selected.depth, selected.node));
                }
            }
        }
        ready
    }

    /// Publish completed expansions and keep the other workers running.
    /// Waits for one completion only when none are ready; never waits for every
    /// worker or for a fixed number of selections. Zero means no progress.
    pub fn search_step(&mut self) -> usize {
        if !self.can_search() {
            self.cancel_stream();
            return 0;
        }
        if !self.select_guidance.is_empty()
            && self.stats().max_depth + 1 >= self.select_guidance.len()
        {
            self.select_guidance.clear();
        }
        if self.pool.is_none() {
            #[cfg(feature = "profile-search")]
            let clock = ProfileClock::start();
            let seed = self.rng.next();
            let mut selection_weights = [0.0; 255];
            let selected = self.select_cached(&mut Random(seed), true, &mut selection_weights);
            #[cfg(feature = "profile-search")]
            {
                let (wall, cpu) = clock.elapsed();
                self.profile.selection_wall_ns += wall;
                self.profile.selection_walk_wall_ns += wall;
                self.profile.selection_cpu_ns += cpu;
            }
            return selected.map_or(0, |selected| {
                let expansion = prepare(
                    selected,
                    self.config.use_hold,
                    &self.weights,
                    self.cavity_mode,
                );
                #[cfg(feature = "profile-search")]
                {
                    self.profile.expansion_wall_ns += expansion.profile.expansion_wall_ns;
                    self.profile.evaluation_wall_ns += expansion.profile.evaluation_wall_ns;
                    self.profile.evaluation_cpu_ns += expansion.profile.evaluation_cpu_ns;
                }
                self.register_expansions(vec![expansion], true)
            });
        }
        let mut first = None;
        loop {
            let ready = self.receive_ready(first.take());
            // Reserve other leaves before publishing and backing up these
            // completions, so those workers run during controller work.
            self.dispatch();
            if !ready.is_empty() {
                let expanded = self.register_expansions(ready, true);
                if self.can_search() {
                    self.dispatch();
                } else {
                    self.cancel_stream();
                }
                if expanded != 0 || !self.can_search() {
                    return expanded;
                }
            }
            if self.stream.outstanding == 0 {
                return 0;
            }
            #[cfg(feature = "profile-search")]
            let clock = Instant::now();
            first = Some(
                self.stream
                    .receiver
                    .get_mut()
                    .unwrap()
                    .recv()
                    .expect("expansion completion channel"),
            );
            #[cfg(feature = "profile-search")]
            {
                self.profile.expansion_wall_ns += clock.elapsed().as_nanos() as u64;
            }
        }
    }
}

impl Drop for Search {
    fn drop(&mut self) {
        self.pause();
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod dispatch_group_tests {
    use super::*;
    use std::sync::Condvar;

    struct ReleaseGate(Arc<(Mutex<bool>, Condvar)>);

    impl ReleaseGate {
        fn open(&self) {
            *self.0.0.lock().unwrap() = true;
            self.0.1.notify_all();
        }
    }

    impl Drop for ReleaseGate {
        fn drop(&mut self) {
            self.open();
        }
    }

    fn search(threads: usize) -> Search {
        let mut search = Search::new(
            Config {
                max_nodes: 50_000,
                use_hold: false,
                ..Config::for_threads(threads)
            },
            Weights::default(),
        )
        .unwrap();
        search.start(Board::default(), vec![PieceKind::T, PieceKind::I]);
        search
    }

    #[test]
    fn a_single_root_is_submitted_without_waiting_for_a_full_group() {
        let mut search = search(2);
        search.dispatch();
        assert_eq!(search.stream.outstanding, 1);
        let completed = search
            .stream
            .receiver
            .get_mut()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .expect("partial root dispatch was not submitted");
        assert_eq!(completed.key, (0, search.root));
        assert!(completed.expansion.is_some());
        assert!(search.stream.take(completed).is_some());
        assert_eq!(search.stream.outstanding, 0);
        assert_eq!(search.stats().nodes, 1);
        assert_eq!(search.stats().edges, 0);
        #[cfg(feature = "profile-search")]
        {
            assert_eq!(search.profile.dispatch_groups, 1);
            assert_eq!(search.profile.dispatch_jobs, 1);
        }
        search.pause();
    }

    #[test]
    fn a_cancelled_queued_group_returns_one_credit_per_leaf_including_the_tail() {
        let threads = 12;
        let mut search = search(threads);
        let workers = search.pool.as_ref().unwrap().current_num_threads();
        // ReleaseGate is dropped before Search on assertion failure, so a
        // failed test cannot leave pause waiting behind blocked worker jobs.
        let gate = ReleaseGate(Arc::new((Mutex::new(false), Condvar::new())));
        let (started, running) = mpsc::channel();
        for _ in 0..workers {
            let gate = Arc::clone(&gate.0);
            let started = started.clone();
            search.pool.as_ref().unwrap().spawn(move || {
                let _ = started.send(());
                let (lock, wake) = &*gate;
                let mut open = lock.lock().unwrap();
                while !*open {
                    open = wake.wait(open).unwrap();
                }
            });
        }
        for _ in 0..workers {
            running
                .recv_timeout(Duration::from_secs(10))
                .expect("worker did not enter the dispatch test gate");
        }
        let selected = search.select(&mut Random::new(9013), false).unwrap();
        let generation = search.stream.generation.load(Ordering::Relaxed);
        let group_len = DISPATCH_GROUP_SIZE;
        let mut group = Vec::with_capacity(group_len);
        let mut expected = HashSet::new();
        for node in 0..group_len {
            let selected = Selected {
                node: node as u32,
                ..selected
            };
            let key = (selected.depth, selected.node);
            search.stream.reserved.insert(key);
            expected.insert(key);
            group.push(selected);
        }
        search.dispatch_group(group);
        let tail = Selected {
            node: 100,
            ..selected
        };
        let tail_key = (tail.depth, tail.node);
        search.stream.reserved.insert(tail_key);
        expected.insert(tail_key);
        search.dispatch_group(vec![tail]);
        assert_eq!(search.stream.outstanding, group_len + 1);
        assert!(search.stream.outstanding <= search.config.threads * 2);
        search.cancel_stream();
        assert_eq!(search.stream.outstanding, group_len + 1);
        assert!(search.stream.reserved.is_empty());
        gate.open();

        let mut completed_keys = HashSet::new();
        for _ in 0..group_len + 1 {
            let completed = search
                .stream
                .receiver
                .get_mut()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .expect("cancelled dispatch group did not return every leaf credit");
            assert_eq!(completed.generation, generation);
            assert!(completed.expansion.is_none());
            assert!(completed_keys.insert(completed.key));
            assert!(search.stream.take(completed).is_none());
        }
        assert_eq!(completed_keys, expected);
        assert_eq!(search.stream.outstanding, 0);
        assert!(search.stream.reserved.is_empty());
        assert_eq!(search.stats().nodes, 1);
        assert_eq!(search.stats().edges, 0);
        #[cfg(feature = "profile-search")]
        {
            assert_eq!(search.profile.dispatch_groups, 2);
            assert_eq!(search.profile.dispatch_jobs, (group_len + 1) as u64);
        }
        search.pause();
    }
}
