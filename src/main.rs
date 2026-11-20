use clap::{Args, Parser, Subcommand};
use hikari_rei::board::{Board, PieceKind};
use hikari_rei::eval::Weights;
use hikari_rei::movegen::find_path;
use hikari_rei::protocol;
use hikari_rei::search::{Config, Search};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    version,
    about = "Original Hikari's board, evaluator and retained DAG in Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct Options {
    /// Number of native search workers; defaults to the available CPU parallelism.
    #[arg(long, default_value_t = available_workers())]
    threads: usize,
    #[arg(long, default_value_t = 32)]
    max_depth: usize,
    /// Stop adding nodes at this bound; keep the existing DAG.
    #[arg(long, default_value_t = 2_000_000)]
    max_nodes: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long)]
    no_hold: bool,
    #[arg(long)]
    no_speculation: bool,
}

fn available_workers() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get())
}

impl Options {
    fn config(&self) -> Config {
        let defaults = Config::for_threads(self.threads);
        Config {
            threads: self.threads,
            max_depth: self.max_depth,
            max_nodes: self.max_nodes,
            seed: self.seed,
            use_hold: !self.no_hold,
            use_speculation: !self.no_speculation,
            ..defaults
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Continuous pondering via JSON-line Tetris Bot Protocol.
    Tbp {
        #[command(flatten)]
        options: Options,
    },
    /// Search one position and print actual unique nodes, edges and depth.
    Bench {
        #[command(flatten)]
        options: Options,
        #[arg(long, default_value_t = 3000)]
        ms: u64,
        /// Benchmark the reference batch implementation with this selection count.
        #[arg(long)]
        reference_batch_size: Option<usize>,
        /// JSON containing board and known queue; without it use an empty board.
        #[arg(long)]
        position: Option<PathBuf>,
    },
    /// Play seeded seven-bag pieces, retaining the DAG between confirmed moves.
    Selfplay {
        #[command(flatten)]
        options: Options,
        #[arg(long, default_value_t = 100)]
        pieces: usize,
        #[arg(long, default_value_t = 20)]
        ms_per_move: u64,
        /// Public countdown thinking time before the first move.
        #[arg(long, default_value_t = 3000)]
        countdown_ms: u64,
        /// Save a native placement/board/path replay as JSON.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Record a 180-second game with real-time continuous pondering.
    Ultra {
        #[command(flatten)]
        options: Options,
        #[arg(long, default_value_t = 180)]
        seconds: u32,
        #[arg(long, default_value_t = 3000)]
        countdown_ms: u64,
        #[arg(long, default_value = "runs/ultra/hikari-rust/replay.json")]
        output: PathBuf,
    },
}

fn main() {
    if let Err(reason) = run() {
        eprintln!("{reason}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::Tbp { options } => protocol::run_tbp(
            options.config(),
            Weights {
                clear4: 1000,
                ren: 140,
                b2b_destroy: -250,
                ..Weights::default()
            },
        ),
        Command::Ultra {
            options,
            seconds,
            countdown_ms,
            output,
        } => {
            let summary = hikari_rei::ultra::run(options.config(), seconds, countdown_ms, &output)?;
            println!("{summary}");
            Ok(())
        }
        Command::Bench {
            options,
            ms,
            position,
            reference_batch_size,
        } => {
            let mut config = options.config();
            if let Some(batch_size) = reference_batch_size {
                config.batch_size = batch_size;
            }
            let (board, queue) = if let Some(path) = position {
                let value: Value = serde_json::from_str(
                    &std::fs::read_to_string(path).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                protocol::parse_position(&value)?
            } else {
                (
                    Board::default(),
                    "TOIJLZSITOJL"
                        .chars()
                        .map(|c| c.to_string().parse().unwrap())
                        .collect(),
                )
            };
            let mut search = Search::new(config.clone(), Weights::default())?;
            search.start(board, queue);
            let start = Instant::now();
            if reference_batch_size.is_some() {
                let deadline = start + Duration::from_millis(ms);
                while Instant::now() < deadline {
                    if search.search_batch() == 0 {
                        let stats = search.stats();
                        if stats.memory_limited || stats.dead {
                            break;
                        }
                    }
                }
            } else {
                search.think(Duration::from_millis(ms));
            }
            search.pause();
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "{}",
                json!({"reference_commit":hikari_rei::REFERENCE_COMMIT,"config":config,
                "search_mode":if reference_batch_size.is_some() { "reference_batches" } else { "continuous" },
                "elapsed_seconds":elapsed,"stats":search.stats(),"best_move":search.best_move(),"plan":search.current_plan()})
            );
            Ok(())
        }
        Command::Selfplay {
            options,
            pieces,
            ms_per_move,
            countdown_ms,
            output,
        } => {
            let mut random = SevenBag::new(options.seed);
            let mut search = Search::new(options.config(), Weights::default())?;
            let queue = (0..12).map(|_| random.draw()).collect();
            search.start(Board::default(), queue);
            search.think(Duration::from_millis(countdown_ms));
            let mut replay = Vec::new();
            let mut attacks = 0;
            let mut lines = 0;
            let mut reason = "piece_limit";
            for index in 0..pieces {
                search.think(Duration::from_millis(ms_per_move));
                let Some(mut chosen) = search.best_move() else {
                    reason = "no_legal_move";
                    break;
                };
                let mut held_empty = false;
                if chosen.piece.is_none() {
                    search.advance(&chosen)?;
                    search.add_next_piece(random.draw())?;
                    search.think(Duration::from_millis(ms_per_move));
                    let Some(next) = search.best_move() else {
                        reason = "no_move_after_hold";
                        break;
                    };
                    chosen = next;
                    held_empty = true;
                }
                let Some(piece) = chosen.piece else {
                    return Err("repeated empty-hold transition".into());
                };
                let path = find_path(search.board(), piece, held_empty || chosen.hold)
                    .ok_or("chosen placement has no input path")?;
                let mut after = *search.board();
                let locked = after.lock(piece);
                attacks += locked.attack().max(0);
                lines += locked.lines() as i32;
                let before = search.stats();
                search.advance(&chosen)?;
                search.add_next_piece(random.draw())?;
                replay.push(json!({"index":index,"piece":piece,"hold":held_empty||chosen.hold,"path":path,
                    "lock":locked,"board":search.board(),"search_before":before,"search_after":search.stats()}));
                if locked.death {
                    reason = "top_out";
                    break;
                }
            }
            let summary = json!({"reference_commit":hikari_rei::REFERENCE_COMMIT,"seed":options.seed,"pieces":replay.len(),
                "attack":attacks,"lines":lines,"reason":reason,"stats":search.stats(),"rules":"Hikari SimpleBoard (24 rows); original input costs; no real-time opponent"});
            if let Some(path) = output {
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&json!({"summary":summary,"moves":replay})).unwrap(),
                )
                .map_err(|e| e.to_string())?;
            }
            println!("{summary}");
            Ok(())
        }
    }
}

/// Environment-only randomizer. Search receives just twelve public pieces.
struct SevenBag {
    state: u64,
    bag: Vec<PieceKind>,
}
impl SevenBag {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            bag: Vec::new(),
        }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn draw(&mut self) -> PieceKind {
        if self.bag.is_empty() {
            self.bag = PieceKind::ALL.to_vec();
            for i in (1..self.bag.len()).rev() {
                let j = self.next() as usize % (i + 1);
                self.bag.swap(i, j);
            }
        }
        self.bag.pop().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench_config(args: &[&str]) -> Config {
        let Command::Bench { options, .. } = Cli::try_parse_from(args).unwrap().command else {
            panic!("expected bench command");
        };
        options.config()
    }

    #[test]
    fn search_workers_use_the_requested_configuration() {
        let config = bench_config(&["hikari_rei", "bench", "--threads", "8"]);
        assert_eq!(config.threads, 8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn explicit_search_options_override_worker_configuration() {
        let config = bench_config(&[
            "hikari_rei",
            "bench",
            "--threads",
            "8",
            "--max-depth",
            "17",
            "--max-nodes",
            "9000",
            "--seed",
            "19",
            "--no-hold",
            "--no-speculation",
        ]);
        assert_eq!(config.threads, 8);
        assert_eq!(config.max_depth, 17);
        assert_eq!(config.max_nodes, 9000);
        assert_eq!(config.seed, 19);
        assert!(!config.use_hold);
        assert!(!config.use_speculation);
    }

    #[test]
    fn zero_workers_remain_invalid() {
        let config = bench_config(&["hikari_rei", "bench", "--threads", "0"]);
        assert_eq!(config.threads, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn continuous_search_cli_has_no_batch_size_option() {
        assert!(Cli::try_parse_from(["hikari_rei", "bench", "--batch-size", "128"]).is_err());
    }

    #[test]
    fn reference_batch_option_is_available_only_for_bench() {
        let Command::Bench {
            reference_batch_size,
            ..
        } = Cli::try_parse_from(["hikari_rei", "bench", "--reference-batch-size", "128"])
            .unwrap()
            .command
        else {
            panic!("expected bench command");
        };
        assert_eq!(reference_batch_size, Some(128));
        for command in ["tbp", "selfplay", "ultra"] {
            assert!(
                Cli::try_parse_from(["hikari_rei", command, "--reference-batch-size", "128",])
                    .is_err()
            );
        }
    }
}
