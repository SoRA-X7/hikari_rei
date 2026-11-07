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
    /// Number of parallel move-generation workers.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Randomized leaf selections per batch (Hikari's ParallelCount).
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
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
impl Options {
    fn config(&self) -> Config {
        Config {
            threads: self.threads,
            batch_size: self.batch_size,
            max_depth: self.max_depth,
            max_nodes: self.max_nodes,
            seed: self.seed,
            use_hold: !self.no_hold,
            use_speculation: !self.no_speculation,
            ..Config::default()
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
        Command::Tbp { options } => protocol::run_tbp(options.config(), Weights::default()),
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
        } => {
            let config = options.config();
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
            search.think(Duration::from_millis(ms));
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "{}",
                json!({"reference_commit":hikari_rei::REFERENCE_COMMIT,"config":config,
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
