//! Search profiling and a fixed-work control without instrumentation.
use clap::Parser;
use hikari_rei::board::Board;
use hikari_rei::eval::Weights;
use hikari_rei::protocol::parse_position;
use hikari_rei::search::{Config, Search};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 3000)]
    ms: u64,
    #[arg(long, conflicts_with = "ms")]
    batches: Option<usize>,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 41001)]
    seed: u64,
    #[arg(long, default_value_t = 32_000_000)]
    max_nodes: usize,
    #[arg(long)]
    position: Option<PathBuf>,
}

#[cfg(feature = "profile-search")]
fn cpu_time() -> Option<u64> {
    hikari_rei::profile_clock::process_cpu_ns()
}
#[cfg(not(feature = "profile-search"))]
fn cpu_time() -> Option<u64> {
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = Config {
        threads: args.threads,
        seed: args.seed,
        max_nodes: args.max_nodes,
        ..Config::default()
    };
    let (board, queue) = if let Some(path) = &args.position {
        parse_position(&serde_json::from_slice::<Value>(&std::fs::read(path)?)?)?
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
    let cpu_start = cpu_time();
    let start = Instant::now();
    if let Some(batches) = args.batches {
        for _ in 0..batches {
            search.search_batch();
        }
    } else {
        search.think(Duration::from_millis(args.ms));
    }
    let elapsed = start.elapsed().as_secs_f64();
    let process_cpu_ns = cpu_time()
        .zip(cpu_start)
        .map(|(end, start)| end.saturating_sub(start));
    #[cfg(feature = "profile-search")]
    let profile = serde_json::to_value(search.profile())?;
    #[cfg(not(feature = "profile-search"))]
    let profile = Value::Null;
    println!(
        "{}",
        json!({
            "instrumented": cfg!(feature = "profile-search"),
            "fast_low_stack": cfg!(feature = "fast-low-stack"),
            "prune_low_stack": cfg!(feature = "prune-low-stack"),
            "fast_registration": cfg!(feature = "fast-registration"),
            "config":config,"elapsed_seconds":elapsed,"process_cpu_ns":process_cpu_ns,
            "stats":search.stats(),"best_move":search.best_move(),"plan":search.current_plan(),
            "profile":profile
        })
    );
    Ok(())
}
