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
#[command(group(clap::ArgGroup::new("reference_mode").args(["batches", "reference_batches"])))]
struct Args {
    #[arg(long, default_value_t = 3000)]
    ms: u64,
    /// Run a fixed number of reference search_batch calls.
    #[arg(long, conflicts_with = "ms")]
    batches: Option<usize>,
    /// Time the reference search_batch implementation instead of continuous search.
    #[arg(long)]
    reference_batches: bool,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Selections per reference batch; requires --batches or --reference-batches.
    #[arg(long, requires = "reference_mode")]
    batch_size: Option<usize>,
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
    let defaults = Config::for_threads(args.threads);
    let config = Config {
        threads: args.threads,
        batch_size: args.batch_size.unwrap_or(defaults.batch_size),
        seed: args.seed,
        max_nodes: args.max_nodes,
        ..defaults
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
    } else if args.reference_batches {
        let deadline = Instant::now() + Duration::from_millis(args.ms);
        while Instant::now() < deadline {
            if search.search_batch() == 0 {
                let stats = search.stats();
                if stats.memory_limited || stats.dead {
                    break;
                }
            }
        }
    } else {
        search.think(Duration::from_millis(args.ms));
    }
    search.pause();
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
            "search_mode": if args.batches.is_some() || args.reference_batches {
                "reference_batches"
            } else {
                "continuous"
            },
            "config":config,"elapsed_seconds":elapsed,"process_cpu_ns":process_cpu_ns,
            "stats":search.stats(),"best_move":search.best_move(),"plan":search.current_plan(),
            "profile":profile
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_requires_a_reference_search_mode() {
        assert!(Args::try_parse_from(["profile_probe", "--batch-size", "16"]).is_err());
        assert!(
            Args::try_parse_from(["profile_probe", "--batches", "2", "--batch-size", "16",])
                .is_ok()
        );
        assert!(
            Args::try_parse_from([
                "profile_probe",
                "--reference-batches",
                "--ms",
                "10",
                "--batch-size",
                "16",
            ])
            .is_ok()
        );
    }
}
