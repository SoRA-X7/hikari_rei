//! Reproducible, public development fixtures and move-generator measurements.
//! Fixture construction deliberately never calls the generator under test.
use clap::{Parser, Subcommand};
use hikari_rei::board::{Board, Piece, PieceKind};
use hikari_rei::movegen::{self, Instruction};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Corpus {
        #[arg(long)]
        output: PathBuf,
    },
    Dump {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Bench {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long, default_value_t = 150)]
        round_ms: u64,
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(Serialize, Deserialize)]
struct Corpus {
    format_version: u32,
    seed: u64,
    entries: Vec<Fixture>,
}
#[derive(Serialize, Deserialize)]
struct Fixture {
    id: String,
    source: String,
    height: usize,
    stratum: String,
    board: Board,
}
const STRATA: [&str; 4] = ["low_0_5", "middle_6_10", "upper_11_15", "fallback_16_plus"];
fn stratum(height: usize) -> &'static str {
    STRATA[match height {
        0..=5 => 0,
        6..=10 => 1,
        11..=15 => 2,
        _ => 3,
    }]
}
fn fixture(id: String, source: String, board: Board) -> Fixture {
    let height = board.max_height();
    Fixture {
        id,
        source,
        height,
        stratum: stratum(height).into(),
        board,
    }
}
fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
/// Only rotations and translations at spawn height, then a hard drop. All
/// intermediates are checked, making these boards reachable in either mode.
fn simple_placements(board: &Board, kind: PieceKind) -> Vec<Piece> {
    let Some(spawn) = board.spawn(kind) else {
        return Vec::new();
    };
    let mut rotations = vec![spawn];
    let mut current = spawn;
    for _ in 0..3 {
        let Some(next) = board.rotate(current, true) else {
            break;
        };
        rotations.push(next);
        current = next;
    }
    let mut placements = Vec::new();
    let mut seen = HashSet::new();
    for rotated in rotations {
        for direction in [-1, 1] {
            let mut shifted = rotated;
            loop {
                if let Some(dropped) = board.sonic_drop(shifted) {
                    let mut cells = dropped.cells();
                    cells.sort_unstable();
                    let mut after = *board;
                    if !after.lock(dropped).death && seen.insert(cells) {
                        placements.push(dropped);
                    }
                }
                shifted = shifted.shifted(direction, 0);
                if board.collides(shifted) {
                    break;
                }
            }
        }
    }
    placements.sort_by_key(|piece| {
        let mut cells = piece.cells();
        cells.sort_unstable();
        cells
    });
    placements
}
fn make_corpus() -> Corpus {
    let mut entries = vec![fixture("empty".into(), "empty".into(), Board::default())];
    let mut quotas = [0; 4];
    let mut seen = HashSet::new();
    for seed in 10_000..10_256_u64 {
        if quotas.iter().all(|&count| count >= 24) {
            break;
        }
        let mut state = seed;
        let mut board = Board::default();
        let mut captured = [false; 4];
        for ply in 0..48 {
            let kind = PieceKind::ALL[(random(&mut state) % 7) as usize];
            let placements = simple_placements(&board, kind);
            if placements.is_empty() {
                break;
            }
            let chosen = placements[(random(&mut state) % placements.len() as u64) as usize];
            assert!(!board.lock(chosen).death);
            let height = board.max_height();
            let group = match height {
                0..=5 => 0,
                6..=10 => 1,
                11..=15 => 2,
                _ => 3,
            };
            let minimum = [1, 6, 11, 16][group] + seed as usize % 5;
            if height >= minimum
                && !captured[group]
                && quotas[group] < 24
                && seen.insert(board.rows)
            {
                entries.push(fixture(
                    format!("reachable-{seed}-{ply}"),
                    "deterministic-legal-spawn-paths".into(),
                    board,
                ));
                quotas[group] += 1;
                captured[group] = true;
            }
        }
    }
    assert!(
        quotas.iter().all(|&count| count == 24),
        "fixture quotas: {quotas:?}"
    );
    // Explicit adversarial geometry. These are synthetic position fixtures,
    // kept separate from the legal-placement and actual-play corpora.
    let patterns: &[(&str, &[u16])] = &[
        ("t-slot", &[0b1111111011, 0b1111110001, 0b1111111011]),
        ("tuck", &[0b1111100000, 0b1111110011, 0b0000011000]),
        ("overhang", &[0b1111100011, 0b1111100011, 0b0000111111]),
        ("cave", &[0b1110001111, 0b1110001111, 0b1111101111]),
        (
            "stair",
            &[0b1111111110, 0b1111111100, 0b1111111000, 0b1111110000],
        ),
        (
            "well",
            &[0b1111011111, 0b1111011111, 0b1111011111, 0b1111011111],
        ),
    ];
    for &(name, rows) in patterns {
        for base in [0, 5, 10, 15] {
            let mut board = Board::default();
            for y in 0..base {
                board.rows[y] = 0b1111011111;
            }
            board.rows[base..base + rows.len()].copy_from_slice(rows);
            entries.push(fixture(
                format!("adversarial-{name}-{base}"),
                "synthetic-spin-tuck-overhang".into(),
                board,
            ));
        }
    }
    Corpus {
        format_version: 1,
        seed: 10_000,
        entries,
    }
}
fn replay(board: &Board, placement: movegen::Placement) -> Result<(), String> {
    let path = movegen::find_path(board, placement.piece, false).ok_or("no path")?;
    if path.instructions.len() > movegen::MAX_INSTRUCTIONS {
        return Err("too many instructions".into());
    }
    let mut current = board.spawn(placement.piece.kind).ok_or("no spawn")?;
    let mut previous = Instruction::None;
    let mut cost = 0;
    for instruction in path.instructions {
        let next = match instruction {
            Instruction::None => Some(current),
            Instruction::Left => Some(current.shifted(-1, 0)),
            Instruction::Right => Some(current.shifted(1, 0)),
            Instruction::Cw => board.rotate(current, true),
            Instruction::Ccw => board.rotate(current, false),
            Instruction::SonicDrop => board.sonic_drop(current),
        }
        .ok_or("failed input")?;
        if board.collides(next) {
            return Err("input collides".into());
        }
        let mut time = match instruction {
            Instruction::None => 0,
            Instruction::SonicDrop => 2 * (current.y as i32 - next.y as i32),
            _ => 1,
        };
        if previous == instruction {
            time += 1;
        }
        cost += time;
        current = next;
        previous = instruction;
    }
    if board.sonic_drop(current) != Some(placement.piece) {
        return Err("final placement differs".into());
    }
    if cost != path.time || path.time != placement.time {
        return Err(format!(
            "cost differs: {cost}/{}/{}",
            path.time, placement.time
        ));
    }
    Ok(())
}
fn write(path: PathBuf, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Corpus { output } => write(output, &make_corpus())?,
        Command::Dump { corpus, output } => {
            let corpus: Corpus = serde_json::from_slice(&std::fs::read(corpus)?)?;
            let mut cases = Vec::new();
            let mut checked = 0;
            let mut errors = 0;
            for fixture in corpus.entries {
                assert_eq!(fixture.height, fixture.board.max_height());
                for kind in PieceKind::ALL {
                    let placements = movegen::generate(&fixture.board, kind);
                    let mut records = Vec::new();
                    let mut failures = Vec::new();
                    for placement in placements {
                        checked += 1;
                        if let Err(error) = replay(&fixture.board, placement) {
                            errors += 1;
                            failures.push(json!({"placement":placement,"error":error}));
                        }
                        let mut cells = placement.piece.cells();
                        cells.sort_unstable();
                        records.push(json!({"cells":cells,"spin":placement.piece.tspin(),"time":placement.time,"piece":placement.piece}));
                    }
                    let mut sorted = records.clone();
                    sorted.sort_by_key(|record| record.to_string());
                    cases.push(json!({"id":fixture.id,"source":fixture.source,"height":fixture.height,"stratum":fixture.stratum,"kind":kind,"placements":records,"sorted_placements":sorted,"path_errors":failures}));
                }
            }
            write(
                output,
                &json!({"format_version":1,"fast_low_stack":cfg!(feature="fast-low-stack"),"checked_paths":checked,"path_errors":errors,"cases":cases}),
            )?;
        }
        Command::Bench {
            corpus,
            round_ms,
            rounds,
            output,
        } => {
            assert!(round_ms > 0 && rounds > 0);
            let corpus: Corpus = serde_json::from_slice(&std::fs::read(corpus)?)?;
            let mut measurements = Vec::new();
            for stratum in STRATA {
                let cases: Vec<_> = corpus
                    .entries
                    .iter()
                    .filter(|fixture| fixture.stratum == stratum)
                    .flat_map(|fixture| PieceKind::ALL.map(|kind| (&fixture.board, kind)))
                    .collect();
                if cases.is_empty() {
                    continue;
                }
                for &(board, kind) in &cases {
                    black_box(movegen::generate(board, kind));
                }
                let mut cursor = 0;
                for round in 0..rounds {
                    let start = Instant::now();
                    let mut calls = 0_u64;
                    let mut placements = 0_u64;
                    while start.elapsed() < Duration::from_millis(round_ms) {
                        // A full corpus traversal per time check avoids timing every call.
                        for _ in 0..cases.len() {
                            let (board, kind) = cases[cursor];
                            placements +=
                                black_box(movegen::generate(black_box(board), black_box(kind)))
                                    .len() as u64;
                            calls += 1;
                            cursor = (cursor + 1) % cases.len();
                        }
                    }
                    let seconds = start.elapsed().as_secs_f64();
                    measurements.push(json!({"stratum":stratum,"round":round,"cases":cases.len(),"calls":calls,"placements":placements,"seconds":seconds,"ns_per_call":seconds*1e9/calls as f64,"calls_per_second":calls as f64/seconds}));
                }
            }
            write(
                output,
                &json!({"format_version":1,"fast_low_stack":cfg!(feature="fast-low-stack"),"measurements":measurements}),
            )?;
        }
    }
    Ok(())
}
