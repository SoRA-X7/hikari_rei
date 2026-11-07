#!/usr/bin/env python3
"""Paired public-development experiments for the low-stack move generator.

Build and preserve both binaries first, then run this script with no other CPU
load. It never opens evaluation seeds or opponent code. All raw observations,
the shared corpus, public seeds, hashes, and exact commands are saved.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


STRATA = ("low_0_5", "middle_6_10", "upper_11_15", "fallback_16_plus")


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratum(height):
    return STRATA[0 if height <= 5 else 1 if height <= 10 else 2 if height <= 15 else 3]


def run(command, commands, output=None):
    command = list(map(str, command))
    started = time.time()
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    commands.append({"command": command, "started_unix": started,
                     "wall_seconds": time.time() - started})
    if output is not None:
        save(output, json.loads(result.stdout))
    return result


def add_actual_boards(corpus_path, replay_path):
    """Add evenly spaced unique actual-play boards; never sample by treatment."""
    corpus = load(corpus_path)
    replay = load(replay_path)
    candidates = []
    seen = set()
    for move in replay["moves"]:
        board = move["board"]
        key = tuple(board["rows"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append((move["index"], board))
    count = min(96, len(candidates))
    indices = sorted({round(i * (len(candidates) - 1) / max(1, count - 1))
                      for i in range(count)})
    for index in indices:
        ply, board = candidates[index]
        height = max((i + 1 for i, row in enumerate(board["rows"]) if row), default=0)
        corpus["entries"].append({
            "id": f"actual-selfplay-{ply}", "source": "selfplay-replay",
            "height": height, "stratum": stratum(height), "board": board,
        })
    corpus["actual_play_source"] = {"path": str(replay_path.resolve()),
                                    "sha256": sha256(replay_path),
                                    "unique_boards_available": len(candidates),
                                    "boards_sampled": len(indices)}
    save(corpus_path, corpus)


def add_ultra_boards(corpus_path, replay_path):
    corpus = load(corpus_path)
    replay = load(replay_path)
    candidates = []
    seen = set()
    for encoded in replay["playback"]["boards"]:
        assert len(encoded) == 240 and set(encoded) <= set("01234567")
        rows = [sum((cell != "0") << x for x, cell in enumerate(encoded[y * 10:(y + 1) * 10]))
                for y in range(24)]
        if tuple(rows) not in seen:
            seen.add(tuple(rows))
            candidates.append(rows)
    count = min(96, len(candidates))
    indices = sorted({round(i * (len(candidates) - 1) / max(1, count - 1)) for i in range(count)})
    for index in indices:
        rows = candidates[index]
        height = max((i + 1 for i, row in enumerate(rows) if row), default=0)
        corpus["entries"].append({
            "id": f"actual-ultra-{index}", "source": "ultra-replay",
            "height": height, "stratum": stratum(height),
            "board": {"rows": rows, "back_to_back": False, "hold": None, "bag": 0, "ren": 0},
        })
    corpus["ultra_source"] = {"path": str(replay_path.resolve()), "sha256": sha256(replay_path),
                              "boards_sampled": len(indices),
                              "note": "Only geometry is used by move generation; other Board fields are neutral."}
    save(corpus_path, corpus)


def key(placement):
    return tuple(map(tuple, placement["cells"])), placement["spin"]


def compare_dumps(baseline_path, candidate_path):
    baseline, candidate = load(baseline_path), load(candidate_path)
    left = {(case["id"], case["kind"]): case for case in baseline["cases"]}
    right = {(case["id"], case["kind"]): case for case in candidate["cases"]}
    if left.keys() != right.keys():
        raise RuntimeError("Probe fixture sets differ")
    totals = collections.defaultdict(collections.Counter)
    differences = []
    for case_key, first in left.items():
        second = right[case_key]
        a, b = first["placements"], second["placements"]
        ka, kb = {key(p) for p in a}, {key(p) for p in b}
        ca, cb = collections.defaultdict(list), collections.defaultdict(list)
        for p in a:
            ca[key(p)].append(p["time"])
        for p in b:
            cb[key(p)].append(p["time"])
        costs = [k for k in ka & kb if sorted(ca[k]) != sorted(cb[k])]
        order_changed = [key(p) for p in a] != [key(p) for p in b]
        representation_changed = a != b
        record = {"cases": 1, "baseline_placements": len(a), "candidate_placements": len(b),
                  "lost": len(ka - kb), "added": len(kb - ka),
                  "set_changed_cases": int(ka != kb), "cost_changed_placements": len(costs),
                  "order_changed_cases": int(order_changed),
                  "raw_changed_cases": int(representation_changed),
                  "baseline_path_errors": len(first["path_errors"]),
                  "candidate_path_errors": len(second["path_errors"])}
        for group in ("all", first["stratum"], "source:" + first["source"]):
            totals[group].update(record)
        if ka != kb or costs or order_changed or representation_changed:
            differences.append({"id": first["id"], "kind": first["kind"],
                                "height": first["height"], "source": first["source"],
                                **record,
                                "lost_examples": [p for p in a if key(p) not in kb][:5],
                                "added_examples": [p for p in b if key(p) not in ka][:5],
                                "cost_examples": [{"cells": k[0], "spin": k[1],
                                                   "baseline": ca[k], "candidate": cb[k]}
                                                  for k in sorted(costs)[:5]]})
    return {"groups": {k: dict(v) for k, v in totals.items()},
            "baseline_checked_paths": baseline["checked_paths"],
            "candidate_checked_paths": candidate["checked_paths"],
            "baseline_path_errors": baseline["path_errors"],
            "candidate_path_errors": candidate["path_errors"],
            "differences": differences}


def describe(values):
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "mean": statistics.mean(values)}


def micro_summary(observations):
    groups = collections.defaultdict(list)
    pairs = collections.defaultdict(dict)
    for observation in observations:
        for row in observation["measurements"]:
            groups[(row["stratum"], observation["variant"])].append(row["ns_per_call"])
            pairs[(observation["trial"], row["stratum"])][observation["variant"]] = row["ns_per_call"]
    result = {}
    for group in STRATA:
        baseline = groups[(group, "baseline")]
        candidate = groups[(group, "candidate")]
        ratios = [p["baseline"] / p["candidate"] for (trial, stratum_name), p in pairs.items()
                  if stratum_name == group]
        result[group] = {"baseline_ns_per_call": describe(baseline),
                         "candidate_ns_per_call": describe(candidate),
                         "paired_speedup": describe(ratios),
                         "median_improvement_percent": (statistics.median(ratios) - 1) * 100}
    return result


def search_summary(observations):
    groups = collections.defaultdict(list)
    pairs = collections.defaultdict(dict)
    for row in observations:
        groups[(row["position"], row["threads"], row["variant"])].append(row)
        pairs[(row["position"], row["threads"], row["trial"])][row["variant"]] = row
    result = {}
    for position, threads in sorted({(r["position"], r["threads"]) for r in observations}):
        summary = {}
        for variant in ("baseline", "candidate"):
            rows = groups[(position, threads, variant)]
            summary[variant] = {field: describe([r["result"]["stats"][field] for r in rows])
                                for field in ("nodes", "generated", "expanded", "max_depth")}
            summary[variant]["nodes_per_second"] = describe([
                r["result"]["stats"]["nodes"] / r["result"]["elapsed_seconds"] for r in rows])
            summary[variant]["memory_limited_runs"] = sum(r["result"]["stats"]["memory_limited"] for r in rows)
        ratios = [
            (p["candidate"]["result"]["stats"]["nodes"] / p["candidate"]["result"]["elapsed_seconds"])
            / (p["baseline"]["result"]["stats"]["nodes"] / p["baseline"]["result"]["elapsed_seconds"])
            for (name, workers, trial), p in pairs.items() if (name, workers) == (position, threads)
        ]
        summary["paired_node_rate_speedup"] = describe(ratios)
        summary["median_node_rate_improvement_percent"] = (statistics.median(ratios) - 1) * 100
        result[f"{position}/{threads}-threads"] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-probe", "candidate-probe"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actual-replay", type=Path,
                        help="Optional selfplay replay to sample additional boards from")
    parser.add_argument("--ultra-replay", type=Path,
                        help="Optional Ultra replay to sample additional boards from")
    parser.add_argument("--search-ms", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--micro-trials", type=int, default=5)
    parser.add_argument("--micro-ms", type=int, default=150)
    parser.add_argument("--max-nodes", type=int, default=32_000_000)
    parser.add_argument("--positions", nargs="+", choices=("empty", "garbage-low", "garbage-middle", "garbage-high"))
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.micro_trials, args.micro_ms, args.search_ms, args.max_nodes) <= 0:
        parser.error("durations, trials and node cap must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("output already contains an experiment; use a fresh directory")
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    probes = {"baseline": args.baseline_probe.resolve(), "candidate": args.candidate_probe.resolve()}
    commands = []
    manifest = {"format_version": 1, "started_unix": time.time(),
                "platform": platform.platform(), "machine": platform.machine(),
                "logical_cpus": os.cpu_count(), "python": sys.version,
                "binaries": {name: {"path": str(path), "sha256": sha256(path)}
                             for name, path in binaries.items()},
                "probes": {name: {"path": str(path), "sha256": sha256(path)}
                           for name, path in probes.items()},
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "public_search_seeds": [41_001 + i for i in range(args.trials)],
                "feature_predicate": "max_height < 16", "commands": commands}
    save(output / "manifest.json", manifest)
    corpus_path = output / "corpus.json"
    run([probes["baseline"], "corpus", "--output", corpus_path], commands)
    if args.actual_replay:
        add_actual_boards(corpus_path, args.actual_replay)
    if args.ultra_replay:
        add_ultra_boards(corpus_path, args.ultra_replay)
    corpus = load(corpus_path)
    manifest["corpus"] = {"path": str(corpus_path), "sha256": sha256(corpus_path),
                          "boards": len(corpus["entries"]),
                          "strata": dict(collections.Counter(f["stratum"] for f in corpus["entries"])),
                          "sources": dict(collections.Counter(f["source"] for f in corpus["entries"]))}
    save(output / "manifest.json", manifest)
    print(f"Corpus: {manifest['corpus']['boards']} boards; validating both modes", flush=True)
    for variant in ("baseline", "candidate"):
        run([probes[variant], "dump", "--corpus", corpus_path,
             "--output", output / f"placements-{variant}.json"], commands)
    comparison = compare_dumps(output / "placements-baseline.json", output / "placements-candidate.json")
    save(output / "placement-comparison.json", comparison)
    print("Placement coverage: " + json.dumps(comparison["groups"]["all"]), flush=True)
    micro = []
    for trial in range(args.micro_trials):
        order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            path = output / f"micro-{trial:02}-{variant}.json"
            run([probes[variant], "bench", "--corpus", corpus_path, "--round-ms", args.micro_ms,
                 "--rounds", 1, "--output", path], commands)
            micro.append({"variant": variant, "trial": trial, **load(path)})
        print(f"Micro pair {trial + 1}/{args.micro_trials} complete", flush=True)
    save(output / "micro-summary.json", micro_summary(micro))
    positions = {"empty": {"board": [0] * 24, "queue": list("TOIJLZSITOJL")}}
    for name, height in (("garbage-low", 4), ("garbage-middle", 9), ("garbage-high", 17)):
        # Stable narrow-well garbage; high position exercises the original fallback.
        rows = [1023 ^ (1 << (4 if y % 4 != 3 else 5)) for y in range(height)]
        positions[name] = {"board": rows + [0] * (24 - height), "queue": list("TOIJLZSITOJL")}
    if args.positions:
        positions = {name: positions[name] for name in args.positions}
    for name, position in positions.items():
        save(output / "positions" / f"{name}.json", position)
    searches = []
    if not args.skip_search:
        total = len(positions) * 2 * args.trials
        completed = 0
        for trial in range(args.trials):
            for position in positions:
                for threads in (1, 4):
                    order = ("baseline", "candidate") if (trial + threads + list(positions).index(position)) % 2 == 0 else ("candidate", "baseline")
                    for variant in order:
                        path = output / "search" / f"{position}-{threads}t-{trial:02}-{variant}.json"
                        command = [binaries[variant], "bench", "--ms", args.search_ms,
                                   "--threads", threads, "--max-nodes", args.max_nodes,
                                   "--seed", 41_001 + trial, "--position", output / "positions" / f"{position}.json"]
                        run(command, commands, path)
                        searches.append({"position": position, "threads": threads,
                                         "trial": trial, "variant": variant, "result": load(path)})
                    completed += 1
                    print(f"Search pair {completed}/{total}: {position}, {threads} threads", flush=True)
                    save(output / "manifest.json", manifest)
    summary = {"placement_comparison": {k: v for k, v in comparison.items() if k != "differences"},
               "micro": micro_summary(micro),
               "search": search_summary(searches) if searches else {},
               "note": "Node-rate measures exploration, not playing strength; changed move costs/order can change the sampled DAG."}
    save(output / "summary.json", summary)
    manifest["completed_unix"] = time.time()
    manifest["wall_seconds"] = manifest["completed_unix"] - manifest["started_unix"]
    save(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    if comparison["baseline_path_errors"] or comparison["candidate_path_errors"]:
        raise SystemExit("Replay validation failures: inspect placement dumps")


if __name__ == "__main__":
    main()
