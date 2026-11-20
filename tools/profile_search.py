#!/usr/bin/env python3
"""Run serial search profiles and fixed-work instrumentation controls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from probe_command import check_batch_size, probe_command


PHASES = ("selection", "expansion", "registration", "evaluation", "backpropagation")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def describe(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def run(binary, arguments, path, commands, batch_size, reference=True):
    command = probe_command(binary, arguments, batch_size, reference=reference)
    started = time.time()
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    value = json.loads(result.stdout)
    if reference:
        check_batch_size(value, batch_size)
    save(path, value)
    commands.append({"command": command, "started_unix": started, "wall_seconds": time.time() - started})
    if value["stats"]["memory_limited"] or value["stats"]["dead"]:
        raise RuntimeError("Profile terminated at a node limit or death")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ms", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--control-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Selection count for the fixed-work reference control")
    parser.add_argument("--reference-batches", action="store_true",
                        help="Profile reference batches instead of normal continuous search")
    args = parser.parse_args()
    if min(args.ms, args.trials, args.control_batches, args.batch_size) <= 0:
        parser.error("durations and counts must be positive")
    output = args.output.resolve()
    if (output / "summary.json").exists():
        parser.error("use a fresh output directory")
    binaries = {"control": args.control.resolve(), "profile": args.profile.resolve()}
    commands = []
    manifest = {"started_unix": time.time(), "platform": platform.platform(),
                "binaries": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for k, p in binaries.items()}, "commands": commands,
                "clock": "Darwin process/thread user+kernel CPU clock; wall phases are separately measured",
                "movegen": "original Mirai; low-stack optimization features disabled",
                "batch_size": args.batch_size,
                "reference_profiles": args.reference_batches,
                "public_seeds": [42001 + n for n in range(args.trials)]}
    rows = [1023 ^ (1 << (4 if y % 4 != 3 else 5)) for y in range(4)]
    positions = {"empty": [0] * 24, "garbage-4": rows + [0] * 20}
    for name, board in positions.items():
        save(output / "positions" / f"{name}.json", {"board": board, "queue": list("TOIJLZSITOJL")})
    controls = []
    for trial in range(args.trials):
        for threads in (1, 4):
            results = {}
            order = ("control", "profile") if trial % 2 == 0 else ("profile", "control")
            for variant in order:
                results[variant] = run(binaries[variant], ["--batches", args.control_batches,
                    "--threads", threads, "--seed", 42001 + trial],
                    output / "control" / f"{threads}t-{trial}-{variant}.json", commands, args.batch_size)
            equal = all(results["control"][key] == results["profile"][key]
                        for key in ("config", "stats", "best_move", "plan"))
            if not equal:
                raise RuntimeError("Instrumentation changed fixed-batch search results")
            controls.append({"threads": threads, "trial": trial, "identical_results": equal,
                             "elapsed_ratio": results["profile"]["elapsed_seconds"] / results["control"]["elapsed_seconds"]})
            print(f"Fixed-work control: {threads} workers, trial {trial + 1}", flush=True)
    measurements = []
    for trial in range(args.trials):
        for name in positions:
            for threads in (1, 4):
                value = run(binaries["profile"], ["--ms", args.ms, "--threads", threads,
                    "--seed", 42001 + trial, "--position", output / "positions" / f"{name}.json"],
                    output / "profiles" / f"{name}-{threads}t-{trial}.json", commands, args.batch_size, args.reference_batches)
                if value["fast_low_stack"] or value["prune_low_stack"] or not value["instrumented"]:
                    raise RuntimeError("Expected original Mirai with profiling enabled")
                cpu = value["process_cpu_ns"]
                if not cpu:
                    raise RuntimeError("Process CPU clock unavailable on this platform")
                p = value["profile"]
                wall = value["elapsed_seconds"] * 1e9
                measurement = {"position": name, "threads": threads, "trial": trial,
                    "search_mode": value.get("search_mode", "reference_batches"),
                    "elapsed_seconds": value["elapsed_seconds"], "process_cpu_seconds": cpu / 1e9,
                    "movegen_cpu_percent": 100 * p["movegen_worker_cpu_ns"] / cpu,
                    "expansion_other_cpu_percent": 100 * (p["expansion_worker_cpu_ns"] - p["movegen_worker_cpu_ns"]) / cpu,
                    "evaluation_worker_cpu_percent": 100 * p.get("evaluation_worker_cpu_ns", 0) / cpu,
                    "phase_wall_percent": {k: 100 * p[k + "_wall_ns"] / wall for k in PHASES},
                    "nodes": value["stats"]["nodes"], "movegen_calls": p["movegen_calls"]}
                measurement["other_wall_percent"] = 100 - sum(measurement["phase_wall_percent"].values())
                if threads == 1:
                    measurement["serial_cpu_percent"] = {k: 100 * p[k + "_cpu_ns"] / cpu
                        for k in ("selection", "registration", "evaluation", "backpropagation")}
                measurements.append(measurement)
                print(f"Profile: {name}, {threads} workers, placement CPU {measurement['movegen_cpu_percent']:.1f}%", flush=True)
                save(output / "manifest.json", manifest)
    summary = {"controls": {str(t): {"profile_elapsed_ratio": describe([v["elapsed_ratio"] for v in controls if v["threads"] == t]),
                "identical_results": all(v["identical_results"] for v in controls if v["threads"] == t)} for t in (1, 4)},
               "measurements": measurements, "groups": {}}
    for name in positions:
        for threads in (1, 4):
            rows = [v for v in measurements if v["position"] == name and v["threads"] == threads]
            group = {k: describe([v[k] for v in rows]) for k in ("movegen_cpu_percent", "expansion_other_cpu_percent", "evaluation_worker_cpu_percent", "process_cpu_seconds", "nodes", "movegen_calls", "other_wall_percent")}
            group["phase_wall_percent"] = {k: describe([v["phase_wall_percent"][k] for v in rows]) for k in PHASES}
            if threads == 1:
                group["serial_cpu_percent"] = {k: describe([v["serial_cpu_percent"][k] for v in rows])
                    for k in ("selection", "registration", "evaluation", "backpropagation")}
            summary["groups"][f"{name}/{threads}"] = group
    manifest["completed_unix"] = time.time()
    manifest["wall_seconds"] = manifest["completed_unix"] - manifest["started_unix"]
    save(output / "manifest.json", manifest)
    save(output / "summary.json", summary)
    print(json.dumps(summary["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
