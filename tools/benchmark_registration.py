#!/usr/bin/env python3
"""Compare exact graph registration with equal-work and equal-time searches."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from probe_command import check_batch_size, probe_command


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run(binary, args, output, commands, *, registration, batch_size, profiling=False):
    command = probe_command(binary, args, batch_size)
    started = time.time()
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    value = json.loads(process.stdout)
    check_batch_size(value, batch_size)
    if (value["fast_registration"] != registration
            or value["instrumented"] != profiling
            or value["fast_low_stack"] or value["prune_low_stack"]):
        raise RuntimeError("Binary feature selection does not match the requested variant")
    value["max_rss_bytes"] = None
    save(output, value)
    commands.append({"command": command, "wall_seconds": time.time() - started})
    return value


def same(a, b):
    return all(a[key] == b[key] for key in ("config", "stats", "best_move", "plan"))


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-profile", "candidate-profile"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-fast-registration", action="store_true",
                        help="Compare against an earlier fast-registration build")
    parser.add_argument("--profile-trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Use the same explicit selection count in both builds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile_trials < 1 or args.batch_size < 1:
        parser.error("--profile-trials and --batch-size must be positive")
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        parser.error("use a fresh output directory")
    binaries = {k: getattr(args, k.replace("-", "_")).resolve()
                for k in ("baseline", "candidate", "baseline-profile", "candidate-profile")}
    commands = []
    manifest = {"started_unix": time.time(), "platform": platform.platform(), "commands": commands,
                "binaries": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for k, p in binaries.items()},
                "low_stack_features": "disabled in all variants", "public_seeds": [43001, 43002, 43003],
                "baseline_fast_registration": args.baseline_fast_registration,
                "profile_trials": args.profile_trials,
                "batch_size": args.batch_size,
                "profile_seeds": [43001 + trial for trial in range(args.profile_trials)]}
    for suffix in ("", "-profile"):
        if manifest["binaries"]["baseline" + suffix]["sha256"] == manifest["binaries"]["candidate" + suffix]["sha256"]:
            parser.error("baseline and candidate executables must differ")
    save(output / "manifest.json", manifest)
    def is_registration(variant):
        return variant == "candidate" or args.baseline_fast_registration
    positions = {"empty": [0] * 24, "garbage-4": [1023 ^ (1 << (4 if y != 3 else 5)) for y in range(4)] + [0] * 20}
    for name, rows in positions.items():
        save(output / "positions" / f"{name}.json", {"board": rows, "queue": list("TOIJLZSITOJL")})
    fixed, capped, timed, profiles = [], [], [], []
    for name in positions:
        for threads in (1, 4):
            for batches in (128, 512):
                pair = {}
                order = ("baseline", "candidate") if batches == 128 else ("candidate", "baseline")
                for variant in order:
                    pair[variant] = run(binaries[variant], ["--batches", batches, "--threads", threads,
                        "--seed", 43001, "--position", output / "positions" / f"{name}.json"],
                        output / "fixed" / f"{name}-{threads}t-{batches}-{variant}.json", commands,
                        registration=is_registration(variant), batch_size=args.batch_size)
                if not same(pair["baseline"], pair["candidate"]):
                    raise RuntimeError("Fixed-work search semantics changed")
                fixed.append({"position": name, "threads": threads, "batches": batches,
                              "identical": True, "speedup": pair["baseline"]["elapsed_seconds"] / pair["candidate"]["elapsed_seconds"]})
                print(f"Exact work matched: {name}, {threads} workers, {batches} batches", flush=True)
    for cap in (2, 2000, 10000):
        for threads in (1, 4):
            pair = {variant: run(binaries[variant], ["--batches", 128, "--threads", threads,
                "--seed", 43003, "--max-nodes", cap],
                output / "caps" / f"{cap}-{threads}t-{variant}.json", commands,
                registration=is_registration(variant), batch_size=args.batch_size)
                for variant in ("baseline", "candidate")}
            if not same(pair["baseline"], pair["candidate"]):
                raise RuntimeError("Node-cap behavior changed")
            capped.append({"cap": cap, "threads": threads, "identical": True})
    for trial in range(3):
        for name in positions:
            for threads in (1, 4):
                pair = {}
                order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                for variant in order:
                    value = run(binaries[variant], ["--ms", 3000, "--threads", threads,
                        "--seed", 43001 + trial, "--position", output / "positions" / f"{name}.json"],
                        output / "timed" / f"{name}-{threads}t-{trial}-{variant}.json", commands,
                        registration=is_registration(variant), batch_size=args.batch_size)
                    if value["stats"]["dead"] or value["stats"]["memory_limited"]:
                        raise RuntimeError("Time-limited search died or hit its node cap")
                    pair[variant] = value
                ratio = (pair["candidate"]["stats"]["nodes"] / pair["candidate"]["elapsed_seconds"]) / (pair["baseline"]["stats"]["nodes"] / pair["baseline"]["elapsed_seconds"])
                timed.append({"position": name, "threads": threads, "trial": trial,
                              "node_rate_speedup": ratio,
                              **{variant: {"nodes": p["stats"]["nodes"], "elapsed_seconds": p["elapsed_seconds"], "max_rss_bytes": p["max_rss_bytes"]} for variant, p in pair.items()}})
                print(f"Timed pair: {name}, {threads} workers, trial {trial + 1}: {ratio:.3f}x", flush=True)
                save(output / "manifest.json", manifest)
    for trial in range(args.profile_trials):
        for name in positions:
            for threads in (1, 4):
                pair = {}
                order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                for variant in order:
                    pair[variant] = run(binaries[variant + "-profile"], ["--batches", 512, "--threads", threads,
                        "--seed", 43001 + trial, "--position", output / "positions" / f"{name}.json"],
                        output / "profile" / f"{name}-{threads}t-{trial}-{variant}.json", commands,
                        registration=is_registration(variant), batch_size=args.batch_size, profiling=True)
                if not same(pair["baseline"], pair["candidate"]):
                    raise RuntimeError("Profiled fixed-work search semantics changed")
                profiles.append({"position": name, "threads": threads, "trial": trial,
                    "total_speedup": pair["baseline"]["elapsed_seconds"] / pair["candidate"]["elapsed_seconds"],
                    "registration_speedup": pair["baseline"]["profile"]["registration_wall_ns"] / pair["candidate"]["profile"]["registration_wall_ns"],
                    **{v: p["profile"] for v, p in pair.items()}})
                print(f"Profiled pair: {name}, {threads} workers, trial {trial + 1}: "
                      f"{profiles[-1]['registration_speedup']:.3f}x registration", flush=True)
                save(output / "manifest.json", manifest)
    groups = {}
    for name in positions:
        for threads in (1, 4):
            rows = [r for r in timed if (r["position"], r["threads"]) == (name, threads)]
            groups[f"{name}/{threads}"] = {"node_rate_speedup": summary([r["node_rate_speedup"] for r in rows]),
                "registration_speedup": summary([r["registration_speedup"] for r in profiles
                    if (r["position"], r["threads"]) == (name, threads)]),
                **{v: {"nodes": summary([r[v]["nodes"] for r in rows]),
                       "max_rss_bytes": summary([r[v]["max_rss_bytes"] for r in rows]) if all(r[v]["max_rss_bytes"] for r in rows) else None}
                   for v in ("baseline", "candidate")}}
    save(output / "summary.json", {"groups": groups, "fixed_work": fixed, "node_caps": capped,
                                   "timed_pairs": timed, "fixed_work_profiles": profiles})
    manifest["completed_unix"] = time.time()
    manifest["wall_seconds"] = manifest["completed_unix"] - manifest["started_unix"]
    save(output / "manifest.json", manifest)
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
