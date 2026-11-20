"""Keep work sizes equal when comparing current and older profile probes."""
from functools import lru_cache
import subprocess


@lru_cache(maxsize=None)
def probe_help(binary):
    result = subprocess.run([str(binary), "--help"], check=True, capture_output=True, text=True)
    return result.stdout


def supports_batch_size(binary):
    return "--batch-size" in probe_help(binary)


def probe_command(binary, arguments, batch_size, reference=True):
    arguments = list(map(str, arguments))
    supports_continuous = "--reference-batches" in probe_help(binary)
    if not reference and supports_continuous:
        return [str(binary), *arguments]
    if supports_batch_size(binary):
        mode = ["--reference-batches"] if supports_continuous and "--batches" not in arguments else []
        return [str(binary), *mode, "--batch-size", str(batch_size), *arguments]
    if batch_size != 128:
        raise ValueError(f"{binary} does not support --batch-size; this older probe can only be compared at 128")
    return [str(binary), *arguments]


def check_batch_size(value, batch_size):
    if value["config"]["batch_size"] != batch_size:
        raise RuntimeError("Probe used a different batch size than requested")


@lru_cache(maxsize=None)
def bench_help(binary):
    return subprocess.run([str(binary), "bench", "--help"], check=True,
                          capture_output=True, text=True).stdout


def reference_bench_command(binary, arguments, batch_size):
    help_text = bench_help(binary)
    if "--reference-batch-size" in help_text:
        option = "--reference-batch-size"
    elif "--batch-size" in help_text:
        option = "--batch-size"
    else:
        raise ValueError(f"{binary} does not expose a reference batch benchmark")
    return [str(binary), "bench", option, str(batch_size), *map(str, arguments)]
