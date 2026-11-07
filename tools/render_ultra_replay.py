"""Embed a recorded Ultra run in the conversation's existing replay fragment."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    recorded = json.loads(args.replay.read_text())
    data = recorded["playback"]
    meta, summary = recorded["metadata"], recorded["summary"]
    assert data["end"] == summary["end_frame"]
    assert data["keys"] and data["keys"][-1][0] == data["end"]
    assert all(len(board) == 240 and set(board) <= set("01234567") for board in data["boards"])
    assert all(len(queue) <= 5 for queue in data["queues"])
    assert all(len(key) == 11 for key in data["keys"])
    assert all(a[0] <= b[0] for a, b in zip(data["keys"], data["keys"][1:]))
    for key in data["keys"]:
        assert 0 <= key[1] < len(data["boards"])
        assert all(p == -1 or 0 <= p < len(data["poses"]) for p in key[2:4])
        assert 0 <= key[5] < len(data["queues"])
    # Preserve every visible change. Sample changes to search counters at 10 Hz.
    compact = []
    for index, key in enumerate(data["keys"]):
        if (not compact or key[1:9] != compact[-1][1:9]
                or key[0] - compact[-1][0] >= 6 or index == len(data["keys"]) - 1):
            compact.append(key)
    data["keys"] = compact
    data["workers"] = meta["config"]["threads"]
    data["model"] = meta.get("binary_sha256", "hikari-rust-" + meta["reference_commit"])
    data["reason"] = "time_limit" if summary["completed"] else "top_out" if summary["topout"] else summary.get("error") or "ended"
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    fragment = args.fragment.read_text()
    result, count = re.subn(r'(<script type="application/json" id="hru-data">).*?(</script>)',
                            lambda match: match[1] + payload + match[2], fragment, flags=re.S)
    assert count == 1, "expected exactly one replay data block"
    assert len(result.encode()) < 1_000_000, "inline replay exceeds 1 MB"
    args.fragment.write_text(result)
    print(json.dumps({"fragment": str(args.fragment), "bytes": len(result.encode()),
                      "frames": data["end"], "keys": len(compact)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
