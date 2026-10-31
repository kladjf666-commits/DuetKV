"""Summarize only measured per-shape A/B results, retaining every raw round."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = json.loads(a.input.read_text(encoding="utf-8"))
    lines = ["PhaseKV optimization measurements", "Protocol: " + source["protocol"],
             "Status: " + source["status"], "Raw evidence: " + str(a.input), ""]
    if source["status"] != "measured":
        lines.append("No GPU speedup claim: this file does not contain completed official GPU timing.")
    else:
        for row in source["rows"]:
            lines.extend([json.dumps(row["shape"], sort_keys=True),
                          "stage | decode GPU/wall ms | wall speedup vs saved baseline | partials | scratch bytes | cooling wall p50/p95 ms"])
            baseline = row["stages"].get("baseline", {}).get("medians", {}).get("decode", {}).get("wall_ms")
            for name, stage in row["stages"].items():
                timing = stage["medians"]["decode"]
                speedup = f"{baseline / timing['wall_ms']:.4f}" if baseline else "unavailable"
                cooling = stage["lifecycle_percentiles_ms"]["cooling"]["wall_ms"]
                lines.append(f"{name} | {timing['gpu_ms']:.6f}/{timing['wall_ms']:.6f} | {speedup} | "
                             f"{stage['partial_count']} | {stage['scratch_bytes']} | {cooling}")
            lines.append("")
        lines.append("Inspect raw JSON for pack, metadata growth, attention, merge, errors, memory and all round values.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
