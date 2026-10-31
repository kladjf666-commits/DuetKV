"""Create an exact, reviewable incremental upload; no model/cache/history payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from save_phasekv_optimization_stage import sources, ROOT, EVIDENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "PhaseKV_optimization_20261027.zip")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing bundle")
    baseline = EVIDENCE / "stages/00_baseline"
    original = json.loads((baseline / "sha256.json").read_text(encoding="utf-8"))
    selected = {}
    for path in sources():
        relative = path.relative_to(ROOT).as_posix()
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if original.get(relative) != current:
            selected[path] = "new" if relative not in original else "modified"
    # The original scripts are imported by the new harness; keep their exact
    # source so the upload is usable against the established standalone layout.
    dependencies = ["scripts/benchmark_4090d.py", "scripts/compare_phasekv_same_round.py",
        "scripts/retest_phasekv_prompt.py", "scripts/sweep_phasekv.py",
        "scripts/save_phasekv_optimization_stage.py", "run_phasekv_same_round.sh",
        "tests/test_phasekv_cache.py", "tests/test_phasekv_launch_config.py", "tests/test_kivi_packed.py",
        "tests/test_qwen2_attention_adapter.py", "tests/test_llama_opt_adapters.py"]
    for relative in dependencies:
        selected.setdefault(ROOT / relative, "unchanged_dependency")
    for relative in ("PHASEKV_OPTIMIZATION_DELIVERY.txt", "run_phasekv_optimizations.sh"):
        selected[ROOT / relative] = "delivery"
    for path in baseline.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            selected[path] = "recoverable_baseline"
    for stage in (EVIDENCE / "stages").iterdir():
        for name in ("changes.patch", "sha256.json"):
            if (stage / name).is_file():
                selected[stage / name] = "stage_patch_or_hashes"
    for path in EVIDENCE.glob("*.json"):
        selected[path] = "current_task_evidence"
    for path in EVIDENCE.glob("*.txt"):
        selected[path] = "current_task_evidence"
    manifest = {"purpose": "Incremental overlay onto the matching PhaseKV baseline; not a standalone model bundle",
                "files": [], "excludes": ["weights", "model/download caches", "historical benchmark results"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, category in sorted(selected.items()):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            # Baseline bytes have already been captured. Refuse normalization
            # that would invalidate their recovery hashes rather than hiding it.
            data.decode("utf-8-sig")
            if b"\r\n" in data:
                if category == "recoverable_baseline":
                    raise ValueError(f"Baseline is not LF: {relative}")
                data = data.replace(b"\r\n", b"\n")
            if data.startswith(b"\xef\xbb\xbf"):
                data = data[3:]
            archive.writestr(relative, data)
            manifest["files"].append(dict(path=relative, category=category, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest()))
        encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        archive.writestr("OPTIMIZATION_BUNDLE_MANIFEST.json", encoded)
    # Re-open and verify the actual shipped bytes, not the input file list.
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for entry in manifest["files"]:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError("ZIP hash mismatch: " + entry["path"])
    (EVIDENCE / "bundle_manifest.json").write_bytes(encoded)
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"]),
                      "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
