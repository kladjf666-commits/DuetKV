"""Save recoverable source snapshots and incremental patches without touching git."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/phasekv_optimization_20261027"


def sources():
    for directory in ("PhaseKV", "models", "KIVI", "tests", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            if not set(path.parts) & {"official", "reference_snapshots", "__pycache__"}:
                yield path


def save(stage):
    destination = EVIDENCE / "stages" / stage
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {destination}")
    stages = sorted((EVIDENCE / "stages").glob("*"))
    previous = stages[-1] if stages else None
    hashes, patches = {}, []
    for path in sorted(sources()):
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        old = previous / "source" / relative if previous else None
        before = old.read_text(encoding="utf-8-sig").splitlines(True) if old and old.exists() else []
        after = data.decode("utf-8-sig").splitlines(True)
        if previous:
            patches.extend(difflib.unified_diff(before, after,
                fromfile="a/" + relative.as_posix(), tofile="b/" + relative.as_posix()))
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    if previous:
        with (destination / "changes.patch").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(patches).replace("\r\n", "\n"))
    print(json.dumps({"stage": stage, "files": len(hashes), "previous": str(previous)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    save(parser.parse_args().stage)
