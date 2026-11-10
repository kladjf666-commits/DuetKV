"""Model-specific entry for all supported KV methods."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.compare_models import main
if __name__ == "__main__":
    raise SystemExit(main(default_profile=Path(__file__).resolve().parent.name))
