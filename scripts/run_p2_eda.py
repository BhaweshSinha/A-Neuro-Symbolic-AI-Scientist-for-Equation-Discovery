from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from nsai_scientist.data.eda.runner import (
    run_p2,
)


config = {
    "raw_root":
        ROOT / "data" / "raw" / "p1",

    "output_root":
        ROOT / "data" / "processed" / "p2",
}


if __name__ == "__main__":
    run_p2(config)