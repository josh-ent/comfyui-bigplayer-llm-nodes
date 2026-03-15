from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.comfyui_runtime import remove_containers_by_labels


def main() -> None:
    remove_containers_by_labels({"bigplayer.ui_test": "1"})


if __name__ == "__main__":
    main()
