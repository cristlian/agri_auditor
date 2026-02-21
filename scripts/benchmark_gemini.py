from __future__ import annotations

import sys
from pathlib import Path

try:
    from agri_auditor.cli import main
except ModuleNotFoundError as exc:
    if exc.name == "agri_auditor":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from agri_auditor.cli import main
    else:
        raise


if __name__ == "__main__":
    raise SystemExit(main(["benchmark-gemini", *sys.argv[1:]]))