"""Deprecated: use scripts/cool_to_sqr.py instead."""

from pathlib import Path
import subprocess
import sys

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cool_to_sqr.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, str(_SCRIPT)], check=True)
