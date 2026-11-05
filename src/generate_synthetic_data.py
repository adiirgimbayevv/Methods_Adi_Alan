"""Wrapper to regenerate synthetic benchmark logs (delegates to src/benchmark.py)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark()
