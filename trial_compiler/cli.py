"""CLI entry point for the trial-side constraint semantic parser."""
from __future__ import annotations

import sys
import os

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Delegate to the trial compiler's existing argparse-based main."""
    from trial_compiler.compile_trial import main as _main
    _main()


if __name__ == "__main__":
    main()
