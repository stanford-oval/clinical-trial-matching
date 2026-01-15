"""CLI entry point for database execution (patient-trial matching)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Delegate to the matcher's existing argparse-based main."""
    from smt_matcher.match_patient_to_trial import main as _main
    _main()


if __name__ == "__main__":
    main()
