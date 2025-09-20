"""CLI entry point for the patient-side constraint semantic parser."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Delegate to the patient compiler's existing argparse-based main."""
    from patient_compiler.compile_patient import main as _main
    _main()


if __name__ == "__main__":
    main()
