"""CLI entry point for SQL-based retrieval (objective-conditioned trial retrieval)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Delegate to the SQL retrieval composition driver."""
    from sql_retrieval.ops.constraint_retrieval import main as _main
    _main()


if __name__ == "__main__":
    main()
