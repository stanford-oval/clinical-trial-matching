#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== Running postprocess_siblings.py ==="
$PYTHON_BIN postprocess_siblings.py

echo "=== Running ingest_sibling_alternatives.py ==="
$PYTHON_BIN ingest_sibling_alternatives.py

echo "=== Running sibling_lift_materializer.py ==="
$PYTHON_BIN sibling_lift_materializer.py

echo "=== Running sibling_positive_lift_materializer.py ==="
$PYTHON_BIN sibling_positive_lift_materializer.py

echo "=== All sibling-side steps finished successfully ==="
