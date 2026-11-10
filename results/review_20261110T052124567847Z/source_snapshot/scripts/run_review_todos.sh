#!/usr/bin/env bash
# Run from any directory. Arguments are forwarded to review_todos.py.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON="$PWD/.venv/bin/python"
    else
        PYTHON=python3
    fi
fi
exec "$PYTHON" scripts/review_todos.py "$@"
