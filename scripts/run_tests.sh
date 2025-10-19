#!/usr/bin/env bash
# scripts/run_tests.sh
# Run the full test suite for Reynolds-QSR with proper PYTHONPATH setup.

set -e  # exit on first error
set -o pipefail

# Resolve project root (one level up from scripts/)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT"

# Run pytest with options
pytest "$PROJECT_ROOT/tests" -v --color=yes --maxfail=1 --disable-warnings