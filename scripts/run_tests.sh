#!/usr/bin/env bash
# scripts/run_tests.sh
# Run the full test suite for Reynolds-QSR with proper PYTHONPATH setup.

set -e  # exit on first error
set -o pipefail

# Resolve project root (one level up from scripts/)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Running tests with pytest..."
    pytest "$PROJECT_ROOT/tests" -v --color=yes --maxfail=1 --disable-warnings
else
    echo "pytest not found, running tests directly with Python..."
    # Run specific test if provided, otherwise run all tests
    if [ -n "$1" ]; then
        echo "Running test: $1"
        python "$1"
    else
        echo "Running all tests in tests/"
        test_count=0
        for test_file in tests/test_*.py; do
            echo "Running $test_file..."
            python "$test_file"
            test_count=$((test_count + 1))
        done
        echo "✅ All $test_count tests passed!"
    fi
fi