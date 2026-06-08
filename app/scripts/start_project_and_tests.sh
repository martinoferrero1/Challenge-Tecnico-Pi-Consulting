#!/usr/bin/env sh
set -eu

PORT="8000"
SKIP_TESTS="0"
SKIP_INDEX="0"

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

usage() {
    echo "Usage: sh app/scripts/start_project_and_tests.sh [--port PORT] [--skip-tests] [--skip-index]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --port|-p)
            if [ "$#" -lt 2 ]; then
                usage
                exit 2
            fi
            PORT="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="1"
            shift
            ;;
        --skip-index)
            SKIP_INDEX="1"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "Using Python: $PYTHON"
echo "Repo root: $REPO_ROOT"

case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "Local Chroma indexing can segfault on Windows/Git Bash."
        echo "Use Docker with: sh app/scripts/start_project_docker.sh"
        echo "Or run this script from Linux/WSL."
        exit 1
        ;;
esac

if [ "$SKIP_TESTS" != "1" ]; then
    echo "Running tests..."
    "$PYTHON" -m pytest
fi

if [ "$SKIP_INDEX" != "1" ]; then
    echo "Indexing document..."
    "$PYTHON" -m app.scripts.index_document
fi

echo "Starting API on http://127.0.0.1:$PORT"
"$PYTHON" -m uvicorn app.main:app --reload --host 127.0.0.1 --port "$PORT"
