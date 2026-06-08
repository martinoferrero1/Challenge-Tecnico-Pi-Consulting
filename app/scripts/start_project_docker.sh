#!/usr/bin/env sh
set -eu

DETACHED="0"
BUILD="1"
FRONTEND="0"

IMAGE_NAME="challenge-rag-app"
API_CONTAINER_NAME="challenge-rag-api"
CHROMA_VOLUME_NAME="challenge_rag_chroma_data"

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd "$SCRIPT_DIR/../.." && pwd)
ENV_FILE="$REPO_ROOT/.env"
DOCUMENT_FILE="$REPO_ROOT/data/original_document.docx"

cd "$REPO_ROOT"

usage() {
    echo "Usage: sh app/scripts/start_project_docker.sh [--frontend] [--detached] [--no-build] [--down]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --detached|-d)
            DETACHED="1"
            shift
            ;;
        --frontend)
            FRONTEND="1"
            shift
            ;;
        --no-build)
            BUILD="0"
            shift
            ;;
        --down)
            docker rm -f "$API_CONTAINER_NAME" >/dev/null 2>&1 || true

            if [ -f "$ENV_FILE" ]; then
                docker compose --env-file "$ENV_FILE" down
            else
                docker compose down
            fi

            exit 0
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

if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not available. Start Docker and try again."
    exit 1
fi

echo "Repo root: $REPO_ROOT"

if [ ! -f "$ENV_FILE" ]; then
    echo "Missing .env file at $ENV_FILE"
    exit 1
fi

if [ ! -f "$DOCUMENT_FILE" ]; then
    echo "Missing source document at $DOCUMENT_FILE"
    echo "Expected relative container path from .env: ./data/original_document.docx"
    exit 1
fi

if [ "$FRONTEND" = "1" ]; then
    set -- docker compose --env-file "$ENV_FILE" up

    if [ "$BUILD" = "1" ]; then
        set -- "$@" --build
    fi

    if [ "$DETACHED" = "1" ]; then
        set -- "$@" --detach
    fi

    echo "Starting Docker Compose stack"
    echo "API: http://127.0.0.1:8000"
    echo "Frontend: http://127.0.0.1:8501"

    "$@"
    exit 0
fi

if [ "$BUILD" = "1" ]; then
    docker build -t "$IMAGE_NAME" .
fi

docker rm -f "$API_CONTAINER_NAME" >/dev/null 2>&1 || true

ENV_FILE_DOCKER="$ENV_FILE"
DATA_DIR_DOCKER="$REPO_ROOT/data"

if command -v cygpath >/dev/null 2>&1; then
    ENV_FILE_DOCKER=$(cygpath -w "$ENV_FILE")
    DATA_DIR_DOCKER=$(cygpath -w "$REPO_ROOT/data")
fi

set -- docker run

if [ "$DETACHED" = "1" ]; then
    set -- "$@" --detach
else
    set -- "$@" --rm
fi

set -- "$@" \
    --name "$API_CONTAINER_NAME" \
    --env-file "$ENV_FILE_DOCKER" \
    --env CHROMA_PERSIST_DIR=/app/.chroma \
    --publish 8000:8000 \
    --mount "type=bind,source=$DATA_DIR_DOCKER,target=/app/data,readonly" \
    --mount "type=volume,source=$CHROMA_VOLUME_NAME,target=/app/.chroma" \
    "$IMAGE_NAME" \
    sh -c "python -m app.scripts.index_document && uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo "Starting backend container"
echo "API: http://127.0.0.1:8000"

MSYS_NO_PATHCONV=1 "$@"