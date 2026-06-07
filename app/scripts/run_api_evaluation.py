import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_DATASET_DIR = Path("data/evaluation")
DEFAULT_ENDPOINT = "/api/questions"


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    dataset_paths = resolve_dataset_paths(args.datasets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"API base URL: {args.base_url}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Run id: {run_id}")
    print(f"Datasets: {', '.join(str(path) for path in dataset_paths)}")

    for dataset_path in dataset_paths:
        results = run_dataset(
            dataset_path=dataset_path,
            base_url=args.base_url,
            endpoint=args.endpoint,
            run_id=run_id,
            timeout_seconds=args.timeout,
            sleep_seconds=args.sleep,
            user_prefix=args.user_prefix,
        )
        output_path = output_dir / f"{dataset_path.stem}-{run_id}.jsonl"
        write_jsonl(output_path, results)
        print(f"Saved results: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run question datasets against the local RAG API.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
        help="API base URL. Defaults to API_BASE_URL or http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("API_QUESTIONS_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"Questions endpoint. Defaults to {DEFAULT_ENDPOINT}.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help=(
            "Dataset JSONL path. Can be passed multiple times. "
            "Defaults to every *.jsonl file in data/evaluation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATASET_DIR / "results"),
        help="Directory where result JSONL files will be written.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id. Defaults to current UTC timestamp.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout per request in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--user-prefix",
        default="eval",
        help="Prefix used to build isolated user names for each run.",
    )

    return parser.parse_args()


def resolve_dataset_paths(dataset_args: list[str] | None) -> list[Path]:
    if dataset_args:
        return [Path(dataset_arg) for dataset_arg in dataset_args]

    return sorted(DEFAULT_DATASET_DIR.glob("*.jsonl"))


def run_dataset(
    dataset_path: Path,
    base_url: str,
    endpoint: str,
    run_id: str,
    timeout_seconds: float,
    sleep_seconds: float,
    user_prefix: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    results: list[dict[str, Any]] = []

    print("")
    print(f"Running dataset: {dataset_path}")

    for index, row in enumerate(rows, start=1):
        user_name = build_user_name(
            user_prefix=user_prefix,
            run_id=run_id,
            dataset_name=dataset_path.stem,
            conversation=str(row.get("conversation", "default")),
        )
        payload = {
            "user_name": user_name,
            "question": row["question"],
        }

        started_at = time.perf_counter()
        response = post_json(
            url=join_url(base_url, endpoint),
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        answer = str(response.get("json", {}).get("answer", ""))
        result = {
            "dataset": dataset_path.stem,
            "index": index,
            "id": row.get("id"),
            "conversation": row.get("conversation"),
            "user_name": user_name,
            "question": row["question"],
            "status_code": response.get("status_code"),
            "latency_ms": latency_ms,
            "answer": answer,
            "error": response.get("error"),
            "raw_response": response.get("json"),
            "notes": row.get("notes"),
        }
        results.append(result)
        print_result(result)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return results


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                rows.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_json(
    url: str,
    payload: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status_code": response.status,
                "json": json.loads(body) if body else {},
            }
    except HTTPError as error:
        body = error.read().decode("utf-8")
        return {
            "ok": False,
            "status_code": error.code,
            "json": parse_json_or_text(body),
            "error": body,
        }
    except URLError as error:
        return {
            "ok": False,
            "status_code": None,
            "json": {},
            "error": str(error),
        }


def parse_json_or_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}

    if isinstance(payload, dict):
        return payload

    return {"value": payload}


def print_result(result: dict[str, Any]) -> None:
    status = "OK" if result["status_code"] and 200 <= result["status_code"] < 300 else "ERROR"
    print(
        f"[{status}] {result['id']} "
        f"status={result['status_code']} "
        f"latency={result['latency_ms']}ms"
    )
    print(f"  Q: {result['question']}")
    print(f"  A: {result['answer']}")


def join_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def build_user_name(
    user_prefix: str,
    run_id: str,
    dataset_name: str,
    conversation: str,
) -> str:
    return f"{user_prefix}-{run_id}-{dataset_name}-{conversation}"


if __name__ == "__main__":
    main()
