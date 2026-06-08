import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_DATASET_DIR = Path("data/evaluation")
DEFAULT_ENDPOINT = "/api/questions/evaluation"
DEFAULT_METRICS_DIR = DEFAULT_DATASET_DIR / "metrics"
DEFAULT_RECALL_K = 4

WORD_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")
SECTION_TITLE_PATTERN = re.compile(r"^(?P<title>[^:\n]{1,120}):\s+")
CITATION_PATTERN = re.compile(r"\[(?P<citation>[^\]]+)\]")
PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\b.*\b(previous|system|developer)\b.*\binstruction", re.I),
    re.compile(r"\bdisregard\b.*\b(previous|system|developer)\b", re.I),
    re.compile(r"\b(system|developer)\s+prompt\b", re.I),
    re.compile(r"\breveal\b.*\b(prompt|instructions?)\b", re.I),
    re.compile(r"\bbypass\b.*\b(instructions?|policy|rules?)\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bprompt injection\b", re.I),
    re.compile(r"\bignora\b.*\b(instrucciones|reglas|sistema|desarrollador)\b", re.I),
    re.compile(r"\bolvida\b.*\b(instrucciones|reglas)\b", re.I),
    re.compile(r"\bmostra\b.*\b(prompt|instrucciones)\b", re.I),
)
DIRECTORY_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
STOPWORDS = {
    "a",
    "al",
    "and",
    "are",
    "as",
    "como",
    "con",
    "de",
    "del",
    "el",
    "ella",
    "en",
    "es",
    "esta",
    "estan",
    "este",
    "for",
    "is",
    "la",
    "las",
    "le",
    "lo",
    "los",
    "of",
    "o",
    "on",
    "por",
    "que",
    "se",
    "su",
    "the",
    "to",
    "un",
    "una",
    "y",
}


def main() -> None:
    """Ejecuta datasets contra la API y guarda resultados y métricas."""
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    dataset_paths = resolve_dataset_paths(args.datasets)
    output_dir = Path(args.output_dir)
    metrics_dir = resolve_metrics_dir(
        base_metrics_dir=Path(args.metrics_dir),
        run_name=args.metrics_run_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"API base URL: {args.base_url}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Run id: {run_id}")
    print(f"Recall K: {args.k}")
    print(f"Metrics dir: {metrics_dir}")
    print(f"Datasets: {', '.join(str(path) for path in dataset_paths)}")

    summaries: list[dict[str, Any]] = []
    for dataset_path in dataset_paths:
        results = run_dataset(
            dataset_path=dataset_path,
            base_url=args.base_url,
            endpoint=args.endpoint,
            run_id=run_id,
            timeout_seconds=args.timeout,
            sleep_seconds=args.sleep,
            user_prefix=args.user_prefix,
            recall_k=args.k,
        )
        summary = summarize_results(
            dataset_name=dataset_path.stem,
            run_id=run_id,
            recall_k=args.k,
            results=results,
        )
        summaries.append(summary)

        output_path = output_dir / f"{dataset_path.stem}-{run_id}.jsonl"
        summary_path = metrics_dir / f"{dataset_path.stem}-{run_id}-summary.json"
        prometheus_path = metrics_dir / f"{dataset_path.stem}-{run_id}.prom"

        write_jsonl(output_path, results)
        write_json(summary_path, summary)
        write_prometheus(prometheus_path, [summary])

        print(f"Saved results: {output_path}")
        print(f"Saved metrics summary: {summary_path}")
        print(f"Saved Prometheus metrics: {prometheus_path}")
        print_summary(summary)

    if summaries:
        run_summary_path = metrics_dir / f"run-{run_id}-summary.json"
        run_prometheus_path = metrics_dir / f"run-{run_id}.prom"
        write_json(run_summary_path, {"run_id": run_id, "datasets": summaries})
        write_prometheus(run_prometheus_path, summaries)
        print("")
        print(f"Saved run summary: {run_summary_path}")
        print(f"Saved run Prometheus metrics: {run_prometheus_path}")


def parse_args() -> argparse.Namespace:
    """Define y parsea argumentos de línea de comandos.

    Returns:
        Namespace con flags como ``base_url``, ``endpoint``, ``datasets``,
        ``output_dir``, ``metrics_dir`` y ``metrics_run_name``.
    """
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
        "--metrics-dir",
        default=str(DEFAULT_METRICS_DIR),
        help="Base directory where metrics summaries and .prom files will be written.",
    )
    parser.add_argument(
        "--metrics-run-name",
        help=(
            "Optional folder name inside --metrics-dir for this run's metrics. "
            "For example: --metrics-run-name cache-test."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id. Defaults to current UTC timestamp.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_RECALL_K,
        help=f"K used for Recall@K and MRR. Defaults to {DEFAULT_RECALL_K}.",
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
    """Resuelve los datasets explícitos o todos los disponibles.

    Args:
        dataset_args: Rutas recibidas por ``--dataset``.

    Returns:
        Lista de rutas JSONL a ejecutar.
    """
    if dataset_args:
        return [Path(dataset_arg) for dataset_arg in dataset_args]

    return sorted(DEFAULT_DATASET_DIR.glob("*.jsonl"))


def resolve_metrics_dir(base_metrics_dir: Path, run_name: str | None) -> Path:
    """Resuelve la carpeta final de métricas para una corrida.

    Args:
        base_metrics_dir: Carpeta base configurada por ``--metrics-dir``.
        run_name: Nombre opcional de subcarpeta para la corrida.

    Returns:
        Carpeta donde se escribirán resúmenes y archivos ``.prom``.
    """
    if run_name is None or not run_name.strip():
        return base_metrics_dir

    return base_metrics_dir / sanitize_directory_name(run_name)


def sanitize_directory_name(value: str) -> str:
    """Normaliza un nombre para usarlo como carpeta local.

    Args:
        value: Nombre crudo recibido por ``--metrics-run-name``.

    Returns:
        Nombre seguro para carpeta local.

    Raises:
        ValueError: Si ``value`` no contiene caracteres válidos.
    """
    sanitized = DIRECTORY_NAME_PATTERN.sub("-", value.strip())
    sanitized = sanitized.strip(".-_")

    if not sanitized:
        raise ValueError("Metrics run name must include at least one valid character")

    return sanitized


def run_dataset(
    dataset_path: Path,
    base_url: str,
    endpoint: str,
    run_id: str,
    timeout_seconds: float,
    sleep_seconds: float,
    user_prefix: str,
    recall_k: int,
) -> list[dict[str, Any]]:
    """Ejecuta ordenadamente las preguntas de un dataset JSONL.

    Args:
        dataset_path: Ruta del dataset JSONL.
        base_url: URL base de la API.
        endpoint: Endpoint de preguntas o evaluación.
        run_id: Identificador de corrida.
        timeout_seconds: Timeout HTTP por request.
        sleep_seconds: Pausa entre requests.
        user_prefix: Prefijo para construir usuarios aislados.
        recall_k: Valor K usado para métricas de retrieval.

    Returns:
        Lista de resultados enriquecidos con respuesta, diagnostics y métricas.
    """
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
        raw_response = response.get("json") or {}
        answer = str(raw_response.get("answer", ""))
        diagnostics = raw_response.get("diagnostics") or {}
        retrieved_chunks = diagnostics.get("retrieved_chunks") or []
        retrieved_sections = extract_retrieved_sections(retrieved_chunks)

        result = {
            "dataset": dataset_path.stem,
            "index": index,
            "id": row.get("id"),
            "conversation": row.get("conversation"),
            "user_name": user_name,
            "question": row["question"],
            "expected_sections": as_string_list(row.get("expected_sections")),
            "status_code": response.get("status_code"),
            "latency_ms": latency_ms,
            "answer": answer,
            "retrieved_sections": retrieved_sections,
            "diagnostics": diagnostics,
            "metrics": build_request_metrics(
                row=row,
                answer=answer,
                diagnostics=diagnostics,
                latency_ms=latency_ms,
                recall_k=recall_k,
                error=response.get("error"),
            ),
            "error": response.get("error"),
            "raw_response": raw_response,
            "notes": row.get("notes"),
        }
        results.append(result)
        print_result(result, recall_k)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return results


def build_request_metrics(
    row: dict[str, Any],
    answer: str,
    diagnostics: dict[str, Any],
    latency_ms: float,
    recall_k: int,
    error: Any,
) -> dict[str, Any]:
    """Calcula métricas por request a partir de respuesta y diagnostics.

    Args:
        row: Fila original del dataset.
        answer: Texto de respuesta devuelto por la API.
        diagnostics: Diagnóstico devuelto por el endpoint de evaluación.
        latency_ms: Latencia HTTP total medida por el script.
        recall_k: Valor K usado para Recall@K y MRR.
        error: Error HTTP o de conexión, si existió.

    Returns:
        Diccionario con métricas por request.
    """
    retrieved_chunks = diagnostics.get("retrieved_chunks") or []
    expected_sections = as_string_list(row.get("expected_sections"))
    retrieved_sections = extract_retrieved_sections(retrieved_chunks)
    context_text = "\n\n".join(str(chunk.get("content", "")) for chunk in retrieved_chunks)

    return {
        "recall_at_k": recall_at_k(expected_sections, retrieved_sections, recall_k),
        "mrr": mean_reciprocal_rank(expected_sections, retrieved_sections, recall_k),
        "context_relevance": context_relevance(
            question=row["question"],
            expected_sections=expected_sections,
            retrieved_chunks=retrieved_chunks,
            recall_k=recall_k,
        ),
        "groundedness": groundedness(answer=answer, context=context_text),
        "answer_relevance": answer_relevance(
            question=row["question"],
            answer=answer,
            expected_answer_terms=as_string_list(row.get("expected_answer_terms")),
        ),
        "citation_accuracy": citation_accuracy(
            answer=answer,
            expected_sections=expected_sections,
        ),
        "latency_total_ms": latency_ms,
        "latency_by_stage_ms": diagnostics.get("stage_latencies_ms") or {},
        "estimated_tokens": estimate_tokens(
            [
                row["question"],
                diagnostics.get("resolved_query"),
                answer,
                context_text,
            ]
        ),
        "error": bool(error),
        "prompt_injection_detected": prompt_injection_detected(row["question"]),
        "cache_hit": bool(diagnostics.get("cache_hit", False)),
    }


def recall_at_k(
    expected_sections: list[str],
    retrieved_sections: list[str],
    recall_k: int,
) -> float | None:
    """Calcula Recall@K sobre secciones esperadas y recuperadas.

    Args:
        expected_sections: Secciones relevantes esperadas por el dataset.
        retrieved_sections: Secciones recuperadas por el RAG.
        recall_k: Cantidad máxima de posiciones consideradas.

    Returns:
        Recall@K redondeado, o ``None`` si no hay secciones esperadas.
    """
    expected = normalized_set(expected_sections)
    if not expected:
        return None

    retrieved = normalized_set(retrieved_sections[:recall_k])
    return round(len(expected & retrieved) / len(expected), 4)


def mean_reciprocal_rank(
    expected_sections: list[str],
    retrieved_sections: list[str],
    recall_k: int,
) -> float | None:
    """Calcula MRR usando la primera sección relevante recuperada.

    Args:
        expected_sections: Secciones relevantes esperadas por el dataset.
        retrieved_sections: Secciones recuperadas por el RAG.
        recall_k: Cantidad máxima de posiciones consideradas.

    Returns:
        Reciprocal rank de la primera coincidencia, ``0.0`` si no aparece, o
        ``None`` si no hay secciones esperadas.
    """
    expected = normalized_set(expected_sections)
    if not expected:
        return None

    for rank, section in enumerate(retrieved_sections[:recall_k], start=1):
        if normalize_text(section) in expected:
            return round(1 / rank, 4)

    return 0.0


def context_relevance(
    question: str,
    expected_sections: list[str],
    retrieved_chunks: list[dict[str, Any]],
    recall_k: int,
) -> float | None:
    """Estima relevancia del contexto recuperado para la pregunta.

    Args:
        question: Pregunta del dataset.
        expected_sections: Secciones esperadas para la pregunta.
        retrieved_chunks: Chunks recuperados por la API de evaluación.
        recall_k: Cantidad máxima de chunks considerados.

    Returns:
        Score de relevancia de contexto o ``None`` si no puede calcularse.
    """
    retrieved_chunks_at_k = retrieved_chunks[:recall_k]
    if not retrieved_chunks_at_k:
        return 0.0

    if expected_sections:
        expected = normalized_set(expected_sections)
        relevant_count = sum(
            1
            for chunk in retrieved_chunks_at_k
            if normalize_text(extract_section_title(chunk)) in expected
        )
        return round(relevant_count / len(retrieved_chunks_at_k), 4)

    question_terms = content_terms(question)
    if not question_terms:
        return None

    scores = [
        lexical_overlap(question_terms, content_terms(str(chunk.get("content", ""))))
        for chunk in retrieved_chunks_at_k
    ]
    return round(sum(scores) / len(scores), 4)


def groundedness(answer: str, context: str) -> float | None:
    """Estima groundedness por solapamiento léxico respuesta-contexto.

    Args:
        answer: Respuesta generada por la API.
        context: Texto concatenado de chunks recuperados.

    Returns:
        Proporción de términos de ``answer`` presentes en ``context``.
    """
    answer_terms = content_terms(answer)
    if not answer_terms:
        return None

    context_terms = content_terms(context)
    if not context_terms:
        return 0.0

    return round(len(answer_terms & context_terms) / len(answer_terms), 4)


def answer_relevance(
    question: str,
    answer: str,
    expected_answer_terms: list[str],
) -> float | None:
    """Estima relevancia de respuesta contra pregunta o términos esperados.

    Args:
        question: Pregunta del dataset.
        answer: Respuesta generada por la API.
        expected_answer_terms: Términos esperados opcionales.

    Returns:
        Score de relevancia de respuesta o ``None`` si no hay términos útiles.
    """
    answer_terms = content_terms(answer)
    if not answer_terms:
        return None

    if expected_answer_terms:
        expected_terms = content_terms(" ".join(expected_answer_terms))
        if not expected_terms:
            return None
        return round(len(expected_terms & answer_terms) / len(expected_terms), 4)

    question_terms = content_terms(question)
    if not question_terms:
        return None

    return round(len(question_terms & answer_terms) / len(question_terms), 4)


def citation_accuracy(answer: str, expected_sections: list[str]) -> float | None:
    """Evalúa citas entre corchetes contra secciones esperadas.

    Args:
        answer: Respuesta generada por la API.
        expected_sections: Secciones esperadas por el dataset.

    Returns:
        Proporción de citas correctas, o ``None`` si no hay citas evaluables.
    """
    citations = [
        normalize_text(match.group("citation"))
        for match in CITATION_PATTERN.finditer(answer)
    ]
    if not citations:
        return None

    expected = normalized_set(expected_sections)
    if not expected:
        return None

    correct_count = sum(
        1
        for citation in citations
        if any(expected_section in citation or citation in expected_section for expected_section in expected)
    )
    return round(correct_count / len(citations), 4)


def prompt_injection_detected(question: str) -> bool:
    """Detecta patrones simples de prompt injection en la pregunta.

    Args:
        question: Pregunta enviada por el usuario/dataset.

    Returns:
        ``True`` si ``question`` matchea patrones sospechosos.
    """
    return any(pattern.search(question) for pattern in PROMPT_INJECTION_PATTERNS)


def estimate_tokens(values: list[Any]) -> int:
    """Estima tokens de forma aproximada usando longitud de caracteres.

    Args:
        values: Valores textuales que componen request, contexto y respuesta.

    Returns:
        Estimación aproximada de tokens.
    """
    text = " ".join(str(value or "") for value in values)
    if not text.strip():
        return 0

    return math.ceil(len(text) / 4)


def summarize_results(
    dataset_name: str,
    run_id: str,
    recall_k: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Agrega métricas de requests en un resumen por dataset.

    Args:
        dataset_name: Nombre del dataset ejecutado.
        run_id: Identificador de corrida.
        recall_k: Valor K usado para métricas de retrieval.
        results: Resultados individuales generados por ``run_dataset``.

    Returns:
        Resumen agregado del dataset.
    """
    request_metrics = [result["metrics"] for result in results]
    success_count = sum(1 for result in results if not result.get("error"))
    error_count = len(results) - success_count
    latency_values = [metric["latency_total_ms"] for metric in request_metrics]

    return {
        "dataset": dataset_name,
        "run_id": run_id,
        "recall_k": recall_k,
        "requests_total": len(results),
        "success_count": success_count,
        "error_count": error_count,
        "error_rate": round(error_count / len(results), 4) if results else 0.0,
        "cache_hit_rate": average_metric(request_metrics, "cache_hit"),
        "recall_at_k": average_metric(request_metrics, "recall_at_k"),
        "mrr": average_metric(request_metrics, "mrr"),
        "context_relevance": average_metric(request_metrics, "context_relevance"),
        "groundedness": average_metric(request_metrics, "groundedness"),
        "answer_relevance": average_metric(request_metrics, "answer_relevance"),
        "citation_accuracy": average_metric(request_metrics, "citation_accuracy"),
        "latency_total_ms_avg": average(latency_values),
        "latency_total_ms_p50": percentile(latency_values, 50),
        "latency_total_ms_p95": percentile(latency_values, 95),
        "latency_by_stage_ms_avg": average_stage_latencies(request_metrics),
        "estimated_tokens_total": sum(metric["estimated_tokens"] for metric in request_metrics),
        "estimated_tokens_avg": average(
            [metric["estimated_tokens"] for metric in request_metrics]
        ),
        "prompt_injection_attempts_detected": sum(
            1 for metric in request_metrics if metric["prompt_injection_detected"]
        ),
    }


def average_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    """Promedia una métrica numérica ignorando valores nulos.

    Args:
        metrics: Lista de diccionarios de métricas por request.
        key: Nombre de la métrica a promediar.

    Returns:
        Promedio de valores numéricos o ``None``.
    """
    values = [metric.get(key) for metric in metrics]
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, bool | int | float)
    ]
    return average(numeric_values)


def average(values: list[int | float]) -> float | None:
    """Calcula promedio redondeado o None si no hay valores.

    Args:
        values: Valores numéricos a promediar.

    Returns:
        Promedio redondeado o ``None`` si ``values`` está vacío.
    """
    if not values:
        return None

    return round(sum(values) / len(values), 4)


def percentile(values: list[int | float], percentile_value: int) -> float | None:
    """Calcula un percentil simple sobre una lista de valores.

    Args:
        values: Valores numéricos de entrada.
        percentile_value: Percentil deseado, por ejemplo 50 o 95.

    Returns:
        Valor del percentil o ``None`` si no hay datos.
    """
    if not values:
        return None

    sorted_values = sorted(values)
    index = math.ceil((percentile_value / 100) * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return round(float(sorted_values[index]), 4)


def average_stage_latencies(metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Promedia latencias agrupadas por etapa del RAG.

    Args:
        metrics: Métricas por request con ``latency_by_stage_ms``.

    Returns:
        Diccionario de latencias promedio por nombre de etapa.
    """
    stage_values: dict[str, list[float]] = {}

    for metric in metrics:
        stage_latencies = metric.get("latency_by_stage_ms") or {}
        for stage, value in stage_latencies.items():
            if isinstance(value, int | float):
                stage_values.setdefault(stage, []).append(float(value))

    return {
        stage: average(values) or 0.0
        for stage, values in sorted(stage_values.items())
    }


def extract_retrieved_sections(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Extrae títulos de sección desde chunks recuperados.

    Args:
        retrieved_chunks: Chunks devueltos por diagnostics.

    Returns:
        Lista de títulos o identificadores de sección.
    """
    return [extract_section_title(chunk) for chunk in retrieved_chunks]


def extract_section_title(chunk: dict[str, Any]) -> str:
    """Obtiene el título de sección desde metadata, contenido o id.

    Args:
        chunk: Chunk serializado en diagnostics.

    Returns:
        Título de sección si existe, o fallback al id del chunk.
    """
    metadata = chunk.get("metadata") or {}
    section_title = metadata.get("section_title")
    if section_title:
        return str(section_title)

    content = str(chunk.get("content", "")).strip()
    title_match = SECTION_TITLE_PATTERN.match(content)
    if title_match:
        return title_match.group("title").strip()

    return str(chunk.get("id", "unknown"))


def lexical_overlap(left_terms: set[str], right_terms: set[str]) -> float:
    """Calcula solapamiento léxico direccional entre conjuntos.

    Args:
        left_terms: Términos base usados como denominador.
        right_terms: Términos contra los cuales se compara.

    Returns:
        Proporción de ``left_terms`` presentes en ``right_terms``.
    """
    if not left_terms:
        return 0.0

    return len(left_terms & right_terms) / len(left_terms)


def content_terms(text: str) -> set[str]:
    """Tokeniza texto y remueve stopwords simples.

    Args:
        text: Texto a tokenizar.

    Returns:
        Conjunto de términos normalizados y filtrados.
    """
    return {
        normalize_text(match.group(0))
        for match in WORD_PATTERN.finditer(text)
        if normalize_text(match.group(0)) not in STOPWORDS
    }


def normalized_set(values: list[str]) -> set[str]:
    """Normaliza una lista de strings como conjunto.

    Args:
        values: Strings a normalizar.

    Returns:
        Conjunto de strings normalizados no vacíos.
    """
    return {normalize_text(value) for value in values if normalize_text(value)}


def normalize_text(value: str) -> str:
    """Normaliza texto para comparaciones estables.

    Args:
        value: Texto o valor convertible a texto.

    Returns:
        Texto case-insensitive, sin espacios externos ni repetidos.
    """
    return " ".join(str(value).strip().casefold().split())


def as_string_list(value: Any) -> list[str]:
    """Convierte un valor arbitrario a lista de strings no vacíos.

    Args:
        value: Valor de dataset que puede ser ``None``, string, lista u otro.

    Returns:
        Lista de strings no vacíos.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    return [str(value)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee un archivo JSONL validando cada línea no vacía.

    Args:
        path: Ruta del archivo JSONL.

    Returns:
        Lista de filas parseadas como diccionarios.

    Raises:
        ValueError: Si alguna línea no vacía no es JSON válido.
    """
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
    """Escribe filas como JSONL UTF-8.

    Args:
        path: Ruta de salida.
        rows: Filas a escribir, una por línea.
    """
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe un JSON indentado UTF-8.

    Args:
        path: Ruta de salida.
        payload: Objeto serializable como JSON.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_prometheus(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Escribe métricas agregadas en formato Prometheus exposition.

    Args:
        path: Ruta del archivo ``.prom``.
        summaries: Resúmenes agregados por dataset.
    """
    lines = [
        "# HELP rag_eval_requests_total Total requests in a RAG evaluation run.",
        "# TYPE rag_eval_requests_total gauge",
    ]

    for summary in summaries:
        labels = prometheus_labels(summary)
        lines.append(f"rag_eval_requests_total{labels} {summary['requests_total']}")
        lines.append(f"rag_eval_errors_total{labels} {summary['error_count']}")
        lines.append(f"rag_eval_error_rate{labels} {summary['error_rate']}")
        lines.append(
            "rag_eval_prompt_injection_attempts_detected_total"
            f"{labels} {summary['prompt_injection_attempts_detected']}"
        )
        lines.append(
            f"rag_eval_tokens_estimated_total{labels} "
            f"{summary['estimated_tokens_total']}"
        )
        write_optional_metric(lines, "rag_eval_cache_hit_rate", labels, summary["cache_hit_rate"])
        write_optional_metric(lines, "rag_eval_recall_at_k", labels, summary["recall_at_k"])
        write_optional_metric(lines, "rag_eval_mrr", labels, summary["mrr"])
        write_optional_metric(
            lines,
            "rag_eval_context_relevance",
            labels,
            summary["context_relevance"],
        )
        write_optional_metric(lines, "rag_eval_groundedness", labels, summary["groundedness"])
        write_optional_metric(
            lines,
            "rag_eval_answer_relevance",
            labels,
            summary["answer_relevance"],
        )
        write_optional_metric(
            lines,
            "rag_eval_citation_accuracy",
            labels,
            summary["citation_accuracy"],
        )
        write_optional_metric(
            lines,
            "rag_eval_latency_seconds_avg",
            labels,
            ms_to_seconds(summary["latency_total_ms_avg"]),
        )
        write_optional_metric(
            lines,
            "rag_eval_latency_seconds_p50",
            labels,
            ms_to_seconds(summary["latency_total_ms_p50"]),
        )
        write_optional_metric(
            lines,
            "rag_eval_latency_seconds_p95",
            labels,
            ms_to_seconds(summary["latency_total_ms_p95"]),
        )

        for stage, latency_ms in summary["latency_by_stage_ms_avg"].items():
            stage_labels = append_prometheus_label(labels, "stage", stage)
            write_optional_metric(
                lines,
                "rag_eval_stage_latency_seconds_avg",
                stage_labels,
                ms_to_seconds(latency_ms),
            )

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")


def write_optional_metric(
    lines: list[str],
    metric_name: str,
    labels: str,
    value: float | None,
) -> None:
    """Agrega una métrica Prometheus solo si tiene valor.

    Args:
        lines: Líneas acumuladas del archivo ``.prom``.
        metric_name: Nombre de la métrica Prometheus.
        labels: Labels ya serializados.
        value: Valor numérico opcional.
    """
    if value is None:
        return

    lines.append(f"{metric_name}{labels} {value}")


def prometheus_labels(summary: dict[str, Any]) -> str:
    """Construye labels Prometheus de baja cardinalidad.

    Args:
        summary: Resumen agregado de dataset.

    Returns:
        Labels Prometheus serializados.
    """
    labels = {
        "dataset": summary["dataset"],
        "run_id": summary["run_id"],
        "k": str(summary["recall_k"]),
    }
    return "{%s}" % ",".join(
        f'{key}="{escape_prometheus_label(value)}"'
        for key, value in labels.items()
    )


def append_prometheus_label(labels: str, key: str, value: str) -> str:
    """Agrega un label adicional a un bloque de labels Prometheus.

    Args:
        labels: Bloque de labels existente.
        key: Nombre del label a agregar.
        value: Valor del label a agregar.

    Returns:
        Bloque de labels con ``key`` y ``value`` incluidos.
    """
    if labels == "{}":
        return f'{{{key}="{escape_prometheus_label(value)}"}}'

    return labels[:-1] + f',{key}="{escape_prometheus_label(value)}"' + "}"


def escape_prometheus_label(value: str) -> str:
    """Escapa caracteres especiales de labels Prometheus.

    Args:
        value: Valor crudo de label.

    Returns:
        Valor escapado para formato Prometheus.
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def ms_to_seconds(value: float | None) -> float | None:
    """Convierte milisegundos a segundos para métricas Prometheus.

    Args:
        value: Latencia en milisegundos.

    Returns:
        Latencia en segundos o ``None``.
    """
    if value is None:
        return None

    return round(value / 1000, 6)


def post_json(
    url: str,
    payload: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Envía un POST JSON y devuelve respuesta o error controlado.

    Args:
        url: URL completa a invocar.
        payload: Body JSON a enviar.
        timeout_seconds: Timeout HTTP en segundos.

    Returns:
        Diccionario con status, JSON parseado y error si corresponde.
    """
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
    """Parsea JSON o encapsula texto plano.

    Args:
        text: Body HTTP recibido como string.

    Returns:
        Diccionario parseado o wrapper con ``text``/``value``.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}

    if isinstance(payload, dict):
        return payload

    return {"value": payload}


def print_result(result: dict[str, Any], recall_k: int) -> None:
    """Imprime una fila de resultado en formato legible.

    Args:
        result: Resultado individual de ``run_dataset``.
        recall_k: Valor K mostrado junto a recall.
    """
    status = "OK" if result["status_code"] and 200 <= result["status_code"] < 300 else "ERROR"
    metrics = result["metrics"]
    metric_parts = [
        format_metric(f"recall@{recall_k}", metrics["recall_at_k"]),
        format_metric("mrr", metrics["mrr"]),
        format_metric("ctx", metrics["context_relevance"]),
        format_metric("grounded", metrics["groundedness"]),
        format_metric("ans_rel", metrics["answer_relevance"]),
    ]
    metric_text = " ".join(part for part in metric_parts if part)

    print(
        f"[{status}] {result['id']} "
        f"status={result['status_code']} "
        f"latency={result['latency_ms']}ms "
        f"{metric_text}"
    )
    print(f"  Q: {result['question']}")
    print(f"  A: {result['answer']}")


def print_summary(summary: dict[str, Any]) -> None:
    """Imprime el resumen agregado de un dataset.

    Args:
        summary: Resumen agregado producido por ``summarize_results``.
    """
    print("")
    print(f"Summary: {summary['dataset']}")
    print(f"  requests={summary['requests_total']} error_rate={summary['error_rate']}")
    print(f"  recall@{summary['recall_k']}={summary['recall_at_k']} mrr={summary['mrr']}")
    print(
        "  relevance="
        f"{summary['context_relevance']} "
        f"groundedness={summary['groundedness']} "
        f"answer_relevance={summary['answer_relevance']}"
    )
    print(
        "  latency_ms="
        f"avg:{summary['latency_total_ms_avg']} "
        f"p50:{summary['latency_total_ms_p50']} "
        f"p95:{summary['latency_total_ms_p95']}"
    )
    print(
        "  tokens="
        f"total:{summary['estimated_tokens_total']} "
        f"avg:{summary['estimated_tokens_avg']}"
    )


def format_metric(name: str, value: Any) -> str:
    """Formatea una métrica para consola si tiene valor.

    Args:
        name: Nombre corto de la métrica.
        value: Valor de la métrica.

    Returns:
        Texto ``name=value`` o string vacío si ``value`` es ``None``.
    """
    if value is None:
        return ""

    return f"{name}={value}"


def join_url(base_url: str, endpoint: str) -> str:
    """Une base URL y endpoint evitando barras duplicadas.

    Args:
        base_url: URL base de la API.
        endpoint: Ruta del endpoint.

    Returns:
        URL completa.
    """
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def build_user_name(
    user_prefix: str,
    run_id: str,
    dataset_name: str,
    conversation: str,
) -> str:
    """Construye un usuario único por corrida, dataset y conversación.

    Args:
        user_prefix: Prefijo configurable para usuarios de evaluación.
        run_id: Identificador de corrida.
        dataset_name: Nombre del dataset actual.
        conversation: Identificador lógico de conversación dentro del dataset.

    Returns:
        ``user_name`` aislado para evitar mezclar historiales entre corridas.
    """
    return f"{user_prefix}-{run_id}-{dataset_name}-{conversation}"


if __name__ == "__main__":
    main()
