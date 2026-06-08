import pytest

from app.scripts import run_api_evaluation


def test_build_request_metrics_with_expected_sections() -> None:
    row = {
        "question": "Que descubre Zara?",
        "expected_sections": ["Ficción Espacial"],
    }
    diagnostics = {
        "cache_hit": False,
        "resolved_query": "Que descubre Zara en la ficcion espacial?",
        "stage_latencies_ms": {
            "embedding": 10.0,
            "retrieval": 20.0,
        },
        "retrieved_chunks": [
            {
                "id": "chunk-1",
                "content": "Ficción Espacial: Zara descubre un antiguo artefacto.",
                "similarity_score": 0.9,
                "metadata": {"section_title": "Ficción Espacial"},
            },
            {
                "id": "chunk-2",
                "content": "Cuento Corto: Emma recibe un dia extra.",
                "similarity_score": 0.5,
                "metadata": {"section_title": "Cuento Corto"},
            },
        ],
    }

    metrics = run_api_evaluation.build_request_metrics(
        row=row,
        answer="Zara descubre un antiguo artefacto.",
        diagnostics=diagnostics,
        latency_ms=120.0,
        recall_k=2,
        error=None,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["context_relevance"] == 0.5
    assert metrics["groundedness"] == 1.0
    assert metrics["latency_by_stage_ms"] == {
        "embedding": 10.0,
        "retrieval": 20.0,
    }
    assert metrics["error"] is False


def test_summarize_results_averages_numeric_metrics() -> None:
    results = [
        {
            "metrics": {
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "context_relevance": 0.5,
                "groundedness": 1.0,
                "answer_relevance": 0.5,
                "citation_accuracy": None,
                "latency_total_ms": 100.0,
                "latency_by_stage_ms": {"retrieval": 20.0},
                "estimated_tokens": 50,
                "error": False,
                "prompt_injection_detected": False,
                "cache_hit": False,
            },
            "error": None,
        },
        {
            "metrics": {
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "context_relevance": 0.0,
                "groundedness": 0.5,
                "answer_relevance": 0.25,
                "citation_accuracy": None,
                "latency_total_ms": 200.0,
                "latency_by_stage_ms": {"retrieval": 40.0},
                "estimated_tokens": 70,
                "error": True,
                "prompt_injection_detected": True,
                "cache_hit": True,
            },
            "error": "provider down",
        },
    ]

    summary = run_api_evaluation.summarize_results(
        dataset_name="sample",
        run_id="run-1",
        recall_k=4,
        results=results,
    )

    assert summary["requests_total"] == 2
    assert summary["error_rate"] == 0.5
    assert summary["recall_at_k"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["latency_total_ms_avg"] == 150.0
    assert summary["latency_by_stage_ms_avg"] == {"retrieval": 30.0}
    assert summary["estimated_tokens_total"] == 120
    assert summary["prompt_injection_attempts_detected"] == 1


def test_resolve_metrics_dir_uses_run_name_as_child_folder() -> None:
    metrics_dir = run_api_evaluation.resolve_metrics_dir(
        base_metrics_dir=run_api_evaluation.DEFAULT_METRICS_DIR,
        run_name=" cache test / rewrite ",
    )

    assert metrics_dir == (
        run_api_evaluation.DEFAULT_METRICS_DIR / "cache-test-rewrite"
    )


def test_resolve_metrics_dir_rejects_empty_sanitized_run_name() -> None:
    with pytest.raises(ValueError):
        run_api_evaluation.resolve_metrics_dir(
            base_metrics_dir=run_api_evaluation.DEFAULT_METRICS_DIR,
            run_name="///",
        )
