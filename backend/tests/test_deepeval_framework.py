"""Unit tests for the DeepEval 2-Pillar RAG Evaluation Framework."""

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from app.rag.evaluation import (
    get_all_rag_metrics,
    get_generation_metrics,
    get_retrieval_metrics,
)


class MockDeepEvalModel(DeepEvalBaseLLM):
    """Deterministic offline evaluator model for unit testing DeepEval metrics."""

    def __init__(self):
        super().__init__(model_name="mock-offline-evaluator")

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return '{"verdict": "yes", "score": 1.0, "reason": "Accurately grounded."}'

    async def a_generate(self, prompt: str) -> str:
        return '{"verdict": "yes", "score": 1.0, "reason": "Accurately grounded."}'

    def get_model_name(self) -> str:
        return "mock-offline-evaluator"


@pytest.fixture
def mock_eval_model():
    return MockDeepEvalModel()


def test_deepeval_retrieval_metrics_factory(mock_eval_model):
    """Verify Pillar 1 retrieval quality metrics are instantiated with proper thresholds."""
    metrics = get_retrieval_metrics(threshold=0.75, model=mock_eval_model)
    assert len(metrics) == 3
    types = [type(m) for m in metrics]
    assert ContextualRecallMetric in types
    assert ContextualPrecisionMetric in types
    assert ContextualRelevancyMetric in types
    for m in metrics:
        assert m.threshold == 0.75


def test_deepeval_generation_metrics_factory(mock_eval_model):
    """Verify Pillar 2 generation quality metrics are instantiated with proper thresholds."""
    metrics = get_generation_metrics(threshold=0.80, model=mock_eval_model)
    assert len(metrics) == 3
    types = [type(m) for m in metrics]
    assert FaithfulnessMetric in types
    assert HallucinationMetric in types
    assert AnswerRelevancyMetric in types
    for m in metrics:
        assert m.threshold == 0.80


def test_deepeval_all_rag_metrics_combined(mock_eval_model):
    """Verify combined factory returns all 6 pillars of the RAG Triad framework."""
    metrics = get_all_rag_metrics(threshold=0.70, model=mock_eval_model)
    assert len(metrics) == 6


def test_deepeval_test_case_construction():
    """Verify LLMTestCase handles RAG retrieval context and expected ground truth."""
    query = "How many days of paid vacation do employees receive?"
    actual_output = "Employees receive 25 days of annual paid vacation leave."
    expected_output = "25 days of paid annual vacation leave."
    retrieval_context = [
        "Company Policy: Employees receive 25 days of annual paid vacation leave. Requests require 14 days advance notice."
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
        context=retrieval_context,
    )

    assert test_case.input == query
    assert test_case.actual_output == actual_output
    assert test_case.retrieval_context == retrieval_context
    assert len(test_case.context) == 1


def test_golden_dataset_structure():
    """Verify GOLDEN_DATASET contains the required 4 ground truth benchmark test cases."""
    from app.rag.evaluation import GOLDEN_DATASET, get_golden_dataset

    assert len(GOLDEN_DATASET) == 4
    for golden in GOLDEN_DATASET:
        assert golden.input
        assert golden.expected_output
        assert len(golden.context) >= 1
        assert "scenario" in golden.additional_metadata

    dataset = get_golden_dataset()
    assert len(dataset.goldens) == 4


@pytest.mark.asyncio
async def test_rag_pipeline_evaluator_golden_dataset(mock_eval_model):
    """Verify RAGPipelineEvaluator executes Golden dataset test cases end-to-end."""
    from app.rag.evaluation import (
        MockDeepEvalModel,
        RAGPipelineEvaluator,
        setup_benchmark_pipeline,
    )

    pipeline, session_id = await setup_benchmark_pipeline(mock=True)
    evaluator = RAGPipelineEvaluator(pipeline=pipeline, threshold=0.7)

    results = await evaluator.evaluate_golden_dataset(
        session_id=session_id,
        metrics_type="retrieval",
        model=MockDeepEvalModel(),
    )

    assert len(results) == 4
    for r in results:
        assert r.input_query
        assert r.actual_output
        assert len(r.metric_scores) == 3
        assert r.overall_pass is True


def test_evaluation_mode_config_toggle(monkeypatch):
    """Verify EVALUATION_MODE can be configured directly from environment variables."""
    from app.core.config import Settings

    monkeypatch.setenv("EVALUATION_MODE", "mock")
    s_mock = Settings()
    assert s_mock.EVALUATION_MODE == "mock"

    monkeypatch.setenv("EVALUATION_MODE", "live")
    s_live = Settings()
    assert s_live.EVALUATION_MODE == "live"


@pytest.mark.asyncio
async def test_setup_benchmark_pipeline_default_from_config(monkeypatch):
    """Verify setup_benchmark_pipeline automatically defaults to mock when EVALUATION_MODE=mock."""
    from app.core.config import get_settings
    from app.providers.embeddings import MockEmbeddingProvider
    from app.rag.evaluation import setup_benchmark_pipeline

    monkeypatch.setenv("EVALUATION_MODE", "mock")
    get_settings.cache_clear()

    pipeline, session_id = await setup_benchmark_pipeline()
    assert isinstance(pipeline.retriever.embedding_provider, MockEmbeddingProvider)
    get_settings.cache_clear()


def test_deepeval_model_config_toggle(monkeypatch):
    """Verify DEEPEVAL_MODEL defaults to qwen/qwen3.8-27b and can be customized."""
    from app.core.config import Settings

    monkeypatch.delenv("DEEPEVAL_MODEL", raising=False)
    monkeypatch.delenv("EVAL_MODEL", raising=False)
    s_default = Settings(_env_file=None)
    assert s_default.DEEPEVAL_MODEL == "qwen/qwen3.8-27b"

    monkeypatch.setenv("DEEPEVAL_MODEL", "custom/judge-model")
    s_custom = Settings(_env_file=None)
    assert s_custom.DEEPEVAL_MODEL == "custom/judge-model"


@pytest.mark.asyncio
async def test_setup_benchmark_pipeline_streamlined_model_selection(monkeypatch):
    """Verify setup_benchmark_pipeline sets both extractor and reranker to the unified DEEPEVAL_MODEL."""
    from unittest.mock import AsyncMock, patch

    from app.core.config import get_settings
    from app.rag.evaluation import setup_benchmark_pipeline

    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setenv("DEEPEVAL_MODEL", "qwen/qwen3.8-27b")
    get_settings.cache_clear()

    with patch(
        "app.providers.embeddings.GeminiEmbeddingProvider.embed_text",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        pipeline, session_id = await setup_benchmark_pipeline(
            mock=False, model_name="qwen/qwen3.8-27b"
        )
        assert pipeline.synthesizer.llm_provider.model == "qwen/qwen3.8-27b"
        assert pipeline.reranker.model == "qwen/qwen3.8-27b"
    get_settings.cache_clear()
