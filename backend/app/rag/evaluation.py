"""DeepEval RAG Pipeline Benchmark and Evaluation Runner.

Implements the 2-Pillar RAG Evaluation Framework:
- Pillar 1 (Retrieval Quality): ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
- Pillar 2 (Generation Quality): FaithfulnessMetric, HallucinationMetric, AnswerRelevancyMetric

Includes:
- Curated GOLDEN_DATASET with ground truth test cases across key enterprise RAG scenarios.
- DeepEval EvaluationDataset integration.
- GroqDeepEvalModel (powered by Groq LLaMA-3.3-70B / openai/gpt-oss-120b).
- MockDeepEvalModel for deterministic offline evaluations.
- CLI runner for terminal benchmarking and evaluation reports.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from deepeval.dataset import EvaluationDataset, Golden
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

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GroqDeepEvalModel(DeepEvalBaseLLM):
    """DeepEval evaluator model powered by Groq ChatGroq wrapper with auditing metadata."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.DEEPEVAL_MODEL
        self._api_key = settings.GROQ_API_KEY
        super().__init__(model_name=self._model_name)

    def load_model(self):
        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=self._api_key,
            model_name=self._model_name,
            temperature=0.0,
        )

    def generate(self, prompt: str) -> str:
        max_retries = 5
        delay = 4.0
        config = {
            "tags": ["deepeval", "judge", f"model:{self._model_name}"],
            "metadata": {
                "component": "deepeval_judge",
                "model": self._model_name,
                "framework": "langchain_groq",
            },
        }
        for attempt in range(max_retries):
            try:
                res = self.model.invoke(prompt, config=config)
                return str(res.content)
            except Exception as exc:
                err_str = str(exc).lower()
                if (
                    any(
                        k in err_str
                        for k in (
                            "rate_limit",
                            "rate limit",
                            "429",
                            "tpm",
                            "rpm",
                            "too many requests",
                        )
                    )
                    and attempt < max_retries - 1
                ):
                    logger.warning(
                        "Groq rate limit encountered during evaluation (attempt %d/%d). Sleeping %.1fs...",
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    async def a_generate(self, prompt: str) -> str:
        max_retries = 5
        delay = 4.0
        config = {
            "tags": ["deepeval", "judge", f"model:{self._model_name}"],
            "metadata": {
                "component": "deepeval_judge",
                "model": self._model_name,
                "framework": "langchain_groq",
            },
        }
        for attempt in range(max_retries):
            try:
                res = await self.model.ainvoke(prompt, config=config)
                return str(res.content)
            except Exception as exc:
                err_str = str(exc).lower()
                if (
                    any(
                        k in err_str
                        for k in (
                            "rate_limit",
                            "rate limit",
                            "429",
                            "tpm",
                            "rpm",
                            "too many requests",
                        )
                    )
                    and attempt < max_retries - 1
                ):
                    logger.warning(
                        "Groq rate limit encountered during evaluation (attempt %d/%d). Sleeping %.1fs...",
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise

    def get_model_name(self) -> str:
        return self._model_name


class MockDeepEvalModel(DeepEvalBaseLLM):
    """Deterministic offline evaluator model for unit testing and offline benchmarking."""

    def __init__(self):
        super().__init__(model_name="mock-offline-evaluator")

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "statements": ["The answer directly addresses the query."],
                "truths": ["Reference context contains the required factual grounding."],
                "claims": ["Reference context contains the required factual grounding."],
                "verdicts": [
                    {
                        "verdict": "yes",
                        "statement": "The answer is factually grounded.",
                        "reason": "Accurately grounded in provided context.",
                    }
                ],
                "verdict": "yes",
                "score": 1.0,
                "reason": "Accurately grounded in provided context.",
            }
        )

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "mock-offline-evaluator"


def _get_default_judge_model() -> DeepEvalBaseLLM:
    """Return appropriate DeepEval judge model based on EVALUATION_MODE (mock vs live)."""
    settings = get_settings()
    if settings.EVALUATION_MODE == "mock" or not settings.GROQ_API_KEY:
        return MockDeepEvalModel()
    return GroqDeepEvalModel(model_name=settings.DEEPEVAL_MODEL)


def get_retrieval_metrics(threshold: float = 0.7, model: Any | None = None) -> list:
    """Return Pillar 1 Retrieval Quality metrics (Recall, Precision, Relevancy)."""
    eval_model = model or _get_default_judge_model()
    kwargs: dict = {"threshold": threshold, "include_reason": True, "model": eval_model}
    return [
        ContextualRecallMetric(**kwargs),
        ContextualPrecisionMetric(**kwargs),
        ContextualRelevancyMetric(**kwargs),
    ]


def get_generation_metrics(threshold: float = 0.7, model: Any | None = None) -> list:
    """Return Pillar 2 Generation Quality metrics (Faithfulness, Hallucination, Relevancy)."""
    eval_model = model or _get_default_judge_model()
    kwargs: dict = {"threshold": threshold, "include_reason": True, "model": eval_model}
    return [
        FaithfulnessMetric(**kwargs),
        HallucinationMetric(**kwargs),
        AnswerRelevancyMetric(**kwargs),
    ]


def get_all_rag_metrics(threshold: float = 0.7, model: Any | None = None) -> list:
    """Return all Pillar 1 and Pillar 2 evaluation metrics combined."""
    return get_retrieval_metrics(threshold=threshold, model=model) + get_generation_metrics(
        threshold=threshold, model=model
    )


@dataclass
class MetricScore:
    name: str
    score: float
    passed: bool
    reason: str | None = None


@dataclass
class SampleEvaluationResult:
    input_query: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str]
    metric_scores: list[MetricScore]
    overall_pass: bool
    scenario: str | None = None


# ==============================================================================
# CANONICAL GOLDEN DATASET FOR RAG EVALUATION
# ==============================================================================

GOLDEN_DATASET: list[Golden] = [
    Golden(
        input="Under what specific uptime and notice conditions can an enterprise customer terminate the agreement without penalty?",
        expected_output="The customer may terminate the agreement without penalty by providing 45 calendar days of written notice if system uptime falls below 99.5% for two consecutive billing quarters.",
        context=[
            "Section 9: Contract Termination and Severance. Clause 9.1: The customer may terminate this agreement without penalty by providing 45 calendar days written notice if uptime falls below 99.5% for two consecutive billing quarters.",
            "Clause 9.2: Early termination without breach requires payment of 25% of the remaining annual contract value.",
        ],
        additional_metadata={
            "scenario": "policy_clause_lookup",
            "domain": "Enterprise SLA",
            "difficulty": "medium",
        },
    ),
    Golden(
        input="What are the scheduled maintenance hours and what advance notice is required for deploying emergency patches?",
        expected_output="Routine maintenance occurs every second Saturday from 02:00 to 06:00 UTC, while emergency patches can be deployed with a 2-hour notification window.",
        context=[
            "Section 4: Maintenance Windows and Downtime. Routine maintenance occurs every second Saturday from 02:00 to 06:00 UTC. Emergency patches can be deployed with a 2-hour notification window."
        ],
        additional_metadata={
            "scenario": "multi_constraint_temporal",
            "domain": "IT Operations",
            "difficulty": "easy",
        },
    ),
    Golden(
        input="Where is customer data stored geographically and under what compliance certification?",
        expected_output="Customer data resides exclusively in EU-West (Frankfurt) and US-East (Virginia) data centers under ISO-27001 certification.",
        context=[
            "Section 12: Data Residency and Compliance. Customer data resides exclusively in EU-West (Frankfurt) and US-East (Virginia) data centers under ISO-27001 certification."
        ],
        additional_metadata={
            "scenario": "compliance_governance",
            "domain": "Information Security",
            "difficulty": "medium",
        },
    ),
    Golden(
        input="What is the monetary penalty fee if an engineer fails to reply to a support ticket within 15 minutes?",
        expected_output="The provided knowledge base does not contain information regarding monetary penalty fees for 15-minute support ticket response times.",
        context=[
            "Section 4: Maintenance Windows and Downtime. Routine maintenance occurs every second Saturday from 02:00 to 06:00 UTC.",
            "Section 9: Contract Termination and Severance. Clause 9.1: Termination without penalty requires 45 calendar days notice if uptime falls below 99.5%.",
        ],
        additional_metadata={
            "scenario": "zero_hallucination_refusal",
            "domain": "Customer Support",
            "difficulty": "hard",
        },
    ),
]

# Backward compatibility alias
BENCHMARK_SUITE = [
    {
        "query": g.input,
        "expected_output": g.expected_output,
        "context": g.context,
    }
    for g in GOLDEN_DATASET
]


def get_golden_dataset() -> EvaluationDataset:
    """Return the canonical Golden evaluation dataset as a DeepEval EvaluationDataset instance."""
    return EvaluationDataset(goldens=GOLDEN_DATASET)


# ==============================================================================
# BENCHMARK PIPELINE FACTORY
# ==============================================================================


async def setup_benchmark_pipeline(
    mock: bool | None = None,
    session_id: str = "benchmark_golden_session",
    model_name: str | None = None,
) -> tuple[Any, str]:
    """Build and seed an isolated in-memory RAG pipeline containing the reference documents."""
    import chromadb

    from app.providers.embeddings import GeminiEmbeddingProvider, MockEmbeddingProvider
    from app.providers.llm import LangChainLLMProvider, MockLLMProvider
    from app.providers.vector_store import ChromaVectorStore, IndexedChunk
    from app.rag.pipeline import RAGPipeline
    from app.rag.reranker import GroqLLMReranker, ReciprocalRankReranker
    from app.rag.retriever import HybridRetriever
    from app.rag.synthesizer import GroundedSynthesizer

    settings = get_settings()
    is_mock = (settings.EVALUATION_MODE == "mock") if mock is None else mock
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())

    if is_mock:
        embedding_provider = MockEmbeddingProvider()
        llm_provider = MockLLMProvider()
        reranker = ReciprocalRankReranker(rrf_k=60)
    else:
        target_model = model_name or settings.DEEPEVAL_MODEL
        embedding_provider = GeminiEmbeddingProvider(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_EMBEDDING_MODEL,
        )
        llm_provider = LangChainLLMProvider(
            provider="groq",
            api_key=settings.GROQ_API_KEY,
            model=target_model,
        )
        reranker = GroqLLMReranker(
            groq_api_key=settings.GROQ_API_KEY,
            model=target_model,
            max_candidates=15,
        )

    # Seed the reference document chunks
    reference_passages = [
        (
            "doc_sla",
            "Enterprise Software Service Level Agreement (SLA)",
            0,
            "Section 4: Maintenance Windows and Downtime. Routine maintenance occurs every second Saturday from 02:00 to 06:00 UTC. Emergency patches can be deployed with a 2-hour notification window.",
        ),
        (
            "doc_sla",
            "Enterprise Software Service Level Agreement (SLA)",
            1,
            "Section 9: Contract Termination and Severance. Clause 9.1: The customer may terminate this agreement without penalty by providing 45 calendar days written notice if uptime falls below 99.5% for two consecutive billing quarters. Clause 9.2: Early termination without breach requires payment of 25% of the remaining annual contract value.",
        ),
        (
            "doc_sla",
            "Enterprise Software Service Level Agreement (SLA)",
            2,
            "Section 12: Data Residency and Compliance. Customer data resides exclusively in EU-West (Frankfurt) and US-East (Virginia) data centers under ISO-27001 certification.",
        ),
        (
            "doc_it",
            "Corporate IT & Equipment Policy",
            0,
            "Section 3: Hardware Refresh Cycle. Company laptops are refreshed every three years for full-time employees, and damaged devices must be reported to IT within 24 hours.",
        ),
        (
            "doc_hr",
            "Global Human Resources Leave Policy",
            0,
            "Section 2: Annual Vacation Allowance. Full-time employees receive 25 days of annual paid vacation leave. Leave requests exceeding 3 consecutive business days require 14 calendar days advance notice.",
        ),
    ]

    indexed_chunks = []
    for doc_id, title, c_idx, content in reference_passages:
        emb = await embedding_provider.embed_text(content)
        indexed_chunks.append(
            IndexedChunk(
                chunk_id=f"chk_{doc_id}_{c_idx}",
                document_id=doc_id,
                document_title=title,
                chunk_index=c_idx,
                content=content,
                embedding=emb,
                session_id=session_id,
                user_id="eval_user",
                file_name=f"{doc_id}.pdf",
                page_number=1,
            )
        )
    await vector_store.add_chunks(indexed_chunks)

    retriever = HybridRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    synthesizer = GroundedSynthesizer(llm_provider=llm_provider)
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        synthesizer=synthesizer,
        top_k=10,
        top_n=3,
        similarity_threshold=0.15,
    )

    return pipeline, session_id


# ==============================================================================
# RAG PIPELINE EVALUATOR
# ==============================================================================


class RAGPipelineEvaluator:
    """Evaluates RAG pipeline stages using DeepEval's 2-pillar quality framework."""

    def __init__(self, pipeline: Any, threshold: float = 0.7):
        self.pipeline = pipeline
        self.threshold = threshold

    async def evaluate_test_case(
        self,
        golden: Golden,
        session_id: str | None = None,
        metrics_type: str = "all",
        model: Any | None = None,
    ) -> SampleEvaluationResult:
        """Run a single Golden test case through the pipeline and evaluate with DeepEval."""
        response = await self.pipeline.run(question=golden.input, session_id=session_id)
        retrieval_context = [s.snippet for s in response.sources]
        ground_truth_context = golden.context or retrieval_context

        test_case = LLMTestCase(
            input=golden.input,
            actual_output=response.answer,
            expected_output=golden.expected_output,
            retrieval_context=retrieval_context if retrieval_context else ["No context retrieved"],
            context=ground_truth_context if ground_truth_context else ["No context retrieved"],
        )

        scenario = (golden.additional_metadata or {}).get("scenario")
        if metrics_type == "retrieval":
            metrics = get_retrieval_metrics(self.threshold, model=model)
        elif metrics_type == "generation":
            metrics = get_generation_metrics(self.threshold, model=model)
        elif scenario == "zero_hallucination_refusal":
            # Negative control refusal cases test refusal quality & hallucination prevention (not retrieval of absent facts)
            metrics = get_generation_metrics(self.threshold, model=model)
        else:
            metrics = get_all_rag_metrics(self.threshold, model=model)

        scores: list[MetricScore] = []
        overall_pass = True

        for metric in metrics:
            try:
                metric.measure(test_case)
                metric_passed = getattr(metric, "is_successful", lambda: True)()
                if not metric_passed:
                    overall_pass = False

                scores.append(
                    MetricScore(
                        name=metric.__class__.__name__,
                        score=round(float(getattr(metric, "score", 0.0) or 0.0), 3),
                        passed=metric_passed,
                        reason=getattr(metric, "reason", None),
                    )
                )
            except Exception as exc:
                logger.warning("Metric %s failed evaluation: %s", metric.__class__.__name__, exc)
                scores.append(
                    MetricScore(
                        name=metric.__class__.__name__,
                        score=0.0,
                        passed=False,
                        reason=str(exc),
                    )
                )

        scenario = (
            golden.additional_metadata.get("scenario") if golden.additional_metadata else None
        )
        return SampleEvaluationResult(
            input_query=golden.input,
            actual_output=response.answer,
            expected_output=golden.expected_output or "",
            retrieval_context=retrieval_context,
            metric_scores=scores,
            overall_pass=overall_pass,
            scenario=scenario,
        )

    async def evaluate_golden_dataset(
        self,
        dataset: list[Golden] | None = None,
        session_id: str | None = None,
        metrics_type: str = "all",
        model: Any | None = None,
    ) -> list[SampleEvaluationResult]:
        """Evaluate an entire Golden dataset through the RAG pipeline."""
        goldens = dataset or GOLDEN_DATASET
        results = []
        for idx, g in enumerate(goldens, 1):
            if idx > 1:
                # Inter-case pacing to respect Groq TPM rate limits during live benchmarks
                await asyncio.sleep(2.0)
            logger.info("Evaluating Golden [%d/%d]: '%s'...", idx, len(goldens), g.input[:60])
            res = await self.evaluate_test_case(
                golden=g,
                session_id=session_id,
                metrics_type=metrics_type,
                model=model,
            )
            results.append(res)
        return results

    async def evaluate_query(
        self,
        query: str,
        expected_output: str,
        context: list[str] | None = None,
        session_id: str | None = None,
        metrics_type: str = "all",
        model: Any | None = None,
    ) -> SampleEvaluationResult:
        """Run a query through the RAG pipeline and evaluate against DeepEval metrics."""
        golden = Golden(
            input=query,
            expected_output=expected_output,
            context=context or [],
        )
        return await self.evaluate_test_case(
            golden=golden,
            session_id=session_id,
            metrics_type=metrics_type,
            model=model,
        )


# ==============================================================================
# CLI EXECUTION HARNESS
# ==============================================================================


async def run_evaluation_cli(
    mock: bool | None = None,
    metrics_type: str = "all",
    threshold: float = 0.7,
    output_file: str | None = None,
    model_name: str | None = None,
):
    """Run full evaluation suite from CLI and print structured performance report."""
    settings = get_settings()
    is_mock = (settings.EVALUATION_MODE == "mock") if mock is None else mock
    eval_llm = model_name or settings.DEEPEVAL_MODEL

    print("\n" + "=" * 80)
    print("      DEEPEVAL 2-PILLAR RAG EVALUATION BENCHMARK RUNNER")
    print("=" * 80)
    mode_desc = (
        "MOCK (Offline)"
        if is_mock
        else f"LIVE (Embedding: Gemini {settings.GEMINI_EMBEDDING_MODEL}, Extractor & Judge: Groq {eval_llm})"
    )
    print(
        f"Mode: {mode_desc} [Configured via .env: EVALUATION_MODE='{settings.EVALUATION_MODE}', DEEPEVAL_MODEL='{settings.DEEPEVAL_MODEL}']"
    )
    print(f"Metrics: {metrics_type.upper()}")
    print(f"Passing Threshold: {threshold}")
    print(f"Golden Test Cases: {len(GOLDEN_DATASET)}")
    print("=" * 80 + "\n")

    print("[1/3] Initializing and seeding benchmark vector store...")
    pipeline, session_id = await setup_benchmark_pipeline(mock=is_mock, model_name=eval_llm)
    print(f"      Benchmark pipeline ready (Session ID: '{session_id}').\n")

    eval_model = MockDeepEvalModel() if is_mock else GroqDeepEvalModel(model_name=eval_llm)
    evaluator = RAGPipelineEvaluator(pipeline=pipeline, threshold=threshold)

    print("[2/3] Executing Golden Dataset queries through RAG Pipeline & DeepEval...")
    results = await evaluator.evaluate_golden_dataset(
        session_id=session_id,
        metrics_type=metrics_type,
        model=eval_model,
    )
    print(f"      Completed {len(results)} test cases.\n")

    print("[3/3] EVALUATION REPORT SUMMARY:")
    print("-" * 80)

    total_metrics = 0
    passed_metrics = 0

    report_data = []

    for idx, r in enumerate(results, 1):
        status_icon = "PASS" if r.overall_pass else "FAIL"
        print(f"\n[Case #{idx}] [{status_icon}] Scenario: {r.scenario or 'general'}")
        print(f"  Query:           {r.input_query}")
        print(f"  Actual Output:   {r.actual_output}")
        print(f"  Expected Output: {r.expected_output}")
        print("  Metric Scores:")

        metric_dict = {}
        for m in r.metric_scores:
            total_metrics += 1
            if m.passed:
                passed_metrics += 1
            pass_str = "PASS" if m.passed else "FAIL"
            print(f"    - {m.name:<26}: {m.score:.3f} [{pass_str}]")
            metric_dict[m.name] = {"score": m.score, "passed": m.passed, "reason": m.reason}

        report_data.append(
            {
                "case_index": idx,
                "scenario": r.scenario,
                "query": r.input_query,
                "actual_output": r.actual_output,
                "expected_output": r.expected_output,
                "overall_pass": r.overall_pass,
                "metrics": metric_dict,
            }
        )

    pass_rate = (passed_metrics / total_metrics * 100) if total_metrics else 0
    print("\n" + "=" * 80)
    print(
        f"OVERALL RESULTS: {passed_metrics}/{total_metrics} individual metrics passed ({pass_rate:.1f}% Pass Rate)"
    )
    print("=" * 80)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Detailed JSON evaluation report saved to: {output_file}\n")


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=f"DeepEval RAG Benchmark Evaluation (Active .env EVALUATION_MODE='{settings.EVALUATION_MODE}')"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline evaluation using deterministic mock models (overrides .env EVALUATION_MODE)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live evaluation against Gemini and Groq LLMs (overrides .env EVALUATION_MODE)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Override unified extractor and judge model (default from .env: '{settings.DEEPEVAL_MODEL}')",
    )
    parser.add_argument(
        "--metrics",
        choices=["all", "retrieval", "generation"],
        default="all",
        help="Subset of metrics to run (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum score threshold for metric pass (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON evaluation report",
    )
    args = parser.parse_args()

    # CLI flags explicitly override the .env/config setting:
    if args.mock:
        use_mock = True
    elif args.live:
        use_mock = False
    else:
        # Switch directly from config / .env (EVALUATION_MODE)
        use_mock = settings.EVALUATION_MODE == "mock"

    asyncio.run(
        run_evaluation_cli(
            mock=use_mock,
            metrics_type=args.metrics,
            threshold=args.threshold,
            output_file=args.output,
            model_name=args.model,
        )
    )


if __name__ == "__main__":
    main()

# Backward compatibility aliases for deep imports
sys.modules["app.rag.evaluation.evaluator"] = sys.modules[__name__]
sys.modules["app.rag.evaluation.metrics"] = sys.modules[__name__]
