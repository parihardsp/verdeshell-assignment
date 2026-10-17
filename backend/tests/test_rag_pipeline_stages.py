"""Unit tests for decoupled RAG Pipeline stages (Retriever, Reranker, Synthesizer, Pipeline)."""

import chromadb
import pytest

from app.providers.embeddings import MockEmbeddingProvider
from app.providers.llm import MockLLMProvider
from app.providers.vector_store import ChromaVectorStore, IndexedChunk
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import (
    GroqLLMReranker,
    MockReranker,
    PassThroughReranker,
    ReciprocalRankReranker,
)
from app.rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    RetrievedChunk,
    VectorRetriever,
)
from app.rag.synthesizer import GroundedSynthesizer


@pytest.fixture
def mock_retriever():
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    embedding_provider = MockEmbeddingProvider()
    return (
        VectorRetriever(embedding_provider=embedding_provider, vector_store=vector_store),
        vector_store,
        embedding_provider,
    )


@pytest.mark.asyncio
async def test_vector_retriever_session_scoping(mock_retriever):
    retriever, vector_store, embedding_provider = mock_retriever

    # Ingest chunks under two distinct sessions
    emb1 = await embedding_provider.embed_text(
        "Employees receive 25 days of annual vacation leave."
    )
    emb2 = await embedding_provider.embed_text("Finance department annual budget is 500,000 USD.")

    chunk_a = IndexedChunk(
        chunk_id="chk_hr_1",
        document_id="doc_hr",
        document_title="HR Policy",
        chunk_index=0,
        content="Employees receive 25 days of annual vacation leave.",
        embedding=emb1,
        session_id="sess_hr",
        file_name="hr.txt",
        page_number=1,
    )
    chunk_b = IndexedChunk(
        chunk_id="chk_fin_1",
        document_id="doc_fin",
        document_title="Finance Budget",
        chunk_index=0,
        content="Finance department annual budget is 500,000 USD.",
        embedding=emb2,
        session_id="sess_fin",
        file_name="finance.txt",
        page_number=1,
    )
    await vector_store.add_chunks([chunk_a, chunk_b])

    # Query scoped to sess_hr
    results_hr = await retriever.retrieve(
        query="How many vacation days?",
        session_id="sess_hr",
        top_k=5,
        threshold=0.20,
    )
    assert len(results_hr) == 1
    assert results_hr[0].session_id == "sess_hr"
    assert results_hr[0].document_title == "HR Policy"

    # Query scoped to sess_fin
    results_fin = await retriever.retrieve(
        query="How many vacation days?",
        session_id="sess_fin",
        top_k=5,
        threshold=0.20,
    )
    assert len(results_fin) == 0


@pytest.mark.asyncio
async def test_bm25_retriever_keyword_search():
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    retriever = BM25Retriever(vector_store=vector_store)

    c1 = IndexedChunk(
        chunk_id="c_err_1",
        document_id="doc_ops",
        document_title="Incident Handbook",
        chunk_index=0,
        content="When system returns ERR_CODE_504_GATEWAY_TIMEOUT, immediately restart the ingress proxy.",
        embedding=[0.1] * 384,
        session_id="s_ops",
    )
    c2 = IndexedChunk(
        chunk_id="c_err_2",
        document_id="doc_billing",
        document_title="Billing Protocol",
        chunk_index=0,
        content="Invoices are issued on the 1st of every month.",
        embedding=[0.2] * 384,
        session_id="s_ops",
    )
    await vector_store.add_chunks([c1, c2])

    # Search for the exact error code
    results = await retriever.retrieve(
        query="ERR_CODE_504_GATEWAY_TIMEOUT",
        session_id="s_ops",
        top_k=3,
    )
    assert len(results) >= 1
    assert results[0].chunk_id == "c_err_1"
    assert results[0].retrieval_channel == "bm25"
    assert results[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_hybrid_retriever_dual_channel():
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    embedding_provider = MockEmbeddingProvider()
    hybrid = HybridRetriever(embedding_provider=embedding_provider, vector_store=vector_store)

    emb_vac = await embedding_provider.embed_text(
        "Employees receive 25 days annual vacation leave."
    )
    emb_tech = await embedding_provider.embed_text("Tech stipend is 2500 USD for equipment.")

    chunk1 = IndexedChunk(
        chunk_id="chk_vac",
        document_id="d1",
        document_title="Vacation Policy",
        chunk_index=0,
        content="Employees receive 25 days annual vacation leave.",
        embedding=emb_vac,
        session_id="s_dual",
    )
    chunk2 = IndexedChunk(
        chunk_id="chk_tech",
        document_id="d2",
        document_title="Tech Equipment",
        chunk_index=0,
        content="Tech stipend is 2500 USD for equipment.",
        embedding=emb_tech,
        session_id="s_dual",
    )
    await vector_store.add_chunks([chunk1, chunk2])

    results = await hybrid.retrieve(
        query="tech stipend equipment 2500",
        session_id="s_dual",
        top_k=5,
        threshold=0.10,
    )
    assert len(results) >= 1
    # Check that chunk was retrieved with hybrid/dense channel info
    ids = [r.chunk_id for r in results]
    assert "chk_tech" in ids


@pytest.mark.asyncio
async def test_reciprocal_rank_reranker():
    reranker = ReciprocalRankReranker(rrf_k=60)
    query = "vacation leave notice requirement"

    # Candidate A: strong semantic match, moderate keyword match
    cand_a = RetrievedChunk(
        chunk_id="c_a",
        document_id="d1",
        document_title="General Info",
        chunk_index=0,
        content="Annual holidays and time off procedures for employees.",
        session_id="s1",
        score=0.75,
    )
    # Candidate B: exact keyword match for 'vacation leave notice requirement'
    cand_b = RetrievedChunk(
        chunk_id="c_b",
        document_id="d2",
        document_title="Leave Policy",
        chunk_index=1,
        content="Vacation leave requests require 14 days advance notice requirement.",
        session_id="s1",
        score=0.70,
    )
    # Candidate C: lower dense score, slight lexical match
    cand_c = RetrievedChunk(
        chunk_id="c_c",
        document_id="d3",
        document_title="Office Hours",
        chunk_index=0,
        content="Notice of working hours and holidays.",
        session_id="s1",
        score=0.65,
    )

    reranked = await reranker.rerank(query=query, candidates=[cand_a, cand_b, cand_c], top_n=2)
    assert len(reranked) == 2
    # Candidate B has higher lexical overlap + high dense score -> RRF boosts Candidate B to #1
    assert reranked[0].chunk_id == "c_b"


@pytest.mark.asyncio
async def test_pass_through_reranker():
    reranker = PassThroughReranker()
    c1 = RetrievedChunk(
        chunk_id="1",
        document_id="d",
        document_title="T",
        chunk_index=0,
        content="A",
        session_id="s",
        score=0.9,
    )
    c2 = RetrievedChunk(
        chunk_id="2",
        document_id="d",
        document_title="T",
        chunk_index=1,
        content="B",
        session_id="s",
        score=0.6,
    )
    c3 = RetrievedChunk(
        chunk_id="3",
        document_id="d",
        document_title="T",
        chunk_index=2,
        content="C",
        session_id="s",
        score=0.8,
    )

    results = await reranker.rerank("test query", [c1, c2, c3], top_n=2)
    assert len(results) == 2
    assert results[0].chunk_id == "1"
    assert results[1].chunk_id == "3"


@pytest.mark.asyncio
async def test_grounded_synthesizer_no_context():
    synthesizer = GroundedSynthesizer(llm_provider=MockLLMProvider())
    result = await synthesizer.synthesize(question="What is the refund policy?", context_chunks=[])
    assert "does not contain information" in result.answer
    assert len(result.sources) == 0


@pytest.mark.asyncio
async def test_rag_pipeline_end_to_end():
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    embedding_provider = MockEmbeddingProvider()
    retriever = VectorRetriever(embedding_provider=embedding_provider, vector_store=vector_store)
    reranker = ReciprocalRankReranker(rrf_k=60)
    synthesizer = GroundedSynthesizer(llm_provider=MockLLMProvider())

    emb = await embedding_provider.embed_text("Laptops are refreshed every three years.")
    chunk = IndexedChunk(
        chunk_id="chk_it_1",
        document_id="doc_it",
        document_title="IT Policy",
        chunk_index=0,
        content="Company laptops are refreshed every three years for full-time employees.",
        embedding=emb,
        session_id="sess_test",
        file_name="it.pdf",
        page_number=2,
    )
    await vector_store.add_chunks([chunk])

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        synthesizer=synthesizer,
        top_k=5,
        top_n=3,
        similarity_threshold=0.20,
    )

    response = await pipeline.run(
        question="How often are company laptops refreshed?",
        session_id="sess_test",
    )
    assert response.session_id == "sess_test"
    assert len(response.sources) > 0
    assert response.sources[0].file_name == "it.pdf"
    assert response.sources[0].page_number == 2


@pytest.mark.asyncio
async def test_mock_reranker():
    reranker = MockReranker()
    c1 = RetrievedChunk(
        chunk_id="1",
        document_id="d",
        document_title="T",
        chunk_index=0,
        content="A",
        session_id="s",
        score=0.4,
    )
    c2 = RetrievedChunk(
        chunk_id="2",
        document_id="d",
        document_title="T",
        chunk_index=1,
        content="B",
        session_id="s",
        score=0.9,
    )
    results = await reranker.rerank("query", [c1, c2], top_n=1)
    assert len(results) == 1
    assert results[0].chunk_id == "2"
    assert results[0].reranker_score == 9.0


@pytest.mark.asyncio
async def test_groq_llm_reranker_happy_path(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from langchain_core.messages import AIMessage

    mock_chat = MagicMock()
    mock_chat.ainvoke = AsyncMock(
        return_value=AIMessage(
            content='[{"index": 1, "score": 9.8, "reason": "Exact match for termination criteria"}]'
        )
    )

    import langchain_groq

    monkeypatch.setattr(langchain_groq, "ChatGroq", MagicMock(return_value=mock_chat))

    reranker = GroqLLMReranker(
        groq_api_key="gsk_mock_key", model="openai/gpt-oss-120b", max_candidates=5
    )

    c0 = RetrievedChunk(
        chunk_id="c0",
        document_id="d1",
        document_title="T1",
        chunk_index=0,
        content="General coffee machine info",
        session_id="s",
        score=0.8,
    )
    c1 = RetrievedChunk(
        chunk_id="c1",
        document_id="d2",
        document_title="T2",
        chunk_index=1,
        content="Termination happens with 30 days notice",
        session_id="s",
        score=0.7,
    )

    results = await reranker.rerank("What are the termination conditions?", [c0, c1], top_n=2)
    assert len(results) == 2
    # Groq scored index 1 as 9.8 -> c1 must be promoted to #1
    assert results[0].chunk_id == "c1"
    assert results[0].reranker_score == 9.8
    assert results[0].score == 0.98


@pytest.mark.asyncio
async def test_groq_llm_reranker_fallback_on_exception(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    mock_chat = MagicMock()
    mock_chat.ainvoke = AsyncMock(side_effect=RuntimeError("Groq rate limit exceeded"))

    import langchain_groq

    monkeypatch.setattr(langchain_groq, "ChatGroq", MagicMock(return_value=mock_chat))

    reranker = GroqLLMReranker(groq_api_key="gsk_mock_key", model="openai/gpt-oss-120b")

    c0 = RetrievedChunk(
        chunk_id="c0",
        document_id="d1",
        document_title="T1",
        chunk_index=0,
        content="Keyword match vacation policy",
        session_id="s",
        score=0.8,
    )
    c1 = RetrievedChunk(
        chunk_id="c1",
        document_id="d2",
        document_title="T2",
        chunk_index=1,
        content="Holiday notice schedule",
        session_id="s",
        score=0.5,
    )

    # Should not throw; should gracefully fall back to RRF fusion ranks
    results = await reranker.rerank("vacation policy", [c0, c1], top_n=2)
    assert len(results) == 2


def test_grounded_synthesis_output_schema():
    from app.schemas.rag import GroundedSynthesisOutput

    # Valid output
    output = GroundedSynthesisOutput(
        answer="The capital of France is Paris.",
        has_sufficient_context=True,
        cited_chunk_indices=[0, 1],
        confidence_score=0.95,
    )
    assert output.answer == "The capital of France is Paris."
    assert output.has_sufficient_context is True
    assert output.cited_chunk_indices == [0, 1]
    assert output.confidence_score == 0.95

    # Test confidence score bounds validation
    with pytest.raises(Exception):
        GroundedSynthesisOutput(answer="Test", confidence_score=1.5)
    with pytest.raises(Exception):
        GroundedSynthesisOutput(answer="Test", confidence_score=-0.1)


def test_rag_prompts_builders():
    from app.rag.prompts.reranker_prompts import (
        RERANKER_SYSTEM_PROMPT,
        build_reranker_passages_text,
        build_reranker_prompt,
    )
    from app.rag.prompts.synthesis_prompts import (
        build_synthesis_context_blocks,
        build_synthesis_user_prompt,
    )

    c0 = RetrievedChunk(
        chunk_id="c0",
        document_id="doc1",
        document_title="Employee Handbook",
        chunk_index=0,
        content="Standard working hours are 9am to 5pm.",
        session_id="sess_1",
        file_name="handbook.pdf",
        page_number=3,
        score=0.9,
    )
    c1 = RetrievedChunk(
        chunk_id="c1",
        document_id="doc2",
        document_title="Security Policy",
        chunk_index=1,
        content="Badges must be worn at all times.",
        session_id="sess_1",
        score=0.7,
    )

    # Test synthesis prompt builders
    blocks, context_str = build_synthesis_context_blocks([c0, c1])
    assert len(blocks) == 2
    assert (
        '<chunk index="0" document="Employee Handbook" file="handbook.pdf" page="3">' in blocks[0]
    )
    assert '<chunk index="1" document="Security Policy" page="1">' in blocks[1]
    assert "Standard working hours are 9am to 5pm." in context_str

    user_prompt = build_synthesis_user_prompt(
        question="What are the working hours?",
        context_str=context_str,
        session_id="sess_1",
    )
    assert "<context>" in user_prompt
    assert '<user_query session_id="sess_1">' in user_prompt
    assert "What are the working hours?" in user_prompt
    assert "<instructions>" in user_prompt

    # Test reranker prompt builders
    passages_text = build_reranker_passages_text([c0, c1], max_chars=100)
    assert (
        '<passage index="0" document="Employee Handbook" file="handbook.pdf" page="3">'
        in passages_text
    )
    assert '<passage index="1" document="Security Policy" page="1">' in passages_text
    assert "Standard working hours are 9am to 5pm." in passages_text

    reranker_prompt = build_reranker_prompt(
        query="working hours", passages_str=passages_text, top_candidates_limit=5
    )
    assert "<query>\nworking hours\n</query>" in reranker_prompt
    assert "<candidate_passages>" in reranker_prompt
    assert RERANKER_SYSTEM_PROMPT in reranker_prompt


@pytest.mark.asyncio
async def test_grounded_synthesizer_citation_filtering():

    class CustomMockLLM(MockLLMProvider):
        async def generate_structured(self, schema, system_prompt, user_prompt, temperature=None):
            return schema(
                answer="Standard working hours are 9am to 5pm.",
                has_sufficient_context=True,
                cited_chunk_indices=[1],  # Only cite index 1
                confidence_score=0.98,
            )

    c0 = RetrievedChunk(
        chunk_id="c0",
        document_id="d1",
        document_title="Doc1",
        chunk_index=0,
        content="Irrelevant cafeteria menu",
        session_id="s",
        score=0.5,
    )
    c1 = RetrievedChunk(
        chunk_id="c1",
        document_id="d2",
        document_title="Doc2",
        chunk_index=0,
        content="Standard working hours are 9am to 5pm.",
        session_id="s",
        score=0.95,
    )

    synthesizer = GroundedSynthesizer(llm_provider=CustomMockLLM())
    res = await synthesizer.synthesize(question="What are the hours?", context_chunks=[c0, c1])

    assert res.answer == "Standard working hours are 9am to 5pm."
    assert len(res.sources) == 1
    assert res.sources[0].document_id == "d2"
    assert res.confidence_score == 0.98


@pytest.mark.asyncio
async def test_grounded_synthesizer_insufficient_context_gating():

    class InsufficientContextMockLLM(MockLLMProvider):
        async def generate_structured(self, schema, system_prompt, user_prompt, temperature=None):
            return schema(
                answer="The provided knowledge base does not contain information about salary bands.",
                has_sufficient_context=False,
                cited_chunk_indices=[],
                confidence_score=0.0,
            )

    c0 = RetrievedChunk(
        chunk_id="c0",
        document_id="d1",
        document_title="Doc1",
        chunk_index=0,
        content="General office guidelines.",
        session_id="s",
        score=0.3,
    )

    synthesizer = GroundedSynthesizer(llm_provider=InsufficientContextMockLLM())
    res = await synthesizer.synthesize(
        question="What is the salary band for senior engineer?", context_chunks=[c0]
    )

    assert "does not contain information" in res.answer
    assert len(res.sources) == 0
    assert res.confidence_score == 0.0
