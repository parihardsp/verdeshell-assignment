"""Prompts and prompt builders for Neural Precision Reranking."""

from app.rag.retriever import RetrievedChunk

RERANKER_SYSTEM_PROMPT = """You are a high-precision neural relevance evaluator for an enterprise search system.

<role_and_objective>
Your objective is to evaluate candidate text passages against a user query and determine how directly, accurately, and completely each passage answers that query.
</role_and_objective>

<scoring_rubric>
- 9.0 - 10.0: Perfect Match. Passage directly, completely, and factually answers the query.
- 7.0 - 8.9: Highly Relevant. Passage provides critical facts, definitions, or clauses directly relevant to the query.
- 4.0 - 6.9: Partially Relevant. Passage is on the general topic or provides background context, but does not answer the core question.
- 0.0 - 3.9: Irrelevant. Passage mentions keywords peripherally, discusses an unrelated subject, or contains no actionable information.
</scoring_rubric>

<constraints>
- Return ONLY a valid JSON array of objects.
- Do NOT wrap your output in markdown code fences (```json or ```).
- Do NOT include conversational framing, commentary, or reasoning outside the JSON objects.
</constraints>"""


def build_reranker_passages_text(candidates: list[RetrievedChunk], max_chars: int = 500) -> str:
    """Format candidate chunks into structured XML passage elements for neural cross-attention."""
    passage_blocks: list[str] = []
    for idx, chunk in enumerate(candidates):
        snippet = chunk.content[:max_chars].replace("\n", " ").strip()
        attrs = [f'index="{idx}"']
        if chunk.document_title:
            attrs.append(f'document="{chunk.document_title}"')
        if chunk.file_name:
            attrs.append(f'file="{chunk.file_name}"')
        if chunk.page_number is not None:
            attrs.append(f'page="{chunk.page_number}"')

        attr_str = " ".join(attrs)
        passage_blocks.append(f"<passage {attr_str}>\n{snippet}\n</passage>")

    return "\n\n".join(passage_blocks)


def build_reranker_prompt(query: str, passages_str: str, top_candidates_limit: int = 6) -> str:
    """Construct the clean XML prompt for zero-temperature JSON relevance scoring."""
    return f"""{RERANKER_SYSTEM_PROMPT}

<query>
{query.strip()}
</query>

<candidate_passages>
{passages_str}
</candidate_passages>

<task_instructions>
1. Evaluate each passage in <candidate_passages> against the <query> using the scoring rubric.
2. Select up to {top_candidates_limit} of the most relevant passages with score > 0.0.
3. Return ONLY a valid JSON array of objects sorted by score descending:
[
  {{"index": 0, "score": 9.5, "reason": "Directly provides the exact figure and policy clause."}}
]
</task_instructions>"""
