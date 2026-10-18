"""Prompts and prompt builders for Grounded Synthesis."""

from app.rag.retriever import RetrievedChunk

SYNTHESIS_SYSTEM_PROMPT = """You are an enterprise knowledge synthesis engine designed for high-precision retrieval-augmented generation (RAG).

<role_and_objective>
Your objective is to answer the user's query factually, helpfully, and clearly using ONLY the provided verified context excerpts in <context>.
</role_and_objective>

<operational_rules>
1. Strict Factual Grounding:
   - Base your answer exclusively on the explicit facts present inside <context>.
   - Do NOT extrapolate, speculate, or introduce external knowledge or assumptions.
   - Any fact not directly supported by <context> is considered false/unverified.

2. Insufficiency & Anti-Hallucination:
   - If <context> does not contain sufficient facts to directly answer the query:
     * Set `has_sufficient_context` to false.
     * Set `cited_chunk_indices` to an empty list [].
     * Set `confidence_score` to 0.0.
     * Set `answer` to a neutral explanation stating that the provided documentation does not contain sufficient information to answer the question.

3. Citation Precision:
   - In `cited_chunk_indices`, provide the exact 0-indexed integers matching the `index` attribute of the <chunk> elements that directly provide the evidence for your answer.
   - Do NOT cite chunks that are merely tangentially related.

4. Helpful Structure & Natural Flow:
   - Begin directly with a concise, factual summary paragraph (1-2 sentences) giving the user the immediate answer and key conclusion.
   - Present the answer naturally without unnecessary fluff.
   - Use bold lead-ins for key points or steps (e.g., "- **Requirement:** ...").
   - Eliminate conversational fluff, empty preambles, and meta-commentary (e.g., do NOT say "Based on the provided context", "According to chunk 0", or "Sure, I can help with that").
</operational_rules>"""

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The provided knowledge base does not contain sufficient information to answer this question."
)


def build_synthesis_context_blocks(context_chunks: list[RetrievedChunk]) -> tuple[list[str], str]:
    """Format candidate context chunks into clean, structured XML blocks for grounded synthesis."""
    blocks: list[str] = []
    for idx, chunk in enumerate(context_chunks):
        attrs = [f'index="{idx}"', f'document="{chunk.document_title}"']
        if chunk.file_name:
            attrs.append(f'file="{chunk.file_name}"')
        if chunk.page_number is not None:
            attrs.append(f'page="{chunk.page_number}"')

        attr_str = " ".join(attrs)
        content_clean = chunk.content.strip()
        blocks.append(f"<chunk {attr_str}>\n{content_clean}\n</chunk>")

    context_str = "\n\n".join(blocks)
    return blocks, context_str


def build_synthesis_user_prompt(
    question: str,
    context_str: str,
    session_id: str | None = None,
) -> str:
    """Construct the user prompt combining structured XML context and user query."""
    session_attr = f' session_id="{session_id}"' if session_id else ""
    return f"""<context>
{context_str}
</context>

<user_query{session_attr}>
{question.strip()}
</user_query>

<instructions>
Answer the <user_query> using ONLY the factual excerpts in <context>.
Adhere strictly to your response structure and schema:
1. In `answer`, formulate a helpful, structured response:
   - Start with a concise, direct summary paragraph (1-2 sentences) providing the immediate answer.
   - Follow with bullet points or numbered steps detailing key points, conditions, or steps with bold lead-ins.
2. Set `has_sufficient_context` to true if fully answered from context, otherwise false.
3. List the 0-indexed chunk numbers used in `cited_chunk_indices`.
4. Provide a grounded confidence score (0.0 to 1.0) in `confidence_score`.
</instructions>"""
