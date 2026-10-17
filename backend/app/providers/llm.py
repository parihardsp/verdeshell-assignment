"""LLM Provider abstractions and implementations.

Includes BaseLLMProvider, LangChainLLMProvider, GroqLLMProvider, GeminiLLMProvider, and MockLLMProvider.
"""

import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import TypeVar

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.exceptions import UpstreamProviderError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Contract for Large Language Model generation."""

    @abstractmethod
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 10000,
    ) -> str:
        """Synthesize a grounded answer based on system instructions and user prompt."""

    async def generate_structured(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> T:
        """Synthesize a grounded response adhering strictly to a Pydantic schema."""
        raw = await self.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature or 0.1,
        )
        import json
        import re

        clean = raw.strip()
        if "```" in clean:
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()
        try:
            return schema.model_validate(json.loads(clean))
        except Exception:
            return schema.model_validate({"answer": raw})


class LangChainLLMProvider(BaseLLMProvider):
    """LLM Provider powered by LangChain Chat Models with native LangSmith tracing."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 10000,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

        if self.provider == "groq":
            from langchain_groq import ChatGroq

            self.chat_model = ChatGroq(
                groq_api_key=api_key,
                model_name=model or "openai/gpt-oss-120b",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider in ("gemini", "google"):
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.chat_model = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model or "gemini-flash-lite-latest",
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        elif self.provider in ("openai", "openrouter"):
            from langchain_openai import ChatOpenAI

            self.chat_model = ChatOpenAI(
                api_key=api_key,
                model=model or "gpt-4o-mini",
                base_url=base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported LangChain LLM provider: {provider}")

        logger.info(
            "Initialized LangChainLLMProvider (provider=%s, model=%s, max_tokens=%d) with LangSmith tracing.",
            self.provider,
            model,
            self.max_tokens,
        )

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 10000,
    ) -> str:
        from app.core.logging import get_session_context, get_user_context

        u_ctx = get_user_context()
        s_ctx = get_session_context()
        config = {
            "tags": [f"user:{u_ctx}", f"session:{s_ctx}", f"provider:{self.provider}"],
            "metadata": {
                "user_id": u_ctx,
                "session_id": s_ctx,
                "provider": self.provider,
                "model": getattr(self.chat_model, "model_name", None)
                or getattr(self.chat_model, "model", None)
                or "default",
            },
        }
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = await self.chat_model.ainvoke(messages, config=config)
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    elif hasattr(part, "text"):
                        text_parts.append(str(part.text))
                    elif isinstance(part, str):
                        text_parts.append(part)
                    else:
                        text_parts.append(str(part))
                return "".join(text_parts).strip()
            return str(content).strip()
        except Exception as exc:
            logger.error("LangChain LLM invocation failed: %s", exc)
            raise UpstreamProviderError(f"LangChain LLM provider failed: {exc!s}", status_code=502)

    async def generate_structured(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> T:
        """Synthesize structured output adhering strictly to a Pydantic schema with native LLM tool calling."""
        from app.core.logging import get_session_context, get_user_context

        u_ctx = get_user_context()
        s_ctx = get_session_context()
        config = {
            "tags": [
                f"user:{u_ctx}",
                f"session:{s_ctx}",
                f"provider:{self.provider}",
                "structured_output:true",
            ],
            "metadata": {
                "user_id": u_ctx,
                "session_id": s_ctx,
                "provider": self.provider,
                "model": getattr(self.chat_model, "model_name", None)
                or getattr(self.chat_model, "model", None)
                or "default",
                "schema": schema.__name__,
            },
        }
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        try:
            structured_model = self.chat_model.with_structured_output(schema)
            result = await structured_model.ainvoke(messages, config=config)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception as exc:
            logger.warning(
                "Native structured output with %s encountered an issue (%s: %s). Falling back to JSON parsing...",
                self.provider,
                type(exc).__name__,
                exc,
            )

        # Fallback: prompt for JSON schema and parse
        import json
        import re

        json_prompt = (
            f"{system_prompt}\n\n"
            f"You must return ONLY a valid JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n"
            f"Do not include any text or markdown fences outside the JSON."
        )
        raw_response = await self.generate_response(
            system_prompt=json_prompt,
            user_prompt=user_prompt,
            temperature=temperature or self.temperature,
        )
        clean_text = raw_response.strip()
        if "```" in clean_text:
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r"```$", "", clean_text, flags=re.MULTILINE).strip()
        parsed = json.loads(clean_text)
        return schema.model_validate(parsed)


class GroqLLMProvider(BaseLLMProvider):
    """Answers questions using Groq's ultra-low-latency Llama inference API directly."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 15.0,
    ):
        if not api_key:
            raise ValueError("Groq API key is required for GroqLLMProvider.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 10000,
    ) -> str:
        """Execute async completion request against Groq endpoint."""
        url = f"{self.base_url}/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    raise UpstreamProviderError(
                        message="Upstream Groq LLM rate limit exceeded. Please retry later.",
                        status_code=503,
                        headers={"Retry-After": "30"},
                    )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise UpstreamProviderError("Groq returned empty response choices.")

                return choices[0].get("message", {}).get("content", "").strip()

            except httpx.TimeoutException as exc:
                logger.error("Groq LLM request timed out: %s", exc)
                raise UpstreamProviderError("Upstream Groq request timed out.", status_code=504)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Groq LLM HTTP error %s: %s",
                    exc.response.status_code,
                    exc.response.text,
                )
                raise UpstreamProviderError(
                    f"Groq API returned status {exc.response.status_code}.",
                    status_code=502,
                )
            except (UpstreamProviderError, Exception):
                raise


class GeminiLLMProvider(BaseLLMProvider):
    """Answers questions using Google Gemini's generateContent REST API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        timeout_seconds: float = 15.0,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required for GeminiLLMProvider.")
        self.api_key = api_key
        self.model = model.replace("models/", "")
        self.timeout = timeout_seconds

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 10000,
    ) -> str:
        """Execute async completion request against Google Gemini."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                if response.status_code == 429:
                    raise UpstreamProviderError(
                        message="Upstream Gemini LLM rate limit exceeded. Please retry later.",
                        status_code=503,
                        headers={"Retry-After": "30"},
                    )
                response.raise_for_status()
                data = response.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    raise UpstreamProviderError("Gemini generated no response candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return "The knowledge assistant could not formulate an answer."

                return parts[0].get("text", "").strip()

            except httpx.TimeoutException as exc:
                logger.error("Gemini LLM request timed out: %s", exc)
                raise UpstreamProviderError("Upstream Gemini request timed out.", status_code=504)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Gemini LLM HTTP error %s: %s",
                    exc.response.status_code,
                    exc.response.text,
                )
                raise UpstreamProviderError(
                    f"Gemini LLM API returned status {exc.response.status_code}.",
                    status_code=502,
                )
            except (UpstreamProviderError, Exception):
                raise


class MockLLMProvider(BaseLLMProvider):
    """Offline deterministic synthesizer for zero-dependency local evaluation."""

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 10000,
    ) -> str:
        """Extract and format a grounded response from the provided context blocks."""
        # Match either XML <context>...</context> or legacy plain text blocks
        xml_context = re.search(r"<context>(.*?)</context>", user_prompt, re.DOTALL | re.IGNORECASE)
        if xml_context:
            context = xml_context.group(1).strip()
        else:
            context_match = re.search(
                r"(?:CONTEXT EXCERPTS|CONTEXT).*?:\n(.*?)\n\n(?:USER QUESTION|QUESTION):",
                user_prompt,
                re.DOTALL | re.IGNORECASE,
            )
            context = context_match.group(1).strip() if context_match else ""

        if not context or "No relevant context found" in context:
            return "The provided knowledge base does not contain sufficient information to answer this question."

        doc_matches = re.findall(
            r'<chunk[^>]*document="([^"]+)"',
            context,
        )
        if not doc_matches:
            doc_matches = re.findall(
                r"(?:\[Document:|\-\-\-\s*\[Chunk\s*\d+\s*\|\s*Document:)\s*([^|\(\]\n]+)",
                context,
            )
        doc_title = doc_matches[0].strip() if doc_matches else "Knowledge Base"

        first_lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip()
            and not line.startswith("<")
            and not line.startswith("[")
            and not line.startswith("---")
        ]

        summary_sentence = first_lines[0] if first_lines else f"Information documented in {doc_title}."
        bullet_items = first_lines[1:4] if len(first_lines) > 1 else []

        if bullet_items:
            bullets_text = "\n".join(f"- **Key Fact:** {item}" for item in bullet_items)
            return f"{summary_sentence}\n\n**Key Points:**\n{bullets_text}"
        return summary_sentence

    async def generate_structured(
        self,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> T:
        """Deterministic structured generation for testing and offline evaluation."""
        raw_answer = await self.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        is_insufficient = (
            "does not contain sufficient information" in raw_answer
            or "does not contain information" in raw_answer
        )
        # Match chunk indices from <chunk index="0"> or legacy "Chunk 0"
        raw_indices = re.findall(r'<chunk[^>]*index="(\d+)"', user_prompt)
        if not raw_indices:
            raw_indices = re.findall(r"Chunk\s*(\d+)", user_prompt)

        cited = (
            [int(m) for m in raw_indices]
            if not is_insufficient and raw_indices
            else ([0] if not is_insufficient else [])
        )
        if hasattr(schema, "has_sufficient_context"):
            return schema.model_validate(
                {
                    "answer": raw_answer,
                    "has_sufficient_context": not is_insufficient,
                    "cited_chunk_indices": cited if not is_insufficient else [],
                    "confidence_score": 1.0 if not is_insufficient else 0.0,
                }
            )
        return schema.model_validate({"answer": raw_answer})


# Backward compatibility aliases for deep imports
sys.modules["app.providers.llm.base"] = sys.modules[__name__]
sys.modules["app.providers.llm.mock"] = sys.modules[__name__]
sys.modules["app.providers.llm.gemini"] = sys.modules[__name__]
sys.modules["app.providers.llm.groq"] = sys.modules[__name__]
sys.modules["app.providers.llm.langchain_provider"] = sys.modules[__name__]
