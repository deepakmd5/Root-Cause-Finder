"""OpenAI-backed LLM adapter.

Uses OpenAI's Chat Completions API via ``httpx``. Kept intentionally
lean - the important contract is :meth:`decide`, which returns a
:class:`AgentDecision` parsed from a strict JSON response.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.base import AgentDecision, DecisionType, LLMMessage

log = get_logger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class OpenAILLM:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def decide(
        self,
        messages: list[LLMMessage],
        available_tools: list[dict[str, Any]],
    ) -> AgentDecision:
        tools_block = json.dumps(available_tools, indent=2)
        system_suffix = (
            "\n\nAvailable tools (JSON):\n"
            f"{tools_block}\n\n"
            "Respond with ONE JSON object, no prose. Schema:\n"
            "{\n"
            '  "type": "use_tool" | "finalize",\n'
            '  "thought": "<why>",\n'
            '  "tool": "<tool_name or null>",\n'
            '  "tool_input": { ... },\n'
            '  "final_answer": { ...RCA report... } | null\n'
            "}"
        )
        payload_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        if payload_messages and payload_messages[0]["role"] == "system":
            payload_messages[0]["content"] += system_suffix
        else:
            payload_messages.insert(
                0, {"role": "system", "content": system_suffix}
            )

        body = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(OPENAI_URL, headers=headers, json=body)

        if resp.status_code >= 400:
            log.error(
                "openai.error",
                status=resp.status_code,
                body=resp.text[:400],
            )
            raise LLMError(f"OpenAI error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Malformed OpenAI response: {exc}") from exc

        return AgentDecision(
            type=DecisionType(parsed.get("type", "finalize")),
            thought=parsed.get("thought", ""),
            tool=parsed.get("tool"),
            tool_input=parsed.get("tool_input") or {},
            final_answer=parsed.get("final_answer"),
        )
