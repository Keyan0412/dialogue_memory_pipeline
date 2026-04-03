from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class JSONLLM(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        ...


class OpenAIJSONLLM:
    """OpenAI-compatible JSON client with tolerant response extraction."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("openai package is required for OpenAIJSONLLM") from e

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
            extra_body={"enable_thinking": False},
        )
        text = self._extract_response_text(response)
        if not text:
            raise ValueError(
                "Model returned no text content. Raw response:\n"
                f"{response.model_dump_json(indent=2)}"
            )
        cleaned = self._extract_json_string(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Model did not return valid JSON.\n"
                f"Raw text:\n{self._truncate(text)}\n\n"
                f"Raw response:\n{response.model_dump_json(indent=2)}"
            ) from exc

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from a Responses API object.

        Accepts either:
        1. `response.output_text` as a non-empty string.
        2. `response.output[*].content[*].text` entries, preferring items whose
           parent output object has `type == "message"`.

        Returns the joined non-empty text content, or an empty string if no text
        payload is present.
        """
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        message_chunks: List[str] = []
        chunks: List[str] = []
        for output_item in getattr(response, "output", []) or []:
            is_message = getattr(output_item, "type", None) == "message"
            for content_item in getattr(output_item, "content", []) or []:
                if getattr(content_item, "type", None) in {"output_text", "text"}:
                    value = getattr(content_item, "text", None)
                    if isinstance(value, str) and value.strip():
                        cleaned = value.strip()
                        chunks.append(cleaned)
                        if is_message:
                            message_chunks.append(cleaned)
        preferred = message_chunks if message_chunks else chunks
        return "\n".join(preferred).strip()

    def _extract_json_string(self, text: str) -> str:
        """Normalize a text response into a JSON string.

        Accepts plain JSON, fenced code blocks like ```json ... ```, or text
        that contains a top-level JSON object or array. Returns the extracted
        JSON substring when found; otherwise returns the stripped original text.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                stripped = "\n".join(lines[1:-1]).strip()

        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

        object_start = stripped.find("{")
        array_start = stripped.find("[")
        candidates: List[tuple[int, str, str]] = []
        if object_start != -1:
            candidates.append((object_start, "{", "}"))
        if array_start != -1:
            candidates.append((array_start, "[", "]"))

        for _, opener, closer in sorted(candidates, key=lambda x: x[0]):
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if end != -1 and end >= start:
                candidate = stripped[start : end + 1].strip()
                if candidate:
                    return candidate
        return stripped

    def _truncate(self, text: str, limit: int = 2000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...<truncated>..."
