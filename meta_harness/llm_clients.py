"""OpenAI-compatible clients for Alibaba ModelStudio / DashScope models."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any


DEFAULT_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAX_DATA_URI_CHARS = 5_000_000


def _encode_data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _detect_mime(raw: bytes, suffix: str) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "image/png" if suffix.lower() == ".png" else "image/jpeg"


def image_to_data_url(path: str | Path, max_data_uri_chars: int | None = DEFAULT_MAX_DATA_URI_CHARS) -> str:
    image_path = Path(path)
    raw = image_path.read_bytes()
    mime = _detect_mime(raw, image_path.suffix)
    data_url = _encode_data_url(raw, mime)
    if max_data_uri_chars is None or len(data_url) <= max_data_uri_chars:
        return data_url

    try:
        from PIL import Image
    except ModuleNotFoundError:
        return data_url

    with Image.open(image_path) as image:
        candidate = image.convert("RGB")

    best_data_url = data_url
    best_len = len(data_url)
    for quality in (85, 75, 65, 55, 45, 35, 30, 25, 20, 15, 10, 5):
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
        candidate_url = _encode_data_url(buffer.getvalue(), "image/jpeg")
        candidate_len = len(candidate_url)
        if candidate_len < best_len:
            best_data_url = candidate_url
            best_len = candidate_len
        if candidate_len <= max_data_uri_chars:
            return candidate_url
    return best_data_url


class ModelStudioChatClient:
    """Thin OpenAI-compatible chat client.

    Defaults to Alibaba ModelStudio's DashScope compatible endpoint. This client
    is intentionally small so proposed harnesses can replace prompt/message logic
    without changing the fixed model identity.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "LINKAPI_API_KEY",
        api_base: str = DEFAULT_API_BASE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.extra_body = {"enable_thinking": False}
        self.timeout = timeout
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _client(self):
        from openai import OpenAI

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        return OpenAI(api_key=api_key, base_url=self.api_base)

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self._client().chat.completions.create(**kwargs)
        if isinstance(response, str):
            return response
        self.total_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.total_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        choices = getattr(response, "choices", None)
        if not choices:
            return str(response)
        message = getattr(choices[0], "message", None)
        if message is None:
            return str(response)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return str(content or "")

    def get_usage(self) -> dict[str, int]:
        return {
            "calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }


def make_stub_client(response: str) -> ModelStudioChatClient:
    class StubClient(ModelStudioChatClient):
        def __init__(self, text: str):
            self.text = text
            self.model = "stub"
            self.total_calls = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0

        def __call__(self, messages: list[dict[str, Any]]) -> str:
            self.total_calls += 1
            self.total_input_tokens += len(str(messages))
            self.total_output_tokens += len(self.text)
            return self.text

    return StubClient(response)
