"""Harness interface for image forgery report generation."""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

Sample = dict[str, Any]
LLMCallable = Callable[[list[dict[str, Any]]], str]


class ForgeryReportHarness(ABC):
    """Base class for candidate harnesses.

    Candidate harnesses may change prompt construction, visual artifacts,
    postprocessing, few-shot selection, and report repair. The fixed base model
    is still supplied as ``model_client``.
    """

    def __init__(self, model_client: LLMCallable, config: dict[str, Any] | None = None):
        self._model_client = model_client
        self.config = config or {}
        self._prompt_local = threading.local()

    def call_model(self, messages: list[dict[str, Any]]) -> str:
        """Call the base model and track last message metadata per thread."""

        serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
        self._prompt_local.last_prompt_len = len(serialized)
        self._prompt_local.last_prompt_hash = hashlib.md5(
            serialized.encode("utf-8")
        ).hexdigest()[:8]
        self._prompt_local.last_prompt_text = serialized
        return self._model_client(messages)

    def get_last_prompt_info(self) -> dict[str, Any]:
        return {
            "prompt_len": getattr(self._prompt_local, "last_prompt_len", None),
            "prompt_hash": getattr(self._prompt_local, "last_prompt_hash", None),
            "prompt_text": getattr(self._prompt_local, "last_prompt_text", None),
        }

    @abstractmethod
    def predict(self, sample: Sample) -> tuple[str, dict[str, Any]]:
        """Generate report before seeing current sample's ground-truth report."""

    @abstractmethod
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Update harness state after a batch of evaluated predictions."""

    def get_context_length(self) -> int:
        return len(self.get_state())

    @abstractmethod
    def get_state(self) -> str:
        """Return serializable state."""

    @abstractmethod
    def set_state(self, state: str) -> None:
        """Restore serializable state."""
