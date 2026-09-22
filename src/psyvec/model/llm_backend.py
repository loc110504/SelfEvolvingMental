"""One chat interface over the three ways this project runs a model.

``openai``
    api.openai.com. Supports ``response_format={"type": "json_object"}``.
``ollama``
    A local Ollama server's OpenAI-compatible endpoint
    (``http://localhost:11434/v1`` by default). Like ``openai``, it accepts
    ``response_format={"type": "json_object"}`` — unlike vLLM, it does not
    understand ``extra_body={"guided_json": ...}``, so it is handled with the
    ``openai`` structured-output path rather than the ``vllm`` one.
``vllm`` (and any other OpenAI-compatible server: LM Studio, OpenRouter, TGI)
    Same wire protocol, different guarantees. vLLM expresses constrained
    decoding through ``extra_body={"guided_json": ...}`` rather than
    ``response_format``, and Qwen3-family chat templates need
    ``chat_template_kwargs={"enable_thinking": false}`` or the reply is
    consumed by a reasoning trace before the JSON appears.
``local``
    ``transformers`` in-process, for a run with no server at all.

The differences above are capability probes, not configuration the caller
should have to get right: every request starts with the strongest structured-
output mode the target plausibly supports and degrades once, permanently, when
the endpoint rejects it. That matters because the failure is silent otherwise —
an endpoint that ignores ``response_format`` returns prose, the JSON parse
fails, and the pipeline records a parse failure that looks like a model quality
problem.

``openai`` and ``vllm`` are imported lazily so this module stays importable
with no optional dependency installed, which keeps the offline test suite
runnable.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

__all__ = [
    "BackendKind",
    "ChatBackend",
    "GenerationRequest",
    "LocalTransformersBackend",
    "OpenAICompatibleBackend",
    "UsageLedger",
    "build_backend",
]

logger = logging.getLogger(__name__)

BackendKind = Literal["openai", "ollama", "vllm", "local"]

#: Model-name fragments whose chat template emits a reasoning trace by default.
#: ``gpt-oss`` (OpenAI's open-weight reasoning model, served here through
#: Ollama) emits its chain of thought through Harmony-format channels; Ollama's
#: OpenAI-compatible endpoint has been observed to pass an ``analysis`` channel
#: through as ordinary message content rather than always separating it out, so
#: this is treated the same defensive way as the other reasoning models: the
#: request asks for no thinking trace, and ``response_parsing.strip_reasoning``
#: is the second line of defense if one leaks through anyway.
_REASONING_HINTS = (
    "qwen3", "qwq", "deepseek-r1", "-r1", "thinking", "reason", "gpt-oss",
)

#: Substrings of a provider error that mean "retry", not "this request is bad".
_RETRYABLE_MARKERS = (
    "rate limit", "ratelimit", "too many requests", "timeout", "timed out",
    "temporarily unavailable", "overloaded", "connection", "connection reset",
    "bad gateway", "service unavailable", "gateway timeout",
    "500", "502", "503", "504", "529",
)
_RETRYABLE_TYPE_MARKERS = (
    "ratelimit", "timeout", "connection", "internalserver", "apierror",
    "apiconnection", "serviceunavailable",
)


def looks_like_reasoning_model(model: str) -> bool:
    """Whether ``model``'s chat template likely emits chain-of-thought first."""
    lowered = model.lower()
    return any(hint in lowered for hint in _REASONING_HINTS)


def is_retryable(error: Exception) -> bool:
    """Whether a provider exception is transient."""
    type_name = type(error).__name__.lower()
    if any(marker in type_name for marker in _RETRYABLE_TYPE_MARKERS):
        return True
    return any(marker in str(error).lower() for marker in _RETRYABLE_MARKERS)


@dataclass
class UsageLedger:
    """Thread-safe token/cost accumulator shared across concurrent workers."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    retries: int = 0
    failures: int = 0
    truncations: int = 0
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self, input_tokens: int, output_tokens: int, *, truncated: bool = False
    ) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.calls += 1
            self.truncations += int(truncated)

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    @property
    def cost(self) -> float:
        return (
            self.input_tokens * self.input_cost_per_1m
            + self.output_tokens * self.output_cost_per_1m
        ) / 1_000_000

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "failed_calls": self.failures,
            "truncated_generations": self.truncations,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.cost, 4),
        }


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One chat turn. ``json_schema`` requests constrained decoding when available."""

    system: str
    user: str
    max_tokens: int = 700
    temperature: float = 0.0
    json_schema: Mapping[str, Any] | None = None


class ChatBackend(Protocol):
    """What the assessment pipeline needs from a model, and nothing more."""

    model: str

    def generate(self, request: GenerationRequest) -> str: ...


class OpenAICompatibleBackend:
    """Chat completions against api.openai.com, vLLM, or any compatible server."""

    def __init__(
        self,
        model: str,
        *,
        kind: BackendKind = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        ledger: UsageLedger | None = None,
        max_retries: int = 6,
        timeout: float = 180.0,
        enable_thinking: bool | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the 'openai' package is required for the openai/vllm backends: "
                "pip install 'openai>=1.0.0'"
            ) from error

        self.model = model
        self.kind = kind
        self.ledger = ledger if ledger is not None else UsageLedger()
        self.max_retries = max_retries

        # A local server needs no real credential but the client insists on one.
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if kind != "openai" and not resolved_key:
            resolved_key = "EMPTY"
        if not resolved_key:
            raise RuntimeError(
                "no API key: set OPENAI_API_KEY or pass api_key explicitly"
            )

        client_kwargs: dict[str, Any] = {"api_key": resolved_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        if enable_thinking is None and looks_like_reasoning_model(model):
            enable_thinking = False
            logger.info(
                "'%s' looks like a reasoning model; disabling the thinking "
                "trace so the token budget reaches the answer.",
                model,
            )
        self.enable_thinking = enable_thinking

        # Capability flags, downgraded permanently on first rejection so one
        # bad request does not cost every later call a wasted round trip.
        self._structured_output = True
        self._template_kwargs = enable_thinking is not None

    def _build_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        extra_body: dict[str, Any] = {}
        if self._structured_output and request.json_schema is not None:
            if self.kind == "vllm":
                extra_body["guided_json"] = dict(request.json_schema)
            else:
                kwargs["response_format"] = {"type": "json_object"}
        elif self._structured_output and self.kind == "openai":
            kwargs["response_format"] = {"type": "json_object"}
        if self._template_kwargs and self.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _downgrade(self, message: str) -> bool:
        """Turn off one unsupported capability; report whether anything changed."""
        lowered = message.lower()
        if self._structured_output and (
            "response_format" in lowered
            or "guided_json" in lowered
            or "guided decoding" in lowered
            or "json_object" in lowered
        ):
            logger.warning(
                "endpoint rejected structured output; continuing without it "
                "(the parsers tolerate prose around the JSON object)"
            )
            self._structured_output = False
            return True
        if self._template_kwargs and "chat_template_kwargs" in lowered:
            logger.warning("endpoint rejected chat_template_kwargs; dropping it")
            self._template_kwargs = False
            return True
        return False

    def generate(self, request: GenerationRequest) -> str:
        last_error: Exception | None = None
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    **self._build_kwargs(request)
                )
            except Exception as error:  # noqa: BLE001 - provider taxonomies differ
                message = str(error)
                if self._downgrade(message):
                    continue  # a capability downgrade is not a retry
                if not is_retryable(error):
                    raise
                last_error = error
                attempt += 1
                self.ledger.record_retry()
                delay = min(2.0**attempt, 30.0) * (0.5 + random.random())
                logger.warning(
                    "retryable error (%d/%d), sleeping %.1fs: %s",
                    attempt, self.max_retries, delay, message[:200],
                )
                time.sleep(delay)
                continue

            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            self.ledger.record(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                truncated=getattr(choice, "finish_reason", None) == "length",
            )
            return (choice.message.content or "").strip()

        self.ledger.record_failure()
        raise RuntimeError(f"exhausted {self.max_retries} retries: {last_error}")


class LocalTransformersBackend:
    """In-process ``transformers`` generation, for a run with no server."""

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        ledger: UsageLedger | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the local backend needs torch and transformers: "
                "pip install '.[eval]'"
            ) from error

        self.model = model
        self.ledger = ledger if ledger is not None else UsageLedger()
        if enable_thinking is None and looks_like_reasoning_model(model):
            enable_thinking = False
        self.enable_thinking = enable_thinking

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self._torch = torch
        logger.info("loading %s on %s", model, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        dtype = torch.float32 if device in {"cpu", "mps"} else torch.bfloat16
        self.hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(
            device
        )
        self.hf_model.eval()

    def generate(self, request: GenerationRequest) -> str:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        template_kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            generated = self.hf_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=request.temperature > 0.0,
                temperature=request.temperature or None,
                top_p=0.9 if request.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        self.ledger.record(
            int(inputs.input_ids.shape[1]),
            int(new_tokens.shape[1]),
            truncated=int(new_tokens.shape[1]) >= request.max_tokens,
        )
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return str(decoded[0]).strip()


#: Where each backend kind points when the caller gives no explicit base URL.
_DEFAULT_BASE_URLS: dict[str, str | None] = {
    "openai": None,  # the client's own default
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}


def build_backend(
    kind: BackendKind,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    ledger: UsageLedger | None = None,
    device: str | None = None,
    enable_thinking: bool | None = None,
    max_retries: int = 6,
    timeout: float = 180.0,
) -> ChatBackend:
    """Construct the backend named by ``kind``."""
    if kind == "local":
        return LocalTransformersBackend(
            model, device=device, ledger=ledger, enable_thinking=enable_thinking
        )
    if kind not in _DEFAULT_BASE_URLS:
        raise ValueError(f"unknown backend kind: {kind!r}")
    return OpenAICompatibleBackend(
        model,
        kind=kind,
        api_key=api_key,
        base_url=base_url or _DEFAULT_BASE_URLS[kind],
        ledger=ledger,
        max_retries=max_retries,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )
