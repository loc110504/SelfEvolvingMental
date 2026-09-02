"""Small text-generation port for legacy memory and simulator call sites."""

from __future__ import annotations

from dataclasses import dataclass

from psyvec.model.backend import Message, ModelBackend, ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class LegacyModelPort:
    """Translate legacy system/user prompts into the shared backend contract."""

    backend: ModelBackend
    model: str
    prompt_version: str
    temperature: float = 0.0
    max_tokens: int | None = None

    def complete(
        self,
        *,
        request_id: str,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelResponse:
        return self.backend.generate(
            ModelRequest(
                request_id=request_id,
                role=role,
                messages=(
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ),
                model=self.model,
                prompt_version=self.prompt_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )
