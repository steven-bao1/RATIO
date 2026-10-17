"""Utilities for replaying a fixed response through vLLM decoding."""

from __future__ import annotations

import torch


class ForcedTrajectoryLogitsProcessor:
    """Force one known response while preserving it for prefix construction."""

    def __init__(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        response_token_ids: tuple[int, ...],
    ) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt token IDs must not be empty")
        if not response_token_ids:
            raise ValueError("response token IDs must not be empty")
        self.prompt_token_ids = prompt_token_ids
        self.response_token_ids = response_token_ids
        self._prompt_validated = False

    def __call__(
        self,
        prompt_token_ids: list[int],
        token_ids: list[int] | torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits is None:
            raise TypeError("Fixed replay requires vLLM's three-argument signature")
        if not self._prompt_validated:
            actual_prompt = tuple(int(value) for value in prompt_token_ids)
            if actual_prompt != self.prompt_token_ids:
                raise ValueError("Fixed replay received an unexpected prompt")
            self._prompt_validated = True

        position = len(token_ids)
        if position >= len(self.response_token_ids):
            raise ValueError(
                f"Replay position {position} exceeds response length "
                f"{len(self.response_token_ids)}"
            )
        target_id = int(self.response_token_ids[position])
        if not 0 <= target_id < logits.shape[-1]:
            raise ValueError(
                f"Replay target token {target_id} is outside the vocabulary"
            )

        logits.fill_(-float("inf"))
        logits[target_id] = 0.0
        return logits
