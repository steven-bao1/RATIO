"""Capture exact top-p next-token distributions during vLLM decoding."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import msgpack
import torch


ONLINE_SCHEMA_VERSION = 6
ONLINE_POST_SAMPLER_SCHEMA_VERSION = 5
ONLINE_NUCLEUS_METHODS = ("minimal", "vllm")


def _prompt_key(prompt_token_ids: list[int]) -> bytes:
    digest = hashlib.sha256()
    for token_id in prompt_token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.digest()[:16]


class OnlineNucleusRecorderLogitsProcessor:
    def __init__(
        self,
        output_path: str,
        *,
        temperature: float,
        top_p: float,
        initial_top_k: int = 128,
        nucleus_method: str = "vllm",
        replay_target_token_ids: tuple[int, ...] | None = None,
        expected_prompt_token_ids: tuple[int, ...] | None = None,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if initial_top_k <= 0:
            raise ValueError("initial_top_k must be positive")
        if nucleus_method not in ONLINE_NUCLEUS_METHODS:
            raise ValueError(
                f"nucleus_method must be one of {ONLINE_NUCLEUS_METHODS}"
            )
        self.output_path = str(Path(output_path).resolve())
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.initial_top_k = int(initial_top_k)
        self.nucleus_method = str(nucleus_method)
        self.replay_target_token_ids = replay_target_token_ids
        self.expected_prompt_token_ids = expected_prompt_token_ids
        if (replay_target_token_ids is None) != (expected_prompt_token_ids is None):
            raise ValueError(
                "replay targets and expected prompt tokens must be provided together"
            )
        if replay_target_token_ids is not None and not replay_target_token_ids:
            raise ValueError("replay target token IDs must not be empty")
        self._replay_prompt_validated = False
        self._descriptor: int | None = None
        self._last_position_by_prompt: dict[bytes, int] = {}
        self._pending_vllm: dict[tuple[bytes, int], tuple] = {}
        self._temperature_by_dtype: dict[torch.dtype, float] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_descriptor"] = None
        state["_pending_vllm"] = {}
        return state

    def __del__(self) -> None:
        descriptor = getattr(self, "_descriptor", None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _append(self, record: list) -> None:
        if self._descriptor is None:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._descriptor = os.open(
                self.output_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
        payload = memoryview(msgpack.packb(record, use_bin_type=True))
        while payload:
            written = os.write(self._descriptor, payload)
            if written <= 0:
                raise OSError("Failed to append online nucleus record")
            payload = payload[written:]

    def _write_support(
        self,
        *,
        key: bytes,
        position: int,
        kept_logits: torch.Tensor,
        kept_ids: torch.Tensor,
        raw_logprobs: torch.Tensor,
        sampling_logprobs: torch.Tensor,
        replay_target: tuple[int, float, float, int, float] | None = None,
    ) -> None:
        nucleus_ids = kept_ids.to(torch.int32).cpu().numpy()
        sampling_logprobs = sampling_logprobs.index_select(0, kept_ids).cpu().numpy()
        truncated_logprobs = torch.log_softmax(
            kept_logits, dim=-1
        ).to(torch.float32).cpu().numpy()
        raw_logprobs = raw_logprobs.index_select(0, kept_ids).cpu().numpy()
        record = [
            (
                ONLINE_SCHEMA_VERSION
                if replay_target is not None
                else ONLINE_POST_SAMPLER_SCHEMA_VERSION
            ),
            key,
            position,
            nucleus_ids.astype("<i4", copy=False).tobytes(),
            raw_logprobs.astype("<f4", copy=False).tobytes(),
            sampling_logprobs.astype("<f4", copy=False).tobytes(),
            truncated_logprobs.astype("<f4", copy=False).tobytes(),
        ]
        if replay_target is not None:
            (
                target_id,
                target_raw_logprob,
                target_sampling_logprob,
                target_rank,
                target_cumulative_mass_before,
            ) = replay_target
            record.extend(
                [
                    target_id,
                    target_raw_logprob,
                    target_sampling_logprob,
                    target_rank,
                    target_cumulative_mass_before,
                ]
            )
        self._append(record)
        # A reference corpus can contain thousands of trajectories. Release each
        # replay descriptor at its final position rather than keeping all open.
        if replay_target is not None and position + 1 == len(self.replay_target_token_ids):
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None

    @torch.no_grad()
    def __call__(
        self,
        prompt_token_ids: list[int],
        token_ids: list[int] | torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits is None:
            raise TypeError("Online recording requires vLLM's three-argument signature")
        key = _prompt_key(prompt_token_ids)
        position = len(token_ids)
        previous = self._last_position_by_prompt.get(key)
        if previous is not None:
            if position == previous:
                return logits
            if position != previous + 1:
                raise RuntimeError(
                    f"Non-contiguous online positions for {key.hex()}: "
                    f"previous={previous}, current={position}"
                )
        self._last_position_by_prompt[key] = position

        raw = logits.float()
        # Stable log_softmax avoids cancellation in logits - logsumexp(logits),
        # and avoids reconstructing raw logits by dividing then multiplying.
        # A near-certain token must never acquire a positive log probability.
        raw_logprobs = torch.log_softmax(raw, dim=-1)
        effective_temperature = self.temperature
        if self.nucleus_method == "vllm":
            # Pinned vLLM constructs SamplingTensors in the incoming logits dtype,
            # THEN casts logits to float32 for division. BF16 therefore represents
            # requested 0.6 as 0.6015625. Both online normalizers and pre-forcing
            # replay snapshots must use that same value, without changing sampling.
            if logits.dtype not in self._temperature_by_dtype:
                self._temperature_by_dtype[logits.dtype] = float(torch.tensor(self.temperature, dtype=logits.dtype))
                print(f"Exact nucleus recorder: logits_dtype={logits.dtype}, "
                      f"requested_temperature={self.temperature}, effective_temperature={self._temperature_by_dtype[logits.dtype]}, "
                      f"requested_top_p={self.top_p}, effective_top_p={float(torch.tensor(self.top_p, dtype=logits.dtype))}", flush=True)
            effective_temperature = self._temperature_by_dtype[logits.dtype]
            if not math.isfinite(effective_temperature) or effective_temperature <= 0:
                raise ValueError("Temperature is invalid in the sampler parameter dtype")
        sampling = raw / effective_temperature
        sampling_logprobs = torch.log_softmax(sampling, dim=-1)

        if self.nucleus_method == "vllm":
            pending_key = (key, position)
            if pending_key in self._pending_vllm:
                raise RuntimeError(
                    f"Duplicate pending vLLM record for {key.hex()}:{position}"
                )
            replay_target = None
            replay_sampling_logits = None
            if self.replay_target_token_ids is not None:
                if not self._replay_prompt_validated:
                    actual_prompt = tuple(int(value) for value in prompt_token_ids)
                    if actual_prompt != self.expected_prompt_token_ids:
                        raise ValueError("Replay scorer received an unexpected prompt")
                    self._replay_prompt_validated = True
                if position >= len(self.replay_target_token_ids):
                    raise ValueError(
                        f"Replay position {position} exceeds target length "
                        f"{len(self.replay_target_token_ids)}"
                    )
                target_id = int(self.replay_target_token_ids[position])
                if not 0 <= target_id < raw.shape[-1]:
                    raise ValueError(
                        f"Replay target token {target_id} is outside the vocabulary"
                    )
                replay_target = (
                    target_id,
                    float(raw_logprobs[target_id].item()),
                    float(sampling_logprobs[target_id].item()),
                    int((raw > raw[target_id]).sum().item()) + 1,
                    float(
                        torch.exp(
                            sampling_logprobs[raw > raw[target_id]]
                        ).sum().item()
                    ),
                )
                # A later logits processor forces the fixed trajectory token.
                # Preserve the untouched model distribution for the sampler hook.
                replay_sampling_logits = sampling.detach().clone()
            self._pending_vllm[pending_key] = (
                raw_logprobs,
                sampling_logprobs,
                replay_sampling_logits,
                replay_target,
            )
            return logits

        vocab_size = int(sampling.shape[0])
        top_k = min(self.initial_top_k, vocab_size)
        while True:
            top_logits, top_ids = torch.topk(
                sampling, k=top_k, largest=True, sorted=True
            )
            top_probabilities = torch.exp(
                sampling_logprobs.index_select(0, top_ids)
            )
            cumulative = top_probabilities.cumsum(dim=-1)
            if bool(cumulative[-1] >= self.top_p) or top_k == vocab_size:
                break
            top_k = min(vocab_size, top_k * 2)
        boundaries = torch.nonzero(cumulative >= self.top_p, as_tuple=False)
        nucleus_size = int(boundaries[0].item()) + 1 if boundaries.numel() else top_k
        self._write_support(
            key=key,
            position=position,
            kept_logits=top_logits[:nucleus_size],
            kept_ids=top_ids[:nucleus_size],
            raw_logprobs=raw_logprobs,
            sampling_logprobs=sampling_logprobs,
        )
        return logits

    def vllm_pre_intervention_sampling_logits(
        self,
        prompt_token_ids: list[int],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the saved pre-forcing sampling logits for fixed-prefix replay."""
        if self.nucleus_method != "vllm":
            return None
        pending = self._pending_vllm.get(
            (_prompt_key(prompt_token_ids), len(token_ids))
        )
        return None if pending is None else pending[2]

    @torch.no_grad()
    def record_vllm_post_top_p(
        self,
        prompt_token_ids: list[int],
        token_ids: list[int] | torch.Tensor,
        masked_sampling_logits: torch.Tensor,
    ) -> None:
        """Record the exact finite support passed by vLLM to softmax/sampling."""
        if self.nucleus_method != "vllm":
            return
        key = _prompt_key(prompt_token_ids)
        position = len(token_ids)
        pending_key = (key, position)
        normalizers = self._pending_vllm.pop(pending_key, None)
        if normalizers is None:
            raise RuntimeError(
                f"No pending pre-top-p record for {key.hex()}:{position}"
            )
        (
            raw_logprobs,
            sampling_logprobs,
            _,
            replay_target,
        ) = normalizers
        kept_ids = torch.nonzero(
            torch.isfinite(masked_sampling_logits), as_tuple=False
        ).squeeze(-1)
        if kept_ids.numel() == 0:
            raise RuntimeError("vLLM top-p mask removed every vocabulary token")
        kept_logits = masked_sampling_logits.index_select(0, kept_ids)
        order = torch.argsort(kept_logits, descending=True)
        self._write_support(
            key=key,
            position=position,
            kept_logits=kept_logits.index_select(0, order),
            kept_ids=kept_ids.index_select(0, order),
            raw_logprobs=raw_logprobs,
            sampling_logprobs=sampling_logprobs,
            replay_target=replay_target,
        )


def build_online_nucleus_recorder(
    output_path: str | None,
    *,
    temperature: float,
    top_p: float,
    initial_top_k: int = 128,
    nucleus_method: str = "vllm",
    replay_target_token_ids: tuple[int, ...] | None = None,
    expected_prompt_token_ids: tuple[int, ...] | None = None,
) -> OnlineNucleusRecorderLogitsProcessor | None:
    if output_path is None:
        return None
    return OnlineNucleusRecorderLogitsProcessor(
        output_path,
        temperature=temperature,
        top_p=top_p,
        initial_top_k=initial_top_k,
        nucleus_method=nucleus_method,
        replay_target_token_ids=replay_target_token_ids,
        expected_prompt_token_ids=expected_prompt_token_ids,
    )
