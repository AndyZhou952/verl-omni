# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared constants and helpers for the Boogu-Image-Turbo FlowGRPO adapters.

Single source of truth for the sigma schedule, the timestep convention, and
the model-output-to-velocity conversion. Both the rollout adapter and the
training adapter MUST use these helpers — a mismatch silently corrupts the
FlowGRPO importance ratios.

Upstream conventions (github.com/boogu-project/Boogu-Image, DMD turbo path):

- The transformer is conditioned on the **signal fraction** ``t ∈ [0, 1]``
  (t=0 pure noise, t=1 clean image); ``timestep_scale=1000`` is applied
  inside the model's sinusoidal embedding.
- The model predicts ``x0 − ε`` (Lumina lineage). verl's
  ``FlowMatchSDEDiscreteScheduler`` expects flow-matching velocity
  ``v = dx/dσ_noise = ε − x0``, so the prediction is negated.
- The DMD student schedule is ``t_i = linspace(conditioning_sigma, 1, N+1)[:-1]``
  in signal fractions, i.e. noise fractions ``σ_i = 1 − t_i`` descending from
  ``1 − conditioning_sigma``.
"""

import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

BOOGU_VAE_SCALE_FACTOR = 8
# Height/width must be divisible by vae_scale_factor * patch_size.
BOOGU_IMAGE_MULTIPLE = BOOGU_VAE_SCALE_FACTOR * 2

# Upstream `dmd_conditioning_sigma` (signal fraction of the first denoising step).
BOOGU_DMD_CONDITIONING_SIGMA = 0.001

# System prompt the deployed pipeline hard-codes for T2I encoding
# (BooguImagePipeline.SYSTEM_PROMPT_4_T2I_UNIFIED). The data preprocessor
# must reproduce it exactly.
BOOGU_SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)

# Upstream encodes prompts WITHOUT a generation prompt; verl's agent loop
# tokenizes with add_generation_prompt=True. This suffix ("<|im_start|>assistant\n"
# under the checkpoint's Qwen3VL chat template) is stripped before encoding.
BOOGU_GENERATION_PROMPT_SUFFIX = "<|im_start|>assistant\n"


def boogu_dmd_sigmas(num_inference_steps: int, conditioning_sigma: float = BOOGU_DMD_CONDITIONING_SIGMA) -> list[float]:
    """Noise-fraction sigmas (descending) reproducing the upstream DMD schedule.

    The first sigma is pinned to exactly ``1.0`` (instead of
    ``1 − conditioning_sigma``) so that ``FlowMatchSDEDiscreteScheduler``'s
    ``sigma == 1`` guard replaces it with ``sigma_max`` in the SDE std —
    otherwise the first-step std would be ``sqrt((1−c)/c) · noise_level``.
    The model-side conditioning timestep is clamped back to
    ``conditioning_sigma`` by :func:`boogu_timestep_from_scheduler_timestep`,
    so the transformer sees the exact upstream conditioning values.
    """
    if num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    # Mirror upstream `_build_dmd_student_sigmas` (signal fractions, ascending).
    t = torch.linspace(conditioning_sigma, 1.0, num_inference_steps + 1, dtype=torch.float32)[:-1]
    sigmas = (1.0 - t).tolist()
    sigmas[0] = 1.0
    return sigmas


def setup_boogu_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_inference_steps: int,
    device: str | torch.device | None = None,
) -> list[float]:
    """Configure *scheduler* with the Boogu DMD schedule; returns the sigmas."""
    sigmas = boogu_dmd_sigmas(num_inference_steps)
    scheduler.set_shift(1.0)  # identity shift — sigmas are already final
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    scheduler.set_begin_index(0)
    return sigmas


def boogu_timestep_from_scheduler_timestep(
    timestep: torch.Tensor,
    conditioning_sigma: float = BOOGU_DMD_CONDITIONING_SIGMA,
) -> torch.Tensor:
    """Convert scheduler timesteps (``sigma*1000``, σ = noise fraction) to the
    model's conditioning timestep (signal fraction, clamped to the DMD
    conditioning sigma at the first step where σ was pinned to 1.0)."""
    return (1.0 - timestep.float() / 1000.0).clamp(min=conditioning_sigma)


def boogu_pred_to_velocity(model_pred: torch.Tensor) -> torch.Tensor:
    """Boogu predicts ``x0 − ε``; the SDE scheduler consumes ``v = ε − x0``."""
    return -model_pred


def get_boogu_freqs_cis(axes_dim_rope, axes_lens, theta: int = 10000) -> list[torch.Tensor]:
    """Precomputed rotary tables for ``BooguImageTransformer2DModel.forward``.

    Cached per (axes_dim, axes_lens, theta); the model moves them to the
    right device internally. Mirrors the upstream pipeline, which computes
    them once outside the transformer.
    """
    key = (tuple(axes_dim_rope), tuple(axes_lens), theta)
    cached = _FREQS_CIS_CACHE.get(key)
    if cached is None:
        from .boogu_model import BooguImageRotaryPosEmbed

        cached = BooguImageRotaryPosEmbed.get_freqs_cis(list(axes_dim_rope), list(axes_lens), theta=theta)
        _FREQS_CIS_CACHE[key] = cached
    return cached


_FREQS_CIS_CACHE: dict[tuple, list[torch.Tensor]] = {}
