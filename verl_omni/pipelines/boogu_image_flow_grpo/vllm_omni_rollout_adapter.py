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

"""Boogu-Image-Turbo rollout-side adapter for FlowGRPO.

The pinned vllm-omni has no Boogu pipeline, so this is a fully
self-contained pipeline: a plain ``nn.Module`` that loads every component
itself (Qwen3VL text encoder, vendored Boogu transformer, VAE, processor)
and runs the SDE denoise loop with per-step log-probabilities.

Because the transformer is built from plain ``nn.Linear`` layers (not
vLLM-native ones), LoRA weight sync from the trainer does NOT work — train
FULL-WEIGHT (``lora_rank=0``). Full-weight sync arrives through
``load_weights`` as ``("transformer." + name, tensor)`` pairs.
"""

import logging
import os
from typing import Any, Iterable, Literal

import torch
from diffusers import AutoencoderKL
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .boogu_model import BooguImageTransformer2DModel
from .common import (
    BOOGU_GENERATION_PROMPT_SUFFIX,
    BOOGU_VAE_SCALE_FACTOR,
    boogu_pred_to_velocity,
    boogu_timestep_from_scheduler_timestep,
    get_boogu_freqs_cis,
    setup_boogu_sigmas,
)

logger = logging.getLogger(__name__)

__all__ = ["BooguImageTurboPipelineWithLogProb"]


def _coalesce_not_none(value, default):
    return default if value is None else value


@VllmOmniPipelineBase.register("BooguImageTurboPipeline", algorithm="flow_grpo")
class BooguImageTurboPipelineWithLogProb(torch.nn.Module):
    """Self-contained Boogu-Image-Turbo rollout pipeline with log-probs.

    Constructed by vllm-omni's custom-pipeline loader as
    ``Pipeline(od_config=od_config)`` on CPU and moved to the device
    afterwards — never allocate CUDA tensors in ``__init__``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device_ = get_local_device()
        self.dtype = od_config.dtype

        model_path = od_config.model
        if not os.path.isdir(model_path):
            raise ValueError(
                f"Boogu rollout pipeline requires a local diffusers-layout checkpoint, got {model_path!r}."
            )

        self.processor = AutoProcessor.from_pretrained(os.path.join(model_path, "processor"))
        # Tokens of "<|im_start|>assistant\n" appended by verl's
        # add_generation_prompt=True tokenization; stripped in encode_prompt
        # because upstream encodes WITHOUT a generation prompt.
        self._generation_prompt_ids = self.processor.tokenizer.encode(
            BOOGU_GENERATION_PROMPT_SUFFIX, add_special_tokens=False
        )

        # The checkpoint's mllm/ is a full Qwen3VLForConditionalGeneration;
        # upstream uses its inner ``.model`` as the instruction encoder
        # (last_hidden_state). The lm_head wrapper is dropped after loading.
        mllm = Qwen3VLForConditionalGeneration.from_pretrained(model_path, subfolder="mllm", torch_dtype=self.dtype)
        self.mllm = mllm.model
        del mllm

        self.transformer = BooguImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.dtype
        )
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=self.dtype)

        self.scheduler = FlowMatchSDEDiscreteScheduler()

        # CPU rotary tables; the transformer moves them to the right device.
        self.freqs_cis = get_boogu_freqs_cis(self.transformer.config.axes_dim_rope, self.transformer.config.axes_lens)

        self.vae_scale_factor = BOOGU_VAE_SCALE_FACTOR
        self.default_sample_size = 128

        self.eval()
        self.requires_grad_(False)

    @property
    def device(self):
        return self.device_

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Apply full-weight sync from the trainer.

        At initial load the loader passes an empty iterator (components were
        already loaded in ``__init__``). During training the trainer streams
        ``("transformer." + name, tensor)`` pairs. Returning ``None`` skips
        the loader's strict all-params-loaded check.
        """
        params = dict(self.transformer.named_parameters())
        num_loaded = 0
        for name, tensor in weights:
            if not name.startswith("transformer."):
                continue
            key = name[len("transformer.") :]
            param = params.get(key)
            if param is None:
                logger.warning("load_weights: no transformer parameter named %s", key)
                continue
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))
            num_loaded += 1
        if num_loaded:
            logger.info("BooguImageTurboPipelineWithLogProb: synced %d transformer tensors", num_loaded)
        return None

    def _strip_generation_prompt(
        self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix = self._generation_prompt_ids
        n = len(suffix)
        if prompt_ids.shape[1] >= n and prompt_ids[0, -n:].tolist() == suffix:
            prompt_ids = prompt_ids[:, :-n]
            attention_mask = attention_mask[:, :-n]
        return prompt_ids, attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenized chat-template ids to instruction embeddings.

        Mirrors the upstream ``_get_instruction_feature_embeds`` T2I path:
        the Qwen3VL inner model's ``last_hidden_state`` over the full
        (right-padded) sequence, with the tokenizer attention mask.
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids)
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        prompt_ids, attention_mask = self._strip_generation_prompt(prompt_ids, attention_mask)

        prompt_ids = prompt_ids.to(self.device_)
        attention_mask = attention_mask.to(self.device_)
        with torch.no_grad():
            hidden = self.mllm(input_ids=prompt_ids, attention_mask=attention_mask).last_hidden_state
        return hidden.to(self.dtype), attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if latents is not None:
            return latents.to(self.device_)
        return randn_tensor(shape, generator=generator, device=self.device_, dtype=dtype)

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
    ):
        """SDE denoise loop collecting the FlowGRPO trajectory.

        Returns ``(latents, all_latents, all_log_probs, all_timesteps)``;
        ``all_latents`` has W+1 entries (pre-step latent plus each post-step
        latent inside the SDE window).
        """
        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []
        self.scheduler.set_begin_index(0)
        model_dtype = self.transformer.x_embedder.weight.dtype

        for i, timestep_value in enumerate(timesteps):
            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            x = latents.to(model_dtype)
            # Scheduler timestep (sigma*1000) -> Boogu signal-fraction conditioning.
            t_model = (
                boogu_timestep_from_scheduler_timestep(timestep_value)
                .to(device=x.device, dtype=x.dtype)
                .expand(x.shape[0])
            )

            model_pred = self.transformer(
                hidden_states=x,
                timestep=t_model,
                instruction_hidden_states=prompt_embeds,
                freqs_cis=self.freqs_cis,
                instruction_attention_mask=prompt_embeds_mask,
                return_dict=False,
            )
            noise_pred = boogu_pred_to_velocity(model_pred)

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            # fp32 trajectory so the trainer recomputes log-probs at full precision.
            if sde_window[0] <= i < sde_window[1]:
                all_latents.append(latents.to(torch.float32))
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1) if all_timesteps else None
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: DiffusionRequestBatch,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 4,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pt",
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 3),
        sde_type: Literal["sde", "cps", "dance_sde"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        custom_prompt: dict[str, Any] = req.prompts[0] if req.prompts else {}
        prompt_token_ids = None
        prompt_mask = None
        if isinstance(custom_prompt, dict):
            prompt_token_ids = custom_prompt.get("prompt_token_ids")
            prompt_mask = custom_prompt.get("prompt_mask")
            # The DMD student forbids CFG (guidance scales fixed to 1/1/0
            # upstream); negative_prompt_ids are intentionally ignored.

        if prompt_token_ids is None:
            # Engine warm-up ships raw text only; nothing to do.
            return DiffusionOutput(output=None, custom_output={})

        sampling_params = req.sampling_params
        height = sampling_params.height or height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps

        extra_args = sampling_params.extra_args or {}
        noise_level = _coalesce_not_none(extra_args.get("noise_level"), noise_level)
        sde_window_size = _coalesce_not_none(extra_args.get("sde_window_size"), sde_window_size)
        sde_window_range = _coalesce_not_none(extra_args.get("sde_window_range"), sde_window_range)
        sde_type = _coalesce_not_none(extra_args.get("sde_type"), sde_type)
        logprobs = _coalesce_not_none(extra_args.get("logprobs"), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device_).manual_seed(sampling_params.seed)

        if isinstance(prompt_token_ids, list):
            prompt_token_ids = torch.tensor(prompt_token_ids, device=self.device_)
        if isinstance(prompt_mask, list):
            prompt_mask = torch.tensor(prompt_mask, device=self.device_)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt_token_ids, prompt_mask)
        batch_size = prompt_embeds.shape[0]

        latents = self.prepare_latents(
            batch_size,
            self.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        setup_boogu_sigmas(self.scheduler, num_inference_steps, device=self.device_)
        timesteps = self.scheduler.timesteps

        if sde_window_size is not None:
            start = torch.randint(
                int(sde_window_range[0]),
                int(sde_window_range[1]) - int(sde_window_size) + 1,
                (1,),
                generator=generator,
                device=self.device_,
            ).item()
            sde_window = (start, start + int(sde_window_size))
        elif len(timesteps) > 1:
            # Default: SDE noise on all but the terminal step (whose noise
            # would land in the decoded image).
            sde_window = (0, len(timesteps) - 1)
        else:
            sde_window = (0, 1)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            latents,
            timesteps,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        if output_type == "latent":
            image = latents
        else:
            lat = latents.to(self.vae.dtype)
            if self.vae.config.scaling_factor is not None:
                lat = lat / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                lat = lat + self.vae.config.shift_factor
            image = self.vae.decode(lat, return_dict=False)[0]
            # [-1, 1] -> [0, 1] float; vllm-omni has no registered
            # post-process func for this architecture.
            image = (image.float() / 2 + 0.5).clamp(0, 1)
            image = image[0]  # single-sample request -> (3, H, W)

        return DiffusionOutput(
            output=image,
            custom_output={
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "all_timesteps": all_timesteps,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": None,
                "negative_prompt_embeds_mask": None,
            },
            to_cpu=True,
        )
