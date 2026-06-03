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

"""Shared helpers for Z-Image FlowGRPO adapters."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift

Z_IMAGE_VAE_SCALE_FACTOR = 8


def maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def coalesce_not_none(value, default):
    return default if value is None else value


def configure_z_image_scheduler(
    scheduler,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    device: str | torch.device,
    sigmas: Optional[list[float]] = None,
    vae_scale_factor: int = Z_IMAGE_VAE_SCALE_FACTOR,
) -> torch.Tensor:
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    image_seq_len = (latent_height // 2) * (latent_width // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    if sigmas is None:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    else:
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    return scheduler.timesteps


def padded_embeds_to_list(prompt_embeds: torch.Tensor, prompt_embeds_mask: torch.Tensor | None) -> list[torch.Tensor]:
    if prompt_embeds_mask is None:
        return list(prompt_embeds)
    prompt_embeds_mask = prompt_embeds_mask.to(device=prompt_embeds.device).bool()
    return [embeds[mask] for embeds, mask in zip(prompt_embeds, prompt_embeds_mask, strict=True)]


def list_embeds_to_padded(
    embeddings_list: list[torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not embeddings_list:
        raise ValueError("Z-Image prompt embeddings must contain at least one sample.")
    max_seq_len = max(embedding.size(0) for embedding in embeddings_list)
    hidden_size = embeddings_list[0].size(-1)
    device = embeddings_list[0].device
    dtype = dtype or embeddings_list[0].dtype
    prompt_embeds = torch.zeros((len(embeddings_list), max_seq_len, hidden_size), device=device, dtype=dtype)
    prompt_embeds_mask = torch.zeros((len(embeddings_list), max_seq_len), device=device, dtype=torch.long)
    for i, embedding in enumerate(embeddings_list):
        seq_len = embedding.size(0)
        prompt_embeds[i, :seq_len] = embedding.to(dtype=dtype)
        prompt_embeds_mask[i, :seq_len] = 1
    return prompt_embeds, prompt_embeds_mask


def prepare_latent_model_input(latents: torch.Tensor) -> list[torch.Tensor]:
    return list(latents.unsqueeze(2).unbind(dim=0))


def stack_z_image_model_output(model_output) -> torch.Tensor:
    if isinstance(model_output, tuple):
        model_output = model_output[0]
    if hasattr(model_output, "sample"):
        model_output = model_output.sample
    if not isinstance(model_output, list):
        raise TypeError(f"Expected Z-Image transformer output to be a list, got {type(model_output).__name__}.")
    return torch.stack([item.float() for item in model_output], dim=0).squeeze(2)


def apply_z_image_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    guidance_scale: float,
    cfg_normalization: float | bool = False,
) -> torch.Tensor:
    pred = noise_pred + guidance_scale * (noise_pred - negative_noise_pred)
    if cfg_normalization and float(cfg_normalization) > 0.0:
        ori_pos_norm = torch.linalg.vector_norm(noise_pred, dim=tuple(range(1, noise_pred.ndim)), keepdim=True)
        new_pos_norm = torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.ndim)), keepdim=True)
        max_new_norm = ori_pos_norm * float(cfg_normalization)
        scale = torch.where(
            new_pos_norm > max_new_norm,
            max_new_norm / new_pos_norm.clamp(min=1e-12),
            torch.ones_like(new_pos_norm),
        )
        pred = pred * scale
    return pred


class ZImageTokenIdPromptMixin:
    """Encode pre-tokenized Z-Image prompts for rollout adapters."""

    def _get_z_image_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = getattr(self, "_execution_device", None) or getattr(self, "device", None)
        if device is None:
            device = prompt_ids.device
        od_config = getattr(self, "od_config", None)
        dtype = dtype or getattr(od_config, "dtype", None) or self.text_encoder.dtype

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        prompt_ids = prompt_ids.to(device)
        attention_mask = attention_mask.to(device)
        prompt_ids = prompt_ids[:, :max_sequence_length]
        attention_mask = attention_mask[:, :max_sequence_length]
        if prompt_ids.shape[1] < max_sequence_length:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                raise ValueError("Z-Image tokenizer must define `pad_token_id` or `eos_token_id`.")
            padded_prompt_ids = prompt_ids.new_full((prompt_ids.shape[0], max_sequence_length), pad_token_id)
            padded_attention_mask = attention_mask.new_zeros((attention_mask.shape[0], max_sequence_length))
            padded_prompt_ids[:, : prompt_ids.shape[1]] = prompt_ids
            padded_attention_mask[:, : attention_mask.shape[1]] = attention_mask
            prompt_ids = padded_prompt_ids
            attention_mask = padded_attention_mask
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids,
            attention_mask=attention_mask.bool(),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-2]
        embeddings_list = [hidden_states[i][attention_mask[i].bool()] for i in range(hidden_states.shape[0])]
        return list_embeds_to_padded(embeddings_list, dtype=dtype)

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_embeds is None:
            if prompt_ids is None:
                raise ValueError("`prompt_ids` must be provided when `prompt_embeds` is None.")
            prompt_embeds, prompt_embeds_mask = self._get_z_image_prompt_embeds(
                prompt_ids,
                attention_mask=attention_mask,
                max_sequence_length=max_sequence_length,
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds, prompt_embeds_mask
