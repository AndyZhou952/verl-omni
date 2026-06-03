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

"""Z-Image vLLM-Omni rollout adapter for FlowGRPO."""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    ZImageTokenIdPromptMixin,
    apply_z_image_cfg,
    coalesce_not_none,
    configure_z_image_scheduler,
    maybe_to_cpu,
    padded_embeds_to_list,
    prepare_latent_model_input,
    stack_z_image_model_output,
)

__all__ = ["ZImagePipelineWithLogProb"]


@VllmOmniPipelineBase.register("ZImagePipeline", algorithm="flow_grpo")
class ZImagePipelineWithLogProb(ZImageTokenIdPromptMixin, ZImagePipeline):
    """Rollout pipeline for Z-Image that captures FlowGRPO trajectories."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float,
        cfg_normalization: float | bool,
        cfg_truncation: float,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: Literal["sde", "cps"],
        generator: torch.Generator | list[torch.Generator] | None,
        logprobs: bool,
    ):
        prompt_embeds_list = padded_embeds_to_list(prompt_embeds, prompt_embeds_mask)
        negative_prompt_embeds_list = (
            padded_embeds_to_list(negative_prompt_embeds, negative_prompt_embeds_mask)
            if negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
            else None
        )
        do_classifier_free_guidance = guidance_scale > 0 and negative_prompt_embeds_list is not None

        all_latents = []
        all_log_probs = []
        all_timesteps = []
        self.scheduler.set_begin_index(0)

        norm_timesteps = ((1000 - timesteps.float()) / 1000).detach().cpu().tolist()
        if not isinstance(norm_timesteps, list):
            norm_timesteps = [norm_timesteps]

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            timestep = timestep_value.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            current_guidance_scale = guidance_scale
            if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                if norm_timesteps[i] > cfg_truncation:
                    current_guidance_scale = 0.0
            apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

            latents_typed = latents.to(self.od_config.dtype)
            if apply_cfg:
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents_typed
                prompt_embeds_model_input = prompt_embeds_list
                timestep_model_input = timestep

            model_out = self.transformer(
                prepare_latent_model_input(latent_model_input),
                timestep_model_input,
                prompt_embeds_model_input,
            )
            noise_pred = stack_z_image_model_output(model_out)
            if apply_cfg:
                pos_noise_pred = noise_pred[: latents.shape[0]]
                neg_noise_pred = noise_pred[latents.shape[0] :]
                noise_pred = apply_z_image_cfg(
                    pos_noise_pred,
                    neg_noise_pred,
                    current_guidance_scale,
                    cfg_normalization,
                )
            noise_pred = -noise_pred

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents.float())
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        height: int | None = 1024,
        width: int | None = 1024,
        num_inference_steps: int = 8,
        sigmas: list[float] | None = None,
        guidance_scale: float = 0.0,
        cfg_normalization: float | bool = False,
        cfg_truncation: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        del callback_on_step_end_tensor_inputs

        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or height or 1024
        width = sampling_params.width or width or 1024
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(f"Height must be divisible by {vae_scale} (got {height}).")
        if width % vae_scale != 0:
            raise ValueError(f"Width must be divisible by {vae_scale} (got {width}).")

        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale

        extra_args = sampling_params.extra_args or {}
        cfg_normalization = coalesce_not_none(extra_args.get("cfg_normalization", None), cfg_normalization)
        cfg_truncation = coalesce_not_none(extra_args.get("cfg_truncation", None), cfg_truncation)
        noise_level = coalesce_not_none(extra_args.get("noise_level", None), noise_level)
        sde_window_size = coalesce_not_none(extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = coalesce_not_none(extra_args.get("sde_window_range", None), sde_window_range)
        sde_type = coalesce_not_none(extra_args.get("sde_type", None), sde_type)
        logprobs = coalesce_not_none(extra_args.get("logprobs", None), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        if prompt_ids is None and prompt_embeds is None:
            return DiffusionOutput(output=None, custom_output={})

        if prompt_ids is not None:
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        has_negative_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if has_negative_prompt:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            self.device,
            generator,
            latents,
        )

        timesteps = configure_z_image_scheduler(
            self.scheduler,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            device=self.device,
            sigmas=sigmas,
            vae_scale_factor=self.vae_scale_factor,
        )
        self._num_timesteps = len(timesteps)

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps) - 1)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            timesteps,
            guidance_scale,
            cfg_normalization,
            cfg_truncation,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=maybe_to_cpu(image),
            custom_output={
                "all_latents": maybe_to_cpu(all_latents),
                "all_log_probs": maybe_to_cpu(all_log_probs),
                "all_timesteps": maybe_to_cpu(all_timesteps),
                "prompt_embeds": maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": maybe_to_cpu(negative_prompt_embeds_mask),
            },
        )
