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

"""Z-Image training-side adapter for FlowGRPO."""

from typing import Optional

import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    apply_z_image_cfg,
    configure_z_image_scheduler,
    padded_embeds_to_list,
    prepare_latent_model_input,
    stack_z_image_model_output,
)

__all__ = ["ZImage"]


def _build_z_image_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


@DiffusionModelBase.register("ZImagePipeline", algorithm="flow_grpo")
class ZImage(DiffusionModelBase):
    """Training adapter for the Z-Image diffusion model under FlowGRPO."""

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = _build_z_image_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        configure_z_image_scheduler(
            scheduler,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ZImageTransformer2DModel,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        del module, micro_batch

        hidden_states = latents[:, step]
        timestep = (1000 - timesteps[:, step]) / 1000.0
        model_inputs = {
            "x": prepare_latent_model_input(hidden_states),
            "t": timestep,
            "cap_feats": padded_embeds_to_list(prompt_embeds, prompt_embeds_mask),
            "return_dict": False,
        }

        guidance_scale = float(model_config.pipeline.guidance_scale)
        has_negative_prompt = isinstance(negative_prompt_embeds, torch.Tensor) and isinstance(
            negative_prompt_embeds_mask, torch.Tensor
        )
        negative_model_inputs = None
        if guidance_scale > 0 and has_negative_prompt:
            negative_model_inputs = {
                "x": prepare_latent_model_input(hidden_states),
                "t": timestep,
                "cap_feats": padded_embeds_to_list(negative_prompt_embeds, negative_prompt_embeds_mask),
                "return_dict": False,
            }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ZImageTransformer2DModel,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = stack_z_image_model_output(module(**model_inputs))
        if negative_model_inputs is not None:
            neg_noise_pred = stack_z_image_model_output(module(**negative_model_inputs))
            noise_pred = apply_z_image_cfg(noise_pred, neg_noise_pred, float(model_config.pipeline.guidance_scale))
        noise_pred = -noise_pred

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
