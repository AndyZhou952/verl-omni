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

"""Boogu-Image-Turbo training-side adapter for diffusers-based diffusion RL."""

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .boogu_model import BooguImageTransformer2DModel
from .common import (
    boogu_pred_to_velocity,
    boogu_timestep_from_scheduler_timestep,
    get_boogu_freqs_cis,
    setup_boogu_sigmas,
)

logger = logging.getLogger(__name__)

__all__ = ["BooguImage"]


@DiffusionModelBase.register("BooguImageTurboPipeline", algorithm="flow_grpo")
class BooguImage(DiffusionModelBase):
    """Training adapter for the Boogu-Image-Turbo (DMD few-step) model.

    The transformer is the vendored ``BooguImageTransformer2DModel``
    (diffusers ``ModelMixin`` with ``_no_split_modules``), loaded via
    ``build_module`` because stock ``diffusers.AutoModel`` does not know the
    architecture. Sigma schedule, timestep convention, and prediction-target
    conversion live in ``common.py`` shared with the rollout adapter.
    """

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> torch.nn.Module:
        logger.info("Loading BooguImageTransformer2DModel from %s", model_config.local_path)
        return BooguImageTransformer2DModel.from_pretrained(
            model_config.local_path, subfolder="transformer", torch_dtype=torch_dtype
        )

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchSDEDiscreteScheduler:
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        setup_boogu_sigmas(scheduler, model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module: BooguImageTransformer2DModel,
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
        hidden_states = latents[:, step]
        timestep = boogu_timestep_from_scheduler_timestep(timesteps[:, step]).to(hidden_states.dtype)
        freqs_cis = get_boogu_freqs_cis(module.config.axes_dim_rope, module.config.axes_lens)

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "instruction_hidden_states": prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": prompt_embeds_mask,
            "return_dict": False,
        }
        # The DMD student forbids CFG; no negative branch.
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module: BooguImageTransformer2DModel,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # The transformer's forward(return_dict=False) returns a bare tensor,
        # not a tuple — the default DiffusionModelBase.forward would index
        # into the batch dimension.
        return module(**model_inputs)

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: BooguImageTransformer2DModel,
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

        model_pred = cls.forward(module, model_config, model_inputs)
        noise_pred = boogu_pred_to_velocity(model_pred)

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
