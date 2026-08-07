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
"""CPU tests for the Boogu-Image-Turbo FlowGRPO adapters.

Necessity: the Boogu integration maps the upstream DMD student sampler
(signal-fraction timesteps, ``x0 − ε`` prediction, fresh-noise renoise) onto
verl's noise-fraction SDE scheduler. A silent mismatch in the sigma schedule,
timestep conversion, or velocity sign corrupts FlowGRPO importance ratios
without crashing, so the numeric contracts in ``common.py`` are pinned here
against upstream reference values, plus registry resolution and the
generation-prompt stripping in the rollout adapter's encode path.
"""

import torch

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    BOOGU_DMD_CONDITIONING_SIGMA,
    BOOGU_GENERATION_PROMPT_SUFFIX,
    boogu_dmd_sigmas,
    boogu_pred_to_velocity,
    boogu_timestep_from_scheduler_timestep,
    setup_boogu_sigmas,
)
from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import BooguImage
from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import (
    BooguImageTurboPipelineWithLogProb,
)
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

ARCHITECTURE = "BooguImageTurboPipeline"
ALGORITHM = "flow_grpo"


def _upstream_dmd_signal_fractions(num_steps: int) -> torch.Tensor:
    """Reference ladder from upstream ``_build_dmd_student_sigmas`` (signal fractions, ascending)."""
    return torch.linspace(BOOGU_DMD_CONDITIONING_SIGMA, 1.0, num_steps + 1, dtype=torch.float32)[:-1]


class TestRegistry:
    def test_training_adapter_resolves(self):
        assert DiffusionModelBase.get_class_by_name(ARCHITECTURE, ALGORITHM) is BooguImage

    def test_rollout_adapter_resolves(self):
        assert VllmOmniPipelineBase.get_class(ARCHITECTURE, ALGORITHM) is BooguImageTurboPipelineWithLogProb

    def test_rollout_pipeline_path_is_importable_dotted_path(self):
        path = VllmOmniPipelineBase.get_pipeline_path(ARCHITECTURE, ALGORITHM)
        assert path == (
            "verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter.BooguImageTurboPipelineWithLogProb"
        )


class TestSigmaSchedule:
    def test_matches_upstream_dmd_ladder_except_pinned_first(self):
        for num_steps in (1, 2, 4, 8):
            sigmas = boogu_dmd_sigmas(num_steps)
            expected = (1.0 - _upstream_dmd_signal_fractions(num_steps)).tolist()

            assert len(sigmas) == num_steps
            # First sigma pinned to exactly 1.0 (SDE scheduler's sigma==1 guard);
            # upstream's is 1 - conditioning_sigma = 0.999.
            assert sigmas[0] == 1.0
            assert abs(expected[0] - (1.0 - BOOGU_DMD_CONDITIONING_SIGMA)) < 1e-6
            torch.testing.assert_close(torch.tensor(sigmas[1:]), torch.tensor(expected[1:]), atol=1e-6, rtol=0.0)

    def test_four_step_reference_values(self):
        # Hardcoded from the upstream demo config: linspace(0.001, 1, 5)[:-1]
        # as signal fractions -> noise fractions, first pinned to 1.0.
        torch.testing.assert_close(
            torch.tensor(boogu_dmd_sigmas(4)),
            torch.tensor([1.0, 0.74925, 0.4995, 0.24975]),
            atol=1e-5,
            rtol=0.0,
        )

    def test_setup_applies_identity_shift_and_terminal_zero(self):
        scheduler = FlowMatchSDEDiscreteScheduler()
        sigmas = setup_boogu_sigmas(scheduler, 4, device="cpu")

        # set_shift(1.0) must leave the sigmas unchanged (identity time shift);
        # the scheduler appends the terminal sigma 0.
        torch.testing.assert_close(
            scheduler.sigmas,
            torch.tensor(sigmas + [0.0], dtype=scheduler.sigmas.dtype),
        )
        # verl convention: timesteps are sigma * num_train_timesteps (1000).
        torch.testing.assert_close(
            scheduler.timesteps,
            torch.tensor(sigmas, dtype=scheduler.timesteps.dtype) * 1000.0,
        )
        assert scheduler.begin_index == 0


class TestTimestepConversion:
    def test_signal_fraction_and_conditioning_clamp(self):
        scheduler = FlowMatchSDEDiscreteScheduler()
        setup_boogu_sigmas(scheduler, 4, device="cpu")

        model_t = boogu_timestep_from_scheduler_timestep(scheduler.timesteps)
        expected = _upstream_dmd_signal_fractions(4)

        # The transformer must see the exact upstream conditioning ladder,
        # including 0.001 at the first step where the scheduler sigma was
        # pinned to 1.0 (1 - 1.0 = 0 would be out of the trained range).
        torch.testing.assert_close(model_t, expected, atol=1e-6, rtol=0.0)

    def test_clamp_floor_is_conditioning_sigma(self):
        t = boogu_timestep_from_scheduler_timestep(torch.tensor([1000.0, 999.5]))
        expected = torch.full((2,), BOOGU_DMD_CONDITIONING_SIGMA)
        torch.testing.assert_close(t, expected)


class TestVelocityConversion:
    def test_negates_model_prediction(self):
        pred = torch.randn(2, 16, 8, 8)
        torch.testing.assert_close(boogu_pred_to_velocity(pred), -pred)

    def test_recovers_x0_under_sde_convention(self):
        # If the model predicts exactly x0 - eps, then x_sigma - sigma * v
        # with v = -(x0 - eps) must recover x0 for the linear flow
        # x_sigma = sigma * eps + (1 - sigma) * x0.
        x0 = torch.randn(1, 16, 4, 4)
        eps = torch.randn(1, 16, 4, 4)
        sigma = 0.4995
        x_sigma = sigma * eps + (1 - sigma) * x0
        v = boogu_pred_to_velocity(x0 - eps)
        torch.testing.assert_close(x_sigma - sigma * v, x0, atol=1e-6, rtol=0.0)


class TestGenerationPromptStripping:
    def _make_pipe(self, suffix_ids: list[int]) -> BooguImageTurboPipelineWithLogProb:
        pipe = object.__new__(BooguImageTurboPipelineWithLogProb)
        pipe._generation_prompt_ids = suffix_ids
        return pipe

    def test_strips_exact_suffix(self):
        suffix = [151644, 77091, 198]
        pipe = self._make_pipe(suffix)
        ids = torch.tensor([[1, 2, 3, *suffix]])
        mask = torch.ones_like(ids)

        out_ids, out_mask = pipe._strip_generation_prompt(ids, mask)

        assert out_ids.tolist() == [[1, 2, 3]]
        assert out_mask.shape == out_ids.shape

    def test_keeps_prompt_without_suffix(self):
        pipe = self._make_pipe([151644, 77091, 198])
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(ids)

        out_ids, out_mask = pipe._strip_generation_prompt(ids, mask)

        assert out_ids.tolist() == [[1, 2, 3, 4]]
        assert out_mask.shape == out_ids.shape

    def test_short_prompt_not_stripped(self):
        suffix = [151644, 77091, 198]
        pipe = self._make_pipe(suffix)
        ids = torch.tensor([[77091, 198]])
        mask = torch.ones_like(ids)

        out_ids, _ = pipe._strip_generation_prompt(ids, mask)

        assert out_ids.tolist() == [[77091, 198]]

    def test_suffix_constant_matches_qwen_chat_template_text(self):
        assert BOOGU_GENERATION_PROMPT_SUFFIX == "<|im_start|>assistant\n"
