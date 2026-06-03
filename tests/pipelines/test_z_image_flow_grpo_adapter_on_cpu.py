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
"""CPU tests for Z-Image FlowGRPO adapter helpers."""

import torch

from verl_omni.pipelines.z_image_flow_grpo.common import (
    list_embeds_to_padded,
    padded_embeds_to_list,
    stack_z_image_model_output,
)


def test_z_image_prompt_embed_padding_roundtrip():
    embeds = [
        torch.arange(6, dtype=torch.float32).view(3, 2),
        torch.arange(4, dtype=torch.float32).view(2, 2),
    ]

    padded, mask = list_embeds_to_padded(embeds)
    restored = padded_embeds_to_list(padded, mask)

    assert padded.shape == (2, 3, 2)
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert torch.equal(restored[0], embeds[0])
    assert torch.equal(restored[1], embeds[1])


def test_z_image_model_output_stack_squeezes_frame_dimension():
    output = ([torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)], {})

    stacked = stack_z_image_model_output(output)

    assert stacked.shape == (2, 2, 4, 4)
    assert torch.equal(stacked[0], torch.ones(2, 4, 4))
    assert torch.equal(stacked[1], torch.zeros(2, 4, 4))
