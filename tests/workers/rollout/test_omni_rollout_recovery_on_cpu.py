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
"""CPU tests for fully-async rollout recovery.

Pins the two recovery paths ``OmniPPOTrainerFullyAsync.get_llm_client`` selects:
token-level continuation (upstream ``FullyAsyncLLMServerClient``) and whole-sample
retry (``WholeSampleRetryLLMServerClient``) — and the multimodal invariants both
must satisfy: media and processor kwargs are re-sent on every resubmission,
token/log-prob streams are merged, and ``min/max_global_steps`` record the
weight-version span.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient

from verl_omni.trainer.omni.ray_omni_trainer_fully_async import OmniPPOTrainerFullyAsync
from verl_omni.workers.rollout.retry_llm_server import WholeSampleRetryLLMServerClient


def _client(cls, **attrs):
    client = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(client, name, value)
    return client


@pytest.mark.parametrize(
    "rollout_recovery, expected_cls",
    [("continue", FullyAsyncLLMServerClient), ("whole_sample_retry", WholeSampleRetryLLMServerClient)],
)
def test_recovery_mode_selects_client(rollout_recovery, expected_cls):
    trainer = OmniPPOTrainerFullyAsync.__new__(OmniPPOTrainerFullyAsync)
    trainer.rollout_recovery = rollout_recovery
    trainer.standalone_server_manager = MagicMock()
    trainer.get_llm_client()
    trainer.standalone_server_manager.get_client.assert_called_once_with(client_cls=expected_cls)


async def test_continuation_resends_media_merges_tokens_and_shrinks_budget():
    """Pin upstream ``FullyAsyncLLMServerClient`` behavior the omni path relies on."""
    outputs = [
        SimpleNamespace(
            token_ids=[7, 8],
            log_probs=[-0.1, -0.2],
            routed_experts=None,
            num_preempted=0,
            stop_reason="abort",
            extra_fields={"global_steps": 3},
        ),
        SimpleNamespace(
            token_ids=[9],
            log_probs=[-0.3],
            routed_experts=None,
            num_preempted=0,
            stop_reason="completed",
            extra_fields={"global_steps": 5},
        ),
    ]
    client = _client(FullyAsyncLLMServerClient)
    media = {"image_data": ["img"], "video_data": None, "audio_data": ["aud"]}
    # sampling_params is mutated in place, so capture the budget at call time.
    seen_budgets = []
    output_iter = iter(outputs)

    async def _record(*args, **kwargs):
        seen_budgets.append(kwargs["sampling_params"]["max_tokens"])
        return next(output_iter)

    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=_record)) as mock_gen:
        final = await client.generate(
            "req-1",
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 8},
            mm_processor_kwargs={"fps": 2},
            **media,
        )

    assert mock_gen.call_count == 2
    first, second = mock_gen.call_args_list
    # Continuation resubmits prompt + generated-so-far tokens with the same media.
    assert first.kwargs["prompt_ids"] == [1, 2, 3]
    assert second.kwargs["prompt_ids"] == [1, 2, 3, 7, 8]
    for call in (first, second):
        assert call.kwargs["image_data"] == ["img"]
        assert call.kwargs["audio_data"] == ["aud"]
        assert call.kwargs["mm_processor_kwargs"] == {"fps": 2}
    # The resumed request's budget shrinks by the tokens already generated.
    assert seen_budgets == [8, 6]

    assert final.token_ids == [7, 8, 9]
    assert final.log_probs == [-0.1, -0.2, -0.3]
    assert final.stop_reason == "completed"
    assert final.extra_fields["min_global_steps"] == 3
    assert final.extra_fields["max_global_steps"] == 5


async def test_whole_sample_retry_discards_partials_stamps_versions_and_gives_up():
    outputs = [
        SimpleNamespace(stop_reason="aborted", extra_fields={"global_steps": 3}, token_ids=[7, 8]),
        SimpleNamespace(stop_reason="completed", extra_fields={"global_steps": 5}, token_ids=[4, 4, 4]),
    ]
    client = _client(WholeSampleRetryLLMServerClient, max_retries=5, retry_wait_s=0.0)
    media = {"image_data": ["img"], "video_data": None, "audio_data": None}

    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=outputs)) as mock_gen:
        final = await client.generate(
            "req-1",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 8},
            mm_processor_kwargs=None,
            **media,
        )

    # Whole-sample retry: identical fresh request each attempt, media re-sent.
    for call in mock_gen.call_args_list:
        assert call.kwargs["prompt_ids"] == [1, 2]
        assert call.kwargs["image_data"] == ["img"]
    assert final.token_ids == [4, 4, 4]
    assert final.extra_fields["min_global_steps"] == 3
    assert final.extra_fields["max_global_steps"] == 5

    # Persistent abort: gives up after max_retries and returns the aborted output.
    aborted = SimpleNamespace(stop_reason="aborted", extra_fields={"global_steps": 1}, token_ids=[])
    client = _client(WholeSampleRetryLLMServerClient, max_retries=3, retry_wait_s=0.0)
    with patch.object(LLMServerClient, "generate", new=AsyncMock(return_value=aborted)) as mock_gen:
        final = await client.generate("req-1", prompt_ids=[1], sampling_params={})
    assert mock_gen.call_count == 3
    assert final.stop_reason == "aborted"
