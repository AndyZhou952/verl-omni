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
"""CPU tests for ``OmniPPOTrainerFullyAsync``: registration, config wiring via
hydra compose, staleness-gate pacing, and weight-sync cadence. No GPU, no Ray.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from verl_omni.trainer.omni.ray_omni_trainer_fully_async import (
    OmniPPOTrainerFullyAsync,
    StalenessGate,
)

_CONFIG_DIR = str((Path(__file__).parents[3] / "verl_omni" / "trainer" / "config").resolve())

_BASE_OVERRIDES = [
    "trainer.v1.trainer_mode=omni_fully_async",
    "data.train_batch_size=4",
    "actor_rollout_ref.actor.ppo_mini_batch_size=4",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.n_gpus_per_node=2",
    "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    "actor_rollout_ref.model.path=/dummy/model",
]


def _compose_config(extra_overrides=()):
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name="omni_trainer", overrides=[*_BASE_OVERRIDES, *extra_overrides])


class TestRegistration:
    def test_registered_under_omni_fully_async(self):
        assert get_trainer_cls("omni_fully_async") is OmniPPOTrainerFullyAsync

    def test_subclass_of_ppo_trainer_separate_async(self):
        assert issubclass(OmniPPOTrainerFullyAsync, PPOTrainerSeparateAsync)


class TestStalenessGate:
    def test_allowed_batches_grows_one_per_step(self):
        gate = StalenessGate(staleness_threshold=1)
        assert gate.allowed_batches(consumed_steps=0, total_training_steps=100) == 2
        assert gate.allowed_batches(consumed_steps=1, total_training_steps=100) == 3
        assert gate.allowed_batches(consumed_steps=50, total_training_steps=100) == 52

    def test_allowed_batches_caps_at_total_steps(self):
        gate = StalenessGate(staleness_threshold=3)
        assert gate.allowed_batches(consumed_steps=98, total_training_steps=100) == 100
        assert gate.allowed_batches(consumed_steps=100, total_training_steps=100) == 100

    def test_zero_threshold_is_synchronous_pacing(self):
        gate = StalenessGate(staleness_threshold=0)
        assert gate.allowed_batches(consumed_steps=0, total_training_steps=10) == 1
        assert gate.allowed_batches(consumed_steps=4, total_training_steps=10) == 5

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", None, True])
    def test_invalid_threshold_raises(self, bad):
        with pytest.raises(ValueError, match="staleness_threshold"):
            StalenessGate(bad)


class TestInitConfigWiring:
    def test_init_with_defaults(self):
        trainer = OmniPPOTrainerFullyAsync(_compose_config())
        assert trainer.staleness_gate.staleness_threshold == 1
        assert trainer.parameter_sync_step == 1
        assert trainer.rollout_recovery == "continue"
        # Fully-async recomputes old_log_probs by default (parent forces bypass).
        assert trainer.config.algorithm.rollout_correction.bypass_mode is False

    def test_use_rollout_log_probs_keeps_bypass(self):
        trainer = OmniPPOTrainerFullyAsync(_compose_config(["trainer.v1.omni_fully_async.use_rollout_log_probs=true"]))
        assert trainer.config.algorithm.rollout_correction.bypass_mode is True

    def test_invalid_rollout_recovery_raises(self):
        with pytest.raises(ValueError, match="rollout_recovery"):
            OmniPPOTrainerFullyAsync(_compose_config(["trainer.v1.omni_fully_async.rollout_recovery=bogus"]))

    def test_staleness_exceeding_buffer_threshold_raises(self):
        with pytest.raises(ValueError, match="max_off_policy_threshold"):
            OmniPPOTrainerFullyAsync(
                _compose_config(
                    [
                        "trainer.v1.omni_fully_async.staleness_threshold=20",
                        "trainer.v1.sampler.max_off_policy_threshold=8",
                    ]
                )
            )

    def test_train_batch_size_must_match_mini_batch_size(self):
        with pytest.raises(AssertionError, match="train_batch_size"):
            OmniPPOTrainerFullyAsync(_compose_config(["data.train_batch_size=8"]))

    def test_parameter_sync_step_relaxes_staleness_bound(self):
        trainer = OmniPPOTrainerFullyAsync(
            _compose_config(
                [
                    "trainer.v1.omni_fully_async.staleness_threshold=15",
                    "trainer.v1.omni_fully_async.parameter_sync_step=2",
                    "trainer.v1.sampler.max_off_policy_threshold=8",
                ]
            )
        )
        assert trainer.parameter_sync_step == 2

    def test_zero_parameter_sync_step_raises(self):
        with pytest.raises(ValueError, match="parameter_sync_step"):
            OmniPPOTrainerFullyAsync(_compose_config(["trainer.v1.omni_fully_async.parameter_sync_step=0"]))

    def test_wait_strategy_requires_strict_staleness_margin(self):
        # (k + 1) == threshold: fine for `drop` (strict > discards), lock-step for `wait`.
        equality = [
            "trainer.v1.omni_fully_async.staleness_threshold=7",
            "trainer.v1.sampler.max_off_policy_threshold=8",
        ]
        trainer = OmniPPOTrainerFullyAsync(_compose_config(equality))
        assert trainer.staleness_gate.staleness_threshold == 7
        with pytest.raises(ValueError, match="wait"):
            OmniPPOTrainerFullyAsync(_compose_config([*equality, "trainer.v1.sampler.max_off_policy_strategy=wait"]))

    def test_exclude_regex_with_dummy_load_format_raises(self):
        with pytest.raises(ValueError, match="load_format"):
            OmniPPOTrainerFullyAsync(
                _compose_config(
                    [
                        "actor_rollout_ref.model.weight_sync_exclude_regex='.*visual.*'",
                        "actor_rollout_ref.rollout.load_format=dummy",
                    ]
                )
            )

    def test_exclude_regex_with_real_load_format_inits(self):
        trainer = OmniPPOTrainerFullyAsync(
            _compose_config(
                [
                    "actor_rollout_ref.model.weight_sync_exclude_regex='.*visual.*'",
                    "actor_rollout_ref.rollout.load_format=safetensors",
                ]
            )
        )
        assert trainer.rollout_recovery == "continue"


class TestInitTokenizer:
    def test_init_tokenizer_wires_omni_model_config(self):
        trainer = OmniPPOTrainerFullyAsync.__new__(OmniPPOTrainerFullyAsync)
        trainer.config = OmegaConf.create({"actor_rollout_ref": {"model": {"path": "/dummy"}}})

        fake_cfg = SimpleNamespace(tokenizer="fake_tok", processor="fake_proc")
        with patch(
            "verl_omni.trainer.omni.ray_omni_trainer_fully_async.omega_conf_to_dataclass",
            return_value=fake_cfg,
        ):
            trainer._init_tokenizer()

        assert trainer.tokenizer == "fake_tok"
        assert trainer.processor == "fake_proc"


def _pacing_trainer(staleness_threshold: int, total_training_steps: int, global_steps: int = 1):
    """Build a bare trainer wired for feed-pacing tests (no config compose)."""
    trainer = OmniPPOTrainerFullyAsync.__new__(OmniPPOTrainerFullyAsync)
    trainer.staleness_gate = StalenessGate(staleness_threshold)
    trainer._submitted_batches = 0
    trainer._consumed_at_start = None
    trainer.total_training_steps = total_training_steps
    trainer.global_steps = global_steps
    return trainer


class TestFeedPacing:
    def test_consumer_feed_hook_is_noop(self):
        trainer = _pacing_trainer(1, 10)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer._add_batch_to_generate()
        mock_feed.assert_not_called()

    def test_train_begin_seeds_k_plus_one_batches(self):
        trainer = _pacing_trainer(2, 10)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
        assert mock_feed.call_count == 3
        assert trainer._submitted_batches == 3

    def test_step_begin_tops_up_one_batch_per_step(self):
        trainer = _pacing_trainer(1, 10)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
            assert mock_feed.call_count == 2
            for step in range(1, 5):
                trainer.global_steps = step
                trainer.on_step_begin()
            # steps 1..4 completed 0..3 consumer steps: 2 seeded + 3 top-ups
            assert mock_feed.call_count == 5
            assert trainer._submitted_batches == 5

    def test_repeated_top_up_within_step_feeds_nothing(self):
        trainer = _pacing_trainer(1, 10)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
            trainer.on_step_begin()
            trainer.on_step_begin()
        assert mock_feed.call_count == 2

    def test_never_feeds_past_total_training_steps(self):
        trainer = _pacing_trainer(3, 4)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
            for step in range(1, 5):
                trainer.global_steps = step
                trainer.on_step_begin()
        assert mock_feed.call_count == 4
        assert trainer._submitted_batches == 4

    def test_checkpoint_resume_seeds_only_k_plus_one(self):
        # fit() resumes with global_steps = 51 at on_train_begin; only fresh batches count.
        trainer = _pacing_trainer(1, 100, global_steps=51)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
            assert mock_feed.call_count == 2
            trainer.global_steps = 52
            trainer.on_step_begin()
        assert mock_feed.call_count == 3

    def test_checkpoint_resume_respects_remaining_total(self):
        # Resume at step 99 of 100: only 2 batches remain to be consumed.
        trainer = _pacing_trainer(3, 100, global_steps=99)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
        assert mock_feed.call_count == 2


class TestLoraAwareWiring:
    def test_actor_worker_is_verl_omni_class(self):
        from verl.trainer.ppo.utils import Role

        trainer = OmniPPOTrainerFullyAsync(_compose_config())
        trainer._init_resource_pool_mgr()
        role = Role.ActorRolloutRef if Role.ActorRolloutRef in trainer.role_worker_mapping else Role.ActorRollout
        modified = trainer.role_worker_mapping[role].__ray_metadata__.modified_class
        assert modified.__module__.startswith("verl_omni"), modified.__module__

    def test_setup_installs_lora_aware_checkpoint_manager(self):
        from verl.checkpoint_engine import CheckpointEngineRegistry

        from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager

        trainer = OmniPPOTrainerFullyAsync(_compose_config())
        with (
            patch.object(PPOTrainerSeparateAsync, "_setup"),
            # The nccl backend registers via GPU-only import side effects.
            patch.object(CheckpointEngineRegistry, "get", return_value=MagicMock()),
        ):
            trainer.actor_rollout_wg = MagicMock()
            trainer.standalone_server_manager = MagicMock()
            trainer.standalone_server_manager.get_replicas.return_value = []
            trainer._setup()
        assert isinstance(trainer.standalone_checkpoint_manager, OmniCheckpointEngineManager)


class TestWeightSyncCadence:
    def _sync_trainer(self, parameter_sync_step: int):
        trainer = OmniPPOTrainerFullyAsync.__new__(OmniPPOTrainerFullyAsync)
        trainer.parameter_sync_step = parameter_sync_step
        trainer.timing_raw = {}
        trainer.standalone_checkpoint_manager = MagicMock()
        return trainer

    def test_syncs_every_step_by_default(self):
        trainer = self._sync_trainer(1)
        for step in (1, 2, 3):
            trainer.global_steps = step
            trainer.on_step_end()
        assert trainer.standalone_checkpoint_manager.update_weights.call_count == 3

    def test_sync_throttled_by_parameter_sync_step(self):
        trainer = self._sync_trainer(2)
        for step in (1, 2, 3, 4):
            trainer.global_steps = step
            trainer.on_step_end()
        assert trainer.standalone_checkpoint_manager.update_weights.call_count == 2
        trainer.standalone_checkpoint_manager.update_weights.assert_called_with(4)
