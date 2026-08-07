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


def test_registered_and_subclasses_separate_async():
    assert get_trainer_cls("omni_fully_async") is OmniPPOTrainerFullyAsync
    assert issubclass(OmniPPOTrainerFullyAsync, PPOTrainerSeparateAsync)


class TestStalenessGate:
    def test_allowed_batches_pacing(self):
        gate = StalenessGate(staleness_threshold=1)
        assert gate.allowed_batches(consumed_steps=0, total_training_steps=100) == 2
        assert gate.allowed_batches(consumed_steps=1, total_training_steps=100) == 3
        assert gate.allowed_batches(consumed_steps=50, total_training_steps=100) == 52
        # Caps at total training steps.
        capped = StalenessGate(staleness_threshold=3)
        assert capped.allowed_batches(consumed_steps=98, total_training_steps=100) == 100
        assert capped.allowed_batches(consumed_steps=100, total_training_steps=100) == 100
        # Zero threshold degrades to synchronous pacing.
        sync = StalenessGate(staleness_threshold=0)
        assert sync.allowed_batches(consumed_steps=0, total_training_steps=10) == 1
        assert sync.allowed_batches(consumed_steps=4, total_training_steps=10) == 5

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

    @pytest.mark.parametrize(
        "overrides, exc_type, match",
        [
            (["trainer.v1.omni_fully_async.rollout_recovery=bogus"], ValueError, "rollout_recovery"),
            (["trainer.v1.omni_fully_async.parameter_sync_step=0"], ValueError, "parameter_sync_step"),
            (
                [
                    "trainer.v1.omni_fully_async.staleness_threshold=20",
                    "trainer.v1.sampler.max_off_policy_threshold=8",
                ],
                ValueError,
                "max_off_policy_threshold",
            ),
            # (k + 1) == threshold: fine for `drop` (strict > discards), lock-step for `wait`.
            (
                [
                    "trainer.v1.omni_fully_async.staleness_threshold=7",
                    "trainer.v1.sampler.max_off_policy_threshold=8",
                    "trainer.v1.sampler.max_off_policy_strategy=wait",
                ],
                ValueError,
                "wait",
            ),
            (
                [
                    "+actor_rollout_ref.model.weight_sync_exclude_frozen=true",
                    "actor_rollout_ref.rollout.load_format=dummy",
                ],
                ValueError,
                "load_format",
            ),
            (["data.train_batch_size=8"], AssertionError, "train_batch_size"),
        ],
    )
    def test_invalid_config_raises(self, overrides, exc_type, match):
        with pytest.raises(exc_type, match=match):
            OmniPPOTrainerFullyAsync(_compose_config(overrides))

    def test_valid_overrides_init(self):
        trainer = OmniPPOTrainerFullyAsync(
            _compose_config(
                [
                    "trainer.v1.omni_fully_async.use_rollout_log_probs=true",
                    "trainer.v1.omni_fully_async.staleness_threshold=15",
                    "trainer.v1.omni_fully_async.parameter_sync_step=2",
                    "trainer.v1.sampler.max_off_policy_threshold=8",
                    "+actor_rollout_ref.model.weight_sync_exclude_frozen=true",
                    "actor_rollout_ref.rollout.load_format=safetensors",
                ]
            )
        )
        # parameter_sync_step=2 relaxes the staleness bound: (15 + 1) / 2 <= 8.
        assert trainer.parameter_sync_step == 2
        assert trainer.staleness_gate.staleness_threshold == 15
        assert trainer.config.algorithm.rollout_correction.bypass_mode is True


def test_init_tokenizer_wires_omni_model_config():
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
    def test_fresh_start_pacing(self):
        trainer = _pacing_trainer(1, 10)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
            assert mock_feed.call_count == 2  # seeds k + 1
            trainer._add_batch_to_generate()  # consumer hook must not feed
            trainer.on_step_begin()  # repeated top-up within a step is a no-op
            assert mock_feed.call_count == 2
            for step in range(1, 5):
                trainer.global_steps = step
                trainer.on_step_begin()
            # steps 1..4 completed 0..3 consumer steps: 2 seeded + 3 top-ups
            assert mock_feed.call_count == 5
            assert trainer._submitted_batches == 5

        # Never feeds past total_training_steps.
        capped = _pacing_trainer(3, 4)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            capped.on_train_begin()
            for step in range(1, 5):
                capped.global_steps = step
                capped.on_step_begin()
        assert mock_feed.call_count == 4
        assert capped._submitted_batches == 4

    def test_checkpoint_resume_pacing(self):
        # fit() resumes with global_steps = 51 at on_train_begin; only fresh batches count.
        trainer = _pacing_trainer(1, 100, global_steps=51)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            trainer.on_train_begin()
            assert mock_feed.call_count == 2
            trainer.global_steps = 52
            trainer.on_step_begin()
            assert mock_feed.call_count == 3

        # Resume at step 99 of 100: only 2 batches remain to be consumed.
        tail = _pacing_trainer(3, 100, global_steps=99)
        with patch.object(PPOTrainerSeparateAsync, "_add_batch_to_generate") as mock_feed:
            tail.on_train_begin()
        assert mock_feed.call_count == 2


class TestLoraAwareWiring:
    def test_actor_worker_is_verl_omni_class(self):
        from verl.trainer.ppo.utils import Role

        trainer = OmniPPOTrainerFullyAsync(_compose_config())
        trainer._init_resource_pool_mgr()
        role = Role.ActorRolloutRef if Role.ActorRolloutRef in trainer.role_worker_mapping else Role.ActorRollout
        modified = trainer.role_worker_mapping[role].__ray_metadata__.modified_class
        assert modified.__module__.startswith("verl_omni"), modified.__module__

    def test_swapped_worker_exposes_upstream_v1_log_prob_methods(self):
        # The upstream v1 trainer calls these on the swapped-in worker group when
        # use_rollout_log_probs=false; an unregistered name only fails at Ray
        # dispatch time on GPU.
        from verl.single_controller.base.decorator import MAGIC_ATTR

        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        for name in ("compute_log_prob", "compute_ref_log_prob"):
            assert hasattr(getattr(ActorRolloutRefWorker, name), MAGIC_ATTR), name

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


def test_weight_sync_cadence_follows_parameter_sync_step():
    for parameter_sync_step, steps, expected_syncs in ((1, 3, 3), (2, 4, 2)):
        trainer = OmniPPOTrainerFullyAsync.__new__(OmniPPOTrainerFullyAsync)
        trainer.parameter_sync_step = parameter_sync_step
        trainer.timing_raw = {}
        trainer.standalone_checkpoint_manager = MagicMock()
        for step in range(1, steps + 1):
            trainer.global_steps = step
            trainer.on_step_end()
        assert trainer.standalone_checkpoint_manager.update_weights.call_count == expected_syncs
    trainer.standalone_checkpoint_manager.update_weights.assert_called_with(4)
