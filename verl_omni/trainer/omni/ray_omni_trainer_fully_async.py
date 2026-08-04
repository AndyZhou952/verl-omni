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
"""Omni fully-async trainer — a PPOTrainerSeparateAsync subclass.

Registered via ``@register_trainer("omni_fully_async")``. Prompt submission is
decoupled from the trainer's step cadence: a staleness gate keeps up to
``staleness_threshold + 1`` prompt batches in flight on the standalone rollout
pool, topped up at every step boundary, while ``step()`` consumes whichever
GRPO groups finished first (oldest-first via the replay buffer). Weight sync to
the standalone replicas runs every ``parameter_sync_step`` steps; aborted
generations recover per ``rollout_recovery`` (token-level continuation or
whole-sample retry).
"""

import logging
import os

import ray
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer

from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineManager
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.engine_workers import ActorRolloutRefWorker
from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class StalenessGate:
    """Producer-side back-pressure: cap the number of prompt batches in flight.

    With one optimizer step per consumed batch (``train_batch_size ==
    ppo_mini_batch_size``), a batch submitted while ``consumed_steps`` steps are
    complete is consumed at most ``staleness_threshold`` weight versions later
    when at most ``staleness_threshold + 1`` batches are in flight.
    """

    def __init__(self, staleness_threshold: int):
        if isinstance(staleness_threshold, bool) or not isinstance(staleness_threshold, int) or staleness_threshold < 0:
            raise ValueError(f"Invalid staleness_threshold: {staleness_threshold}. Must be an integer >= 0")
        self.staleness_threshold = staleness_threshold

    def allowed_batches(self, consumed_steps: int, total_training_steps: int) -> int:
        """Total batches that may have been submitted after ``consumed_steps`` completed steps."""
        return min(consumed_steps + self.staleness_threshold + 1, total_training_steps)


@register_trainer("omni_fully_async")
class OmniPPOTrainerFullyAsync(PPOTrainerSeparateAsync):
    """``PPOTrainerSeparateAsync`` subclass that wires tokenizer/processor from
    ``OmniModelConfig``, feeds prompts through a staleness gate instead of
    once per step, and reads its knobs from the ``omni_fully_async`` config key.
    """

    VALID_ROLLOUT_RECOVERY = {"continue", "whole_sample_retry"}

    def __init__(self, config):
        super().__init__(config)

        fa_config = config.trainer.v1.omni_fully_async
        self.staleness_gate = StalenessGate(fa_config.staleness_threshold)
        self._submitted_batches = 0
        self._consumed_at_start = None  # set at on_train_begin; nonzero on checkpoint resume

        if not isinstance(self.parameter_sync_step, int) or self.parameter_sync_step < 1:
            raise ValueError(f"Invalid parameter_sync_step: {self.parameter_sync_step}. Must be an integer >= 1")

        self.rollout_recovery = fa_config.rollout_recovery
        if self.rollout_recovery not in self.VALID_ROLLOUT_RECOVERY:
            raise ValueError(
                f"Invalid rollout_recovery: {self.rollout_recovery}. "
                f"Must be one of {sorted(self.VALID_ROLLOUT_RECOVERY)}"
            )

        # The gate bounds staleness at (k + 1) / parameter_sync_step buffer units. The
        # buffer's `drop` strategy discards on strict >, so equality is fine; `wait`
        # stalls sampling on >= (replay_buffer._has_enough_samples), which at equality
        # degrades every step to lock-step — require strict < there.
        max_off_policy_threshold = config.trainer.v1.sampler.max_off_policy_threshold
        max_off_policy_strategy = config.trainer.v1.sampler.max_off_policy_strategy
        worst_staleness = (self.staleness_gate.staleness_threshold + 1) / self.parameter_sync_step
        over_budget = (
            worst_staleness >= max_off_policy_threshold
            if max_off_policy_strategy == "wait"
            else worst_staleness > max_off_policy_threshold
        )
        if over_budget:
            raise ValueError(
                f"staleness_threshold={self.staleness_gate.staleness_threshold} with "
                f"parameter_sync_step={self.parameter_sync_step} reaches staleness {worst_staleness}, "
                f"exceeding trainer.v1.sampler.max_off_policy_threshold={max_off_policy_threshold} "
                f"under strategy '{max_off_policy_strategy}'"
            )

        # Hybrid replicas are woken for the first sampling window and validations; with
        # dummy load_format their excluded (never-synced) towers would stay random.
        exclude_regex = config.actor_rollout_ref.model.get("weight_sync_exclude_regex", None)
        if exclude_regex and config.actor_rollout_ref.rollout.load_format == "dummy":
            raise ValueError(
                "weight_sync_exclude_regex requires a real-weight rollout load_format "
                "(e.g. safetensors); load_format=dummy would leave excluded towers uninitialized"
            )

        # Parent trusts rollout log-probs (bypass). Continuations may span weight
        # versions, so default to recomputing old_log_probs under the current policy.
        self.config.algorithm.rollout_correction.bypass_mode = fa_config.use_rollout_log_probs

    def _init_tokenizer(self):
        # Skip super(): OmniModelConfig loads tokenizer/processor via the registered adapter.
        model_config: OmniModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, OmniModelConfig)
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _init_resource_pool_mgr(self):
        # verl_omni's worker adds the adapter-only LoRA send path (base_sync_done=True)
        # and get_lora_peft_config; the upstream worker would broadcast stale base
        # weights forever under lora.merge=False + non-naive backends.
        super()._init_resource_pool_mgr()
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        self.role_worker_mapping[actor_role] = ray.remote(ActorRolloutRefWorker)

    def _setup(self):
        super()._setup()
        # Replace the parent's manager with the LoRA-aware one: it pushes the actor's
        # peft_config to standalone replicas so adapter deltas apply via add_lora.
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = OmniCheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

    def get_llm_client(self):
        if self.rollout_recovery == "whole_sample_retry":
            # Generic abort-then-resubmit-whole-sample client (despite the diffusion name).
            return self.standalone_server_manager.get_client(client_cls=DiffusionWholeSampleRetryLLMServerClient)
        return super().get_llm_client()

    def _add_batch_to_generate(self):
        # No-op in the consumer path: feeding is owned by _top_up_generation.
        return

    def _top_up_generation(self):
        """Submit prompt batches until the staleness gate is saturated.

        Counts are relative to ``_consumed_at_start`` so a checkpoint resume
        (``global_steps`` > 1 at ``on_train_begin``) seeds only ``k + 1`` batches
        instead of one per already-consumed step.
        """
        consumed = self.global_steps - 1 - self._consumed_at_start
        remaining_total = self.total_training_steps - self._consumed_at_start
        target = self.staleness_gate.allowed_batches(consumed, remaining_total)
        while self._submitted_batches < target:
            super()._add_batch_to_generate()
            self._submitted_batches += 1

    def on_train_begin(self):
        # Implicit warmup: the gate admits staleness_threshold + 1 batches before the
        # first (or first resumed) step.
        self._consumed_at_start = self.global_steps - 1
        self._top_up_generation()
        logger.info(f"Seeded {self._submitted_batches} prompt batches (staleness gate)")

    def on_step_begin(self):
        self._top_up_generation()

    def on_step_end(self):
        # Parent reads the separate_async config key; same cadence, our key.
        if self.global_steps % self.parameter_sync_step == 0:
            with marked_timer("update_weights", self.timing_raw, color="red"):
                self.standalone_checkpoint_manager.update_weights(self.global_steps)
