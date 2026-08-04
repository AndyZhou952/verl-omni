(fully_async_omni)=
# Fully-Async RL Training for Qwen3-Omni

Last updated: 08/04/2026

`trainer.v1.trainer_mode=omni_fully_async` decouples rollout generation from
the trainer's step cadence for omni AR models (Qwen3-Omni thinker). Standalone
rollout replicas generate continuously; a staleness gate keeps
`staleness_threshold + 1` prompt batches in flight; the trainer consumes
whichever GRPO groups finish first (oldest-first) and pushes weights every
`parameter_sync_step` steps over a non-naive checkpoint engine (nccl/nixl/...).
Generations aborted by a weight sync are resubmitted under the new weights
(see `rollout_recovery` and the abort caveats below), with
`min/max_global_steps` recording the weight-version span of every sample.

## When to use

- Long-tail completions (large `rollout.n`, high length variance) leave the
  trainer or rollout GPUs idle in `omni_sync`.
- Multimodal prefill (image/video/audio encoders) makes generation the
  dominant phase of the step.

Off-policyness is bounded: a sample is trained on at most `staleness_threshold`
weight versions after it was generated. Keep it small (1–2); the replay buffer's
`trainer.v1.sampler.max_off_policy_threshold` remains the hard backstop
(`drop` or `wait`).

## GPU layout

`trainer.n_gpus_per_node × trainer.nnodes` GPUs run the FSDP actor;
`actor_rollout_ref.rollout.n_gpus_per_node × actor_rollout_ref.rollout.nnodes`
additional GPUs run standalone rollout replicas
(`n_gpus_per_node / tensor_model_parallel_size` replicas per node). Single-node
replicas only for AR omni today (`run_headless` is not implemented upstream).

## Run

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_fully_async_v1.sh
```

The example splits 4 GPUs into 2 trainer + 2 rollout (one TP=2 replica) and
uses GSPO + GRPO advantages with LoRA. Key overrides:

| knob | default | meaning |
|---|---|---|
| `trainer.v1.omni_fully_async.staleness_threshold` | 1 | in-flight budget k; warmup is implicit (k + 1 batches before step 1) |
| `trainer.v1.omni_fully_async.parameter_sync_step` | 1 | push weights to standalone replicas every N steps |
| `trainer.v1.omni_fully_async.rollout_recovery` | `continue` | `continue` resumes aborted generations under new weights; `whole_sample_retry` regenerates them |
| `trainer.v1.omni_fully_async.use_rollout_log_probs` | false | keep the separate-async bypass instead of recomputing old_log_probs |
| `actor_rollout_ref.model.weight_sync_exclude_regex` | null | frozen parameters to skip during full-param weight sync, e.g. `".*visual.*\|.*audio_tower.*"` (fails fast if a match is trainable; full-param + fsdp2 only; rejected with `load_format=dummy`) |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | — | must be non-naive (`nccl`, `nixl`, `mooncake`) |

Constraints enforced at startup: `data.train_batch_size ==
actor_rollout_ref.actor.ppo_mini_batch_size` (one optimizer step per consumed
batch), rollout GPUs > 0, non-naive checkpoint backend, and
`(staleness_threshold + 1) / parameter_sync_step <= max_off_policy_threshold`.

LoRA recipes should set `actor_rollout_ref.model.lora.merge=False` so weight
sync ships only adapter tensors (applied on the replicas via the LoRA-aware
checkpoint engine manager).

## Monitor

Watch `training/off_policy/*` metrics (staleness mean/max, dropped samples).
Rising dropped-sample counts mean the rollout pool cannot keep up with the
staleness window — add rollout GPUs, raise `staleness_threshold`, or raise
`parameter_sync_step`.

## Test

CPU (no GPU required; covers registration, config wiring, staleness-gate
pacing, weight-sync cadence, recovery semantics incl. multimodal payload
re-submission, and the weight-sync exclusion filter):

```bash
pytest -s --asyncio-mode=auto \
    tests/trainer/omni/test_ray_omni_trainer_fully_async_on_cpu.py \
    tests/workers/rollout/test_omni_rollout_recovery_on_cpu.py \
    tests/workers/test_omni_fsdp_engine_on_cpu.py
```

GPU smoke (2 GPUs, tiny-random Qwen3-Omni, 3 fully-async steps):

```bash
bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_fully_async_smoke.sh
```

For a parity check against the synchronous baseline, run the same recipe with
`trainer.v1.trainer_mode=omni_sync` and compare reward curves; at
`staleness_threshold=1` they should overlap within noise.

## Limitations

- **Abort semantics (today's AR server).** `abort_all_requests` drains in-flight
  requests first (up to its drain window), so a weight sync can wait on the
  freshest generations; raise `parameter_sync_step` if sync stalls dominate.
  Requests that get hard-aborted are synthesized with zero generated tokens, so
  a "continuation" restarts generation from the prompt (correct, but equivalent
  to whole-sample retry in cost), and a hard abort with log-probs requested can
  mark the group as failed. True mid-sequence resume for AR omni arrives with
  the #290 server hardening.
- Single-node standalone replicas for AR omni (`vLLMOmniHttpServer.run_headless`
  is not implemented).
- Hybrid (colocated) replicas are not idle: they serve the first sampling
  window and every validation (kept current via the colocated checkpoint
  engine), and sleep during training phases; idle-borrowing beyond that follows
  the upstream `should_switch_to_rollout` TODO.
- `weight_sync_exclude_regex` requires fsdp2 (FSDP1 exposes flat parameters
  only) and a real-weight `load_format`.
- NPU AR sleep/wake relies on vllm-ascend behavior.
- Decoupled-PPO-style correction for version-spanning sequences is tracked
  upstream; use `rollout_correction` (e.g. TIS) when raising staleness.
