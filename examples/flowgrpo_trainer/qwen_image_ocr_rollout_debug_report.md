# Qwen-Image OCR Real Rollout Debug Report

## Summary

I ran three short OCR training jobs with the real vLLM-Omni rollout server enabled:

- Full-weight, SP=2: `run_qwen_image_ocr.sh`
- LoRA, no SP: `run_qwen_image_ocr_lora.sh`
- LoRA, SP=2: `run_qwen_image_ocr_lora_sp2.sh`

To make all three runs practical while still exercising the real OCR data, reward model, actor,
FSDP engine, and vLLM-Omni rollout server, I used the same reduced debug workload for each:

- `data.train_batch_size=8`
- `actor_rollout_ref.rollout.n=2`
- `actor_rollout_ref.actor.ppo_mini_batch_size=4`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2`
- `trainer.total_training_steps=3`
- `trainer.test_freq=-1`
- `trainer.save_freq=-1`
- `trainer.logger=["console"]`
- `+trainer.debug_rl_path=True`
- `actor_rollout_ref.rollout.calculate_log_probs=True`

The reduced batch keeps the real system path but should not be interpreted as a convergence run.
It is a behavioral consistency check for the first three updates.

## Instrumentation Added

I added rollout/actor parity metrics in `ray_diffusion_trainer.py`:

- `debug/logprob_parity/mean_abs`
- `debug/logprob_parity/max_abs`
- `debug/logprob_parity/mean_signed`
- `debug/logprob_parity/rollout_mean`
- `debug/logprob_parity/actor_old_mean`

I added policy ratio metrics in `diffusion_algos.py`:

- `actor/log_ratio_mean`
- `actor/log_ratio_std`
- `actor/log_ratio_abs_max`
- `actor/ratio_mean`
- `actor/ratio_std`

I added sampled trainable-parameter update metrics in `engine_workers.py`:

- `actor/debug/param_sample_count`
- `actor/debug/param_sample_norm_before`
- `actor/debug/param_sample_norm_after`
- `actor/debug/param_sample_update_norm`
- `actor/debug/param_sample_update_ratio`

The parameter update metric samples local trainable parameter shards/tensors to avoid cloning the
full Qwen-Image model.

## Results

Full-weight SP=2 completed 3 steps successfully.

- Step 1 reward mean: `0.5371`
- Step 2 reward mean: `0.6077`
- Step 3 reward mean: `0.6876`
- Logprob parity mean abs: `2.85e-05`, `4.22e-05`, `3.56e-05`
- Logprob parity max abs: `1.32e-04`, `2.40e-04`, `1.70e-04`
- PPO KL: `2.26e-05`, `2.10e-05`, `1.21e-05`
- PPO clipfrac: `0.09375`, `0.0625`, `0.0`
- Grad norm: `0.0143`, `0.0195`, `0.0124`
- Sampled update ratio: `1.32e-04`, `8.44e-05`, `5.97e-05`

LoRA no-SP completed 3 steps successfully.

- Step 1 reward mean: `0.6925`
- Step 2 reward mean: `0.5862`
- Step 3 reward mean: `0.7627`
- Logprob parity mean abs: `4.34e-05`, `4.57e-05`, `2.93e-05`
- Logprob parity max abs: `1.27e-04`, `1.34e-04`, `1.55e-04`
- PPO KL: `3.67e-07`, `5.83e-06`, `1.10e-06`
- PPO clipfrac: `0.0`, `0.03125`, `0.0`
- Grad norm: `7.31e-04`, `9.02e-04`, `1.01e-03`
- Sampled update ratio: `9.24e-03`, `6.83e-03`, `5.76e-03`

LoRA SP=2 completed 3 steps successfully.

- Step 1 reward mean: `0.5418`
- Step 2 reward mean: `0.7140`
- Step 3 reward mean: `0.7832`
- Logprob parity mean abs: `3.20e-05`, `3.32e-05`, `2.61e-05`
- Logprob parity max abs: `2.11e-04`, `2.29e-04`, `1.10e-04`
- PPO KL: `5.40e-06`, `5.52e-06`, `5.54e-06`
- PPO clipfrac: `0.0625`, `0.03125`, `0.0`
- Grad norm: `1.03e-03`, `2.83e-03`, `4.58e-04`
- Sampled update ratio: `9.72e-03`, `8.63e-03`, `5.66e-03`

## Interpretation

The full-weight run does not show evidence of actor/rollout sync mismatch in this 3-step real
rollout test. Actor-recomputed old log-probs and rollout-server log-probs agree to roughly
`1e-4` max absolute error, similar to both LoRA runs.

The full-weight run also does not show early PPO ratio instability. Its `ratio_mean` stays near
`0.99998`, `actor/ppo_kl` is around `1e-05` to `2e-05`, and `actor/pg_clipfrac` decreases to zero
by step 3. This is not the signature of an immediately broken forward/loss/update path.

The major difference is update geometry:

- Full-weight sampled update ratio is about `6e-05` to `1.3e-04`.
- LoRA sampled update ratio is about `5.7e-03` to `9.7e-03`.

This is expected because LoRA trains small adapter tensors with LR `3e-4`, while full-weight trains
the base transformer with LR `5e-5`. The full-weight global grad norm is larger than LoRA, but its
sampled normalized update is much smaller. So `actor/grad_norm` alone is not a good cross-method
comparison; normalized update and log-prob/function change are better.

The first three reduced-workload rewards are noisy but not diagnostic of non-convergence. All three
runs can move up/down in three steps because the batch has only 8 prompts and 2 samples per prompt.
This short test validates mechanics and early behavior; it does not explain the 60-90 step
convergence gap by itself.

## Issues Found

The first full-weight debug attempt failed because debug parameter metrics were plain scalars. The
existing metric aggregation path expects `Metric` objects or iterable values. I fixed the debug
metrics to use `Metric(..., AggregationType.MEAN)`, and all subsequent real runs completed.

A smaller SP=2 debug attempt with `data.train_batch_size=4`, `rollout.n=2`, and micro batch size `4`
failed because the local log-prob batch was not divisible by `sp_size * micro_batch_size_per_gpu`.
Using micro batch size `2` and `data.train_batch_size=8` resolved this.

All real runs emitted repeated vLLM-Omni warnings:

- `SHM pack failed, falling back to raw enqueue: Got unsupported ScalarType BFloat16`
- `reset_encoder_cache not yet supported with Orchestrator process`
- `reset_prefix_cache not yet supported with Orchestrator process`

These warnings did not prevent completion, but the bf16 SHM fallback may affect performance.

## Current Suspicion

Based on these 3-step checks, I would deprioritize:

- actor forward correctness
- FlowGRPO loss sign/ratio computation
- actor optimizer step being no-op
- full-weight actor-to-vLLM-Omni sync being stale or mismatched

The more likely explanation for the longer-run full-weight reward fluctuation is optimization
dynamics rather than plumbing:

- Full-weight updates all base parameters and may disturb generation quality over longer horizons
  even when each single update is numerically small.
- LoRA constrains the update subspace and may act as a stabilizer.
- The current full-weight LR may still be too high for base-model RL, or it may need warmup/KL.
- With GRPO and OCR rewards, reward variance and per-prompt zero-std groups can dominate early
  learning. Longer logging should compare `critic/rewards/std_mean`,
  `critic/rewards/zero_std_ratio`, and `actor/pg_clipfrac` over 60-90 steps.

## Recommended Next Test

Run two longer reduced-cost comparisons before changing code:

- Full-weight SP=2 for 30 steps with the same debug metrics.
- LoRA SP=2 for 30 steps with the same debug metrics.

If full-weight parity remains good but reward fluctuates while LoRA improves, test optimization-only
changes:

- full-weight LR `1e-5` or `2e-5`
- LR warmup for the first 10-30 steps
- enable KL regularization or add a smaller KL coefficient sweep
- optionally freeze early blocks or train only attention/MLP subsets before full unfreeze

The key success signal for a fix should be reward trend plus stable `actor/ppo_kl`,
`actor/pg_clipfrac`, and normalized sampled update ratio, not raw `grad_norm` parity with LoRA.
