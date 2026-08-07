---
name: verl-omni-rollout-pipeline
description: Write or debug a vllm-omni custom rollout pipeline adapter for verl-omni (VllmOmniPipelineBase registration, self-contained pipelines for architectures vllm-omni does not know, FlowGRPO trajectory outputs). Use when implementing the rollout side of a model integration or when rollout generation misbehaves.
---

# vllm-omni Rollout Pipeline Adapters

## How dispatch works (read once, saves hours)

verl-omni's async server (`vllm_omni_async_server.py`) looks up
`VllmOmniPipelineBase.get_pipeline_path(architecture, algorithm)` — the
architecture is the checkpoint's `model_index.json` `_class_name`. When
registered, it passes the dotted class path to vllm-omni as
`custom_pipeline_args={"pipeline_class": ...}` with a dummy initial load.
Inside each vllm-omni worker, `re_init_pipeline` resolves that path and calls
`YourPipeline(od_config=od_config)` **on CPU**, then moves it to the device.
Consequences:

- Your class must be importable by its dotted path inside the engine worker
  process and constructible with only `od_config`.
- Never allocate CUDA tensors in `__init__` (breaks sleep-mode memory
  tracking; the loader moves the module afterwards).
- The loader then calls `pipeline.load_weights(<weights from
  self.weights_sources>)`. If you return `None` from `load_weights`, the
  strict "all params loaded" check is skipped.

## Two kinds of adapters

**A. Subclass of an upstream vllm-omni pipeline** (architecture already in
`vllm_omni/diffusion/models/`): inherit it, swap the scheduler for
`verl_omni.pipelines.schedulers.FlowMatchSDEDiscreteScheduler`, override
`forward` to accept token-id prompts and return trajectories. Weights load
through the parent's `weights_sources`. LoRA sync works (vLLM-native
layers). Template: the Z-Image / Qwen-Image `vllm_omni_rollout_adapter.py`.

**B. Self-contained pipeline** (architecture unknown to vllm-omni): build a
plain `nn.Module` that loads every component itself in `__init__` with HF
`from_pretrained` (on CPU, dtype from `od_config.dtype`, subfolders from the
checkpoint), typically: text encoder, vendored transformer, VAE, tokenizer.
Then:

- `def load_weights(self, weights)`: at init the loader passes an empty
  iterator — weights are already loaded; but the SAME method receives
  full-weight sync updates from the trainer as `("transformer."+name, tensor)`
  pairs. Implement: strip the `transformer.` prefix, `copy_` into
  `self.transformer`'s parameters, ignore other prefixes, return `None`.
- LoRA sync does NOT work for type B (vllm-omni's `DiffusionLoRAManager`
  wraps only vLLM-native linear layers). Train full-weight, or upstream the
  pipeline into vllm-omni later.

## The `forward` contract (both kinds)

```python
def forward(self, req, ...) -> DiffusionOutput
```

- The pinned vllm-omni passes a `DiffusionRequestBatch`
  (`vllm_omni.diffusion.worker.request_batch`), not a bare
  `OmniDiffusionRequest`: the per-request payload dict exposes
  `custom_prompt` with keys `prompt_token_ids` / `prompt_mask` (and
  `negative_*`). Check the installed vllm-omni source for the exact shape
  before coding — this changed across versions. During engine warm-up the
  request may carry only raw text — return
  `DiffusionOutput(output=None, custom_output={})` when neither ids nor
  embeds are present.
- Read generation params from `req.sampling_params` (height, width,
  num_inference_steps, seed, `guidance_scale_provided`, and free-form
  `extra_args` for algo knobs like `noise_level`, `sde_window_size`,
  `sde_type`, `logprobs`).
- The trainer tokenizes with `add_generation_prompt=True`. If the model's
  deployed encoder does not (check upstream!), strip the trailing
  generation-prompt token ids before encoding, in ONE place next to the
  encode call, with a comment.
- Denoise loop: for each scheduler timestep, convert to the model's timestep
  convention, forward the transformer, convert the prediction to velocity,
  then `FlowMatchSDEDiscreteScheduler.step(..., noise_level=...,
  return_logprobs=...)`. Collect `all_latents` (T+1 entries: pre-step latent
  plus each post-step latent inside the SDE window), `all_log_probs`,
  `all_timesteps`.
- Return `DiffusionOutput(output=<decoded image>, custom_output={...})` with
  keys `all_latents`, `all_log_probs`, `all_timesteps`, `prompt_embeds`,
  `prompt_embeds_mask` (+negatives). `custom_output` tensors must be CPU;
  the trainer pads `prompt_embeds*` to
  `rollout.pipeline.max_sequence_length`.

## Scheduler notes

- `FlowMatchSDEDiscreteScheduler` extends diffusers'
  `FlowMatchEulerDiscreteScheduler`; timesteps are `sigma*1000`, sigmas =
  noise fraction, descending; `step()` adds SDE noise and returns per-step
  Gaussian log-probs.
- To reproduce a bespoke upstream schedule exactly, pass explicit
  `set_timesteps(sigmas=[...], mu=...)` (construct the scheduler with
  `use_dynamic_shifting=True` when using `mu`; diffusers' exponential time
  shift `exp(mu)/(exp(mu)+(1/σ−1))` equals the common "v1 logistic" shift).
- If the final sigma is 0, drop the last timestep from the SDE loop (a
  0-noise terminal step has no log-prob) — see
  `get_z_image_flow_grpo_timesteps`.

## Debugging

- Pipeline import errors surface inside Ray worker logs
  (`/tmp/ray/session_*/logs/`), not the driver; grep for your module name.
- `sampling_params` silently drops unknown keys — algo knobs must go under
  `rollout.algo.*` (mapped to `extra_args`), not invented top-level names.
- Wrong images but no crash: check timestep convention and prediction-target
  sign first; both fail silently.
- OOM at load: components load on CPU then move — check the worker's
  `gpu_memory_utilization` and whether sleep mode is enabled rather than
  shrinking the model.
