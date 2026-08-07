---
name: verl-omni-model-integration
description: Migrate a new generative model (diffusion T2I/I2I or non-diffusers) into verl-omni for RL training (FlowGRPO etc.). Use when asked to integrate, add, or port a model into this repository. Orchestrates env setup, upstream analysis, adapter implementation, end-to-end run, and testing/CI.
---

# Integrating a New Model into verl-omni

You are migrating a model into verl-omni's RL training stack. The repository
already documents the file-level mechanics — your job is to follow those
guides in the right order, close the gaps they cannot know about (the new
model's quirks), and leave the trail better documented than you found it.

## Phase 0 — Environment and duplicate check

1. Run the `verl-omni-env-setup` skill.
2. Duplicate-work check per `AGENTS.md` (gh issue/PR search). Stop if an open
   PR already integrates this model.
3. Create a feature branch from `main` of `verl-project/verl-omni`.

## Phase 1 — Classify the model, pick the guide

Read the model's `model_index.json` (diffusers layout) or HF config, plus its
reference inference code. Then choose the primary guide in
`docs/contributing/`:

| Model shape | Guide |
| --- | --- |
| diffusers-loadable T2I diffusion | `integrating_a_diffusion_model.md` |
| image-conditioned (I2I / edit) diffusion | `integrating_an_i2i_diffusion_model.md` |
| not loadable via stock `diffusers.AutoModel` (custom code, `trust_remote_code`, unified AR+diffusion) | `integrating_a_non_diffusers_model.md` |
| needs stepwise continuous-batching rollout | `integrating_a_stepwise_continuous_batching_model.md` (additive) |

Read the chosen guide fully before writing code. Also read one existing
integration package under `verl_omni/pipelines/` as a living reference —
pick the registered model closest in architecture (search
`DiffusionModelBase.register` and `VllmOmniPipelineBase.register` for the
list), and verify the package actually contains source files before relying
on it (stale `__pycache__`-only directories exist). The Qwen-Image FlowGRPO
package is the most complete in-tree template.

## Phase 2 — Upstream analysis (do this before any adapter code)

Answer ALL of the following from the model's own code, and write the answers
into a scratch note — every one maps to a line of adapter code:

1. **Rollout support**: does the pinned vllm-omni support this architecture
   natively (search `vllm_omni/diffusion/models/`)? If not, the rollout
   adapter must be fully self-contained (see `verl-omni-rollout-pipeline`
   skill, "self-contained pipeline" section) — this is supported and does not
   require forking vllm-omni.
2. **Prompt encoding**: exact chat template / system prompt, which encoder
   module, which hidden states (last layer? layer -2?), padding side, and
   whether upstream appends a generation prompt. verl's agent loop tokenizes
   with `add_generation_prompt=True`; if upstream does not, strip those
   trailing tokens in the rollout adapter's encode path or override the chat
   template via `actor_rollout_ref.model.custom_chat_template`.
3. **Latent geometry**: latent channels, VAE scale factor, patch size,
   resolution divisibility constraints, packed (B, L, C) vs unpacked
   (B, C, H, W) at the pipeline boundary.
4. **Timestep convention**: what does the transformer receive — noise
   fraction σ, signal fraction (1-σ), or an integer index? Any internal
   `timestep_scale`? verl's `FlowMatchSDEDiscreteScheduler` timesteps are
   `sigma*1000` with σ = noise fraction, descending.
5. **Prediction target**: what does the model output mean? verl's SDE
   scheduler expects flow-matching velocity `v = ε − x0` (d x / dσ_noise).
   Lumina/Z-Image lineage models predict `x0 − ε`: negate the output.
   Distilled few-step models (DMD/Turbo) that "predict x0 then renoise" are
   the same velocity family — map algebraically, do not port their sampler.
6. **Sigma schedule**: reproduce the upstream schedule exactly (including
   time-shift: static or dynamic, shift formula, seq_len). Reproduce it via
   `set_timesteps(sigmas=..., mu=...)` on the shared SDE scheduler rather
   than porting the upstream scheduler class.
7. **CFG**: does the deployed model use CFG / negative prompts? Distilled
   models usually forbid it — then skip negative embeds everywhere.
8. **VAE decode**: scaling_factor / shift_factor and the exact
   (de)normalization order.
9. **Custom model code**: if the transformer is not in stock diffusers,
   vendor it (see Phase 3).

Verify your understanding by RUNNING the upstream inference demo once on this
machine and saving the output image. This catches env quirks (env vars,
flash-attn assumptions, dtype tricks) before they surface inside Ray workers.

## Phase 3 — Vendoring custom model code (only when needed)

When the transformer requires the upstream repo on `PYTHONPATH` or
`trust_remote_code`, vendor the minimal module set into
`verl_omni/pipelines/<model>_flow_grpo/<model>_model/`:

- Copy only what the transformer imports (trace imports transitively).
- Rewrite intra-package imports to relative imports.
- Remove environment-dependent behavior: `os.environ["device"]`-style gating,
  Triton kernels without CPU fallback (replace with `torch.nn` equivalents —
  e.g. `torch.nn.RMSNorm`), hard `flash_attn` requirements (guard with an
  availability check and fall back to SDPA).
- Keep the upstream license header, add the Bytedance line above it, and add
  a provenance comment: source repo, commit, and each functional
  modification.
- Verify: import the class and instantiate it on the `meta` device from the
  real checkpoint config; parameter count must match the model card.

## Phase 4 — Implement the integration package

Create `verl_omni/pipelines/<model>_flow_grpo/` with `common.py` (shared
constants: scheduler setup, prediction-target conversion, VAE factors),
`diffusers_training_adapter.py` (`DiffusionModelBase.register(...)`), and
`vllm_omni_rollout_adapter.py` (`VllmOmniPipelineBase.register(...)`).
Register both under the `_class_name` from the checkpoint's
`model_index.json`, then wire the package into
`verl_omni/pipelines/__init__.py`. Details and contracts: the chosen Phase 1
guide plus the `verl-omni-rollout-pipeline` skill.

Non-obvious contracts the guides under-emphasize:

- Training and rollout MUST share one source of truth for the sigma schedule
  and the model-output-to-velocity conversion. Put both in `common.py`; a
  mismatch shows up only as silently wrong importance ratios.
- `prepare_model_inputs` returns kwargs matching the transformer's `forward`
  signature exactly (the engine calls `module(**model_inputs)`).
- If the transformer's `forward(return_dict=False)` returns a bare tensor
  (not a tuple), the default `DiffusionModelBase.forward` — which does
  `module(**inputs)[0]` — silently returns the first batch element. Override
  `forward` in the training adapter for such models.
- Custom `build_module` override loads vendored transformers; FSDP needs
  `_no_split_modules` on the model class.
- Weight sync back to rollout: LoRA sync (`add_lora`) only works when the
  rollout transformer is built from vLLM-native linear layers (i.e. the
  pipeline lives upstream in vllm-omni). A repo-local rollout pipeline built
  on plain `nn.Linear` must train FULL-WEIGHT (`lora_rank=0`) and implement
  `load_weights(weights)` mapping the `"transformer."`-prefixed names the
  trainer streams. State this limitation in the example script.

## Phase 5 — Data + example script

- Data preprocessor in `examples/flowgrpo_trainer/data_process/`: write
  `prompt` as chat messages reproducing the model's exact template (Phase 2
  item 2). Reuse the OCR dataset flow from an existing preprocessor.
- Launch script in `examples/flowgrpo_trainer/<model>/`: copy the nearest
  model's script; change model path, resolution constraints, scheduler
  params, and set `actor_rollout_ref.model.tokenizer_path` if the tokenizer
  lives in a non-standard subfolder (e.g. `processor/`).

## Phase 6 — End-to-end verification (in this order, cheap to expensive)

1. **Import + registry**: `python -c` importing both adapters; assert
   registry keys.
2. **Offline parity**: load real weights, run the rollout adapter's diffuse
   path standalone against the upstream demo with the same seed. Check, in
   order: (a) prompt embeddings match exactly; (b) teacher-forced
   single-step parity — feed upstream's per-step latents through your
   mapping and compare predictions (this isolates the math from noise
   handling); (c) images visually equivalent. Full-trajectory latent match
   is only achievable when the upstream sampler is deterministic given the
   seed — predict-x0-then-renoise (DMD) samplers draw fresh noise per step
   and will diverge from the noise_level=0 ODE path by design; do not chase
   that diff. If vendoring substituted kernels (e.g. Triton RMSNorm →
   torch), expect O(1e-1) bf16 output drift over deep stacks; verify the
   mapping with matched kernels before blaming the adapter.
3. **Smoke train**: 4-GPU run, few steps, tiny batch, a rule-based reward
   (e.g. jpeg compressibility) to avoid a reward-model dependency. Wire rule
   rewards through `MultiVisualRewardManager` (see the qwen e2e smoke
   script) — the default `VisualRewardManager` passes
   `data_source`/`ground_truth`/`extra_info` kwargs that narrow-signature
   rule functions reject. Watch for: OOM (reduce resolution first, not
   batch), Ray actor deaths (read the worker log file it names, not just
   the driver output).
4. **Real reward run**: the model's intended reward (e.g. OCR) for enough
   steps to see reward move.

Record EVERY discrepancy between the guides and what you actually had to do.
These go into the PR description and, when general, into the guides
themselves (read `docs/contributing/editing-agent-instructions.md` first).

## Phase 7 — Tests + CI

Run the `verl-omni-testing-ci` skill: L1 CPU tests for the adapter package,
tiny-random builder + L2 GPU smoke script, wire into
`.github/workflows/gpu_smoke.yml`.

## Fallback and escalation

- If the model cannot satisfy the log-probability contract (e.g. sampler is
  not expressible as Gaussian transitions), stop and report — do not force
  FlowGRPO onto it.
- If rollout requires engine changes inside vllm-omni itself, prepare the
  verl-omni side, then open a vllm-omni issue; do not fork the pin.
