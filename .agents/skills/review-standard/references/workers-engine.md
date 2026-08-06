# Reviewing workers / engine changes

Scope: `verl_omni/workers/**` (engine_workers.py, engine/fsdp, engine/veomni, config,
checkpoint_engine.py, utils), `verl_omni/models/**`, `tests/workers/**`.

### Key invariants

- `verl_omni/workers/engine_workers.py` (TrainingWorker, ActorRolloutRefWorker) is
  engine-agnostic. No diffusers-/model-family-specific parsing or branching belongs there.
- Config defaults and derived values live in the config dataclass `__post_init__`
  (`verl_omni/workers/config/diffusion/model.py`, `verl_omni/workers/config/omni/model.py`),
  never patched at worker call sites.
- Engine hierarchy: `DiffusersFSDPEngine(LoRAAdapterMixin, BaseEngine)` with per-algorithm
  subclasses (PPO/DPO/NFT) in `verl_omni/workers/engine/fsdp/diffusers_impl.py`;
  `OmniFSDPEngine(FSDPEngineWithLMHead)` in `engine/fsdp/omni_impl.py` subclasses upstream
  verl and must stay a minimal delta.
- LoRA adapter lifecycle goes in `verl_omni/workers/engine/lora_adapter_mixin.py`;
  FSDP/PEFT wrapping specifics stay in the engine impl, not the mixin.
- Custom overrides of upstream verl classes (e.g. `OmniCheckpointEngineManager`) exist only
  where upstream provably cannot serve (vLLMOmniAsyncServer weight-send differs from
  vLLMAsyncServer). Each new override needs the same justification.
- Algorithm-specific helpers (flowgrpo/mixgrpo-only, MFU metadata) live in named util modules
  (`verl_omni/utils/mfu/`) with a TODO stating the limitation — never inline in worker classes.
- Optional model families / engine backends are lazy-imported; no import-time warnings for
  missing optional deps (fail at use time).
- Diffusion and Omni config stacks are intentionally decoupled — do not introduce shared base
  configs between them yet.

### Review checklist

- Verify every added function, config field, and dispatch registration is exercised by code in
  this same PR; anything "for a later PR" must be dropped now and re-added when used.
- Verify no file is touched for reasons outside the PR title's modules; request revert of
  drive-by changes.
- Verify bundled workarounds/patch-fixes are removed — root-cause fixes go in separate PRs.
- For any new class subclassing or paralleling upstream verl (worker, engine, checkpoint
  manager, agent loop, loss config): demand the specific upstream gap; prefer a thin wrapper
  or fusing behavior into the existing engine class over a new worker type.
- Check structural changes against the current RFC (v1 omni design) and upstream verl v1.
- Check config keys/defaults against main and concurrently-open PRs for naming drift
  (e.g. `model_type: "omni"` vs `"omni_model"`).
- Check method families are named in parallel and keep standard RL/diffusion terminology
  (`prepare_model_inputs_for_policy_gradient` / `..._for_direct_preference`); resist renames
  that genericize algorithmically meaningful names (`forward_and_sample_previous_step`).
- Require `# TODO(owner): remove when X` on every knowingly-temporary block or test hack.
- Require a dedicated test file per new engine/backend in `tests/workers/`
  (pattern: `test_diffusers_veomni_engine.py`, `test_omni_fsdp_engine_on_cpu.py`) and suggest a
  CI smoke test; behavior that deviates from upstream verl deserves a test capturing the delta.
- Check FSDP-version symmetry: any `fsdp_version == 1` gate must have a correct FSDP2 path
  (offload/onload, state-dict, adapter contexts).
- Check distributed math uses the right process group: metrics gathered over `dp_group` must
  not be normalized by global world size when sequence parallel / Ulysses shrinks dp.
- Check loss/util functions stay pure: no `del` or in-place mutation of input dicts.
- Check abstract methods have a body (`pass`), duplicated setup calls are merged
  (e.g. a single `set_attention_backend` site), and docstrings/comments are 1-2 tidy lines.
- Ask "why" on every deletion (why safe to remove) and every new indirection (what breaks
  without it); accept concrete technical answers, then move on.
- Treat bot (gemini/Copilot) comments as leads: confirm against config invariants before
  endorsing; reject defensive handling for states the config system makes impossible
  (e.g. a backend field that "cannot be None").

### Common pitfalls

- Config patched in `engine_workers.py` instead of `DiffusionModelConfig.__post_init__`
  (`verl_omni/workers/config/diffusion/model.py`) — recurring in metric/feature PRs.
- LoRA logic scattered across engine impls instead of `LoRAAdapterMixin`
  (`verl_omni/workers/engine/lora_adapter_mixin.py`) — or, inversely, FSDP-specific PEFT
  wrapping stuffed into the mixin.
- `is_fsdp_module`-style checks handling only FSDP1, silently skipping FSDP2 offloaded params
  (`verl_omni/workers/engine/fsdp/diffusers_impl.py`).
- Model-agnostic loading logic hidden in a model-specific patch file
  (`verl_omni/models/transformers/qwen3_omni_thinker.py`) — generic parts go to the registry.
- Sprawling monkeypatches of HF model classes; ask which lines are truly required.
- AI-generated bloat: functions returning their own input, verbose docstrings, decorative
  markdown, redundant fallback logic ("too AI" is a review verdict here).
- Tests loosened (thresholds/tolerances) to pass instead of fixing the regression — never
  allow; thresholds are guarded by policy.

### Red flags that should block approval

- New worker/manager class duplicating an upstream verl abstraction without a stated,
  verified upstream gap.
- Dead code ("will be used in the next PR") anywhere in the diff.
- A bug workaround bundled into a feature PR with no root cause identified.
- Refactor-only churn mixed into a functional PR (split required).
- Numeric test tolerances relaxed.
- In-place mutation (`del`) of shared data dicts in loss/util functions.
- New GPU/engine path with no test file and no CI smoke coverage plan.
- Weight-sync / checkpoint-engine / zmq-handle changes not signed off by the owning expert
  (these are always routed to a domain expert before merge).
