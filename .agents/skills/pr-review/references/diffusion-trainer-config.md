## Reviewing diffusion trainer / config changes

Scope: `verl_omni/trainer/diffusion/` (incl. `v1/`), `verl_omni/trainer/config/` (diffusion yamls
and `_generated_*.yaml`), `verl_omni/workers/config/diffusion/`, related algo/model code and
`tests/trainer/diffusion/`.

### Key invariants

- **Upstream verl is the reference.** Anything not diffusion-specific must match verl's
  structure: `_compute_*` methods keep verl's return signatures (only `_compute_old_log_prob`
  returns a tuple — carry extra metrics inside the TensorDict/DataProto instead of widening
  returns); `reduce_metrics`/`rename_dict` are used only in `fit`; separate trainer yamls follow
  verl precedent (e.g. a separate veomni trainer yaml like verl's `ppo_veomni_trainer.yaml`);
  never re-register mode names verl already registers (e.g. "sync"). Stated goal: "verl's user
  has zero-cost to use verl-omni."
- **Config taxonomy** (in `verl_omni/workers/config/diffusion/rollout.py`): every new rollout key
  goes into exactly one bucket — `DiffusionRolloutConfig` (engine-general),
  `DiffusionSamplingConfig` (validation sampling, verl practice),
  `DiffusionRolloutAlgoConfig` (algorithm-dependent), `DiffusionPipelineConfig`
  (model-dependent, e.g. height/inference_steps). Do NOT create per-algorithm config subclasses;
  add labeled fields ("# MixGRPO-only configs") to the shared class. Reuse existing key families
  plus a boolean switch over inventing parallel ones (e.g. reuse
  `sde_window_range`/`sde_window_size` + `sde_continue` flag, not new `sde_steps`/`num_sde_steps`).
- **Seeding is decomposed, never global.** `data.seed` and `rollout.seed` are independent knobs;
  rollout seed derives per step as `seed + global_steps - 1` with per-row global indices
  (`_rollout_seed_global_idx` in `verl_omni/trainer/diffusion/ray_diffusion_trainer.py`).
  "Seed everything" is banned — it silently makes all n rollouts of a prompt identical and
  collapses the FlowGRPO group advantage. Algorithm-internal RNG knobs get specific names
  (`sde_window_seed`), never bare `seed`.
- **Fail fast on config access.** No `.get("key", default)` / defaults at call sites to paper
  over missing config — read the attribute and let it raise. Validation guards live where the
  value is materialized (`__post_init__` of the config class, or the engine worker right after
  init) — never scattered in the trainer loop.
- **`ray_diffusion_trainer.py` is a protected choke point.** Model/scheduler-dependent behavior
  belongs in `DiffusionModelBase` / `verl_omni/pipelines/utils.py::build_scheduler`, not patched
  into the trainer. Metric computation belongs in
  `verl_omni/trainer/diffusion/diffusion_metric_utils.py` (`compute_xxx_metrics_diffusion`
  helpers), not inline in `fit`. Monkey-patching is rejected outright — extend classes instead.
- **One trainer per paradigm.** New training paradigms (DPO/SFT/NFT) get their own trainer file
  inheriting the base, not `if algo == ...` branches in the FlowGRPO trainer. Omni trainers must
  not modify `BaseRayDiffusionTrainer` — disentangle instead ("but you are doing omni model?").
- **Losses:** one class per loss, named `<Algo>Loss` (not `...LossFunc`), body inlined into the
  class (no separate `compute_diffusion_loss_xxx` free function). Base classes are plain `ABC` —
  no `TypeVar` generics. Abstract methods need a docstring stating their contract. Shared
  `DiffusionLossConfig`, not per-algo config subclasses.

### Review checklist

- Grep every added symbol (function, class, config key, constant) for real usage; demand removal
  of anything write-only or made idle by a later refactor iteration — re-check on every push.
- For each changed yaml default: require an explicit stated reason in the PR/thread; otherwise
  request revert ("please do not change the default value unless you have provided the reason").
- Check each new yaml key sits in the correct config bucket and its dataclass mirror
  (`verl_omni/workers/config/diffusion/*.py`) matches the yaml in name, type, and position;
  string-like values must not be encoded as numbers.
- Yaml comments: one line, state what the knob does and enumerate allowed values; no algorithm
  tutorials, no paper math. Name keys for the mechanism, not the first algorithm using it
  ("not only MixGRPO will use this") — but label truly algo-specific blocks "# <Algo>-only".
- Reject backwards-compatibility shims, deprecation aliases, and "legacy" fallbacks — no public
  release exists, so break cleanly.
- Verify every file in the diff is required by the PR's stated purpose. Unrelated defaults,
  profiler tweaks, seed features, regenerated `_generated_*_trainer.yaml` churn (a diffusion PR
  must not touch `_generated_omni_trainer.yaml` and vice versa — pre-commit regeneration side
  effects are a known cause), and edits to shared base classes all go to separate PRs.
- After a rebase, check for accidentally deleted or duplicated yaml lines and for re-adding keys
  already on main.
- Diff comments against the base: restored/reworded/deleted pre-existing comments without
  functional cause are findings ("keep old comments stays").
- For any deviation from verl in a shared code path, ask "why is this different from verl?" and
  require either alignment or a concrete justification. If the author claims verl-alignment,
  verify against actual verl source before accepting.
- Seeding changes: confirm data/rollout seeds stay separate, per-rollout uniqueness is preserved,
  and 1-indexed `global_steps` is converted before hitting 0-indexed schedulers.
- Public APIs need Google-style docstrings (summary / Args / Returns); private helpers should
  usually have none. Empty class/function bodies end with `pass` (or `raise NotImplementedError`
  for abstract-ish stubs) — and `del` on locals is unsafe, use `pass`/scope instead.
- New tests: check they don't overlap existing CPU tests
  (`tests/trainer/diffusion/test_diffusion_core_algos_on_cpu.py` for algo math); every new
  reusable helper or feature should come with one light-weight `*_on_cpu.py` test.
- If you cannot explain what a block does after reading it, say so — "I dont get what happens
  here" is a valid, blocking review comment in this repo; the fix is a rewrite, a docstring, or
  deletion.

### Common pitfalls (seen repeatedly)

- Helpers that became dead after review iterations (`_TRAINER_ONLY_FIELDS`, `to_rollout_dict`,
  `collect_mode`) surviving into the final diff — including their docs.
- One-line wrapper functions (`_build_rollout_seed`) instead of inlining at the call site.
- New dataclasses/enums for two fields; enum types forcing "tedious checks" — use plain strings
  with the allowed values listed in the yaml comment.
- Metric/MFU logic pasted per-call-site instead of computed once, verl-style, in
  `diffusion_metric_utils.py`.
- Scheduler construction special-cased in the trainer instead of
  `verl_omni/pipelines/utils.py::build_scheduler` reading the user's selected scheduler.
- Section-divider banners (`# ----------------`) and verbose AI narration in
  `verl_omni/trainer/diffusion/v1/` files — "not our repo style".
- Config key placed under a stale section after the schema moved (e.g. under
  `rollout_correction` in `verl_omni/trainer/config/diffusion/actor/diffusion_actor.yaml`).

### Red flags — block approval

- "Seed everything" or any single RNG stream shared across rollout `n` (silent FlowGRPO
  advantage collapse).
- `config.get("key", default)` masking missing configuration.
- Changed return signature of a `_compute_*` trainer method that verl keeps single-valued.
- Monkey-patching in `ray_diffusion_trainer.py`.
- Regenerated or hand-edited `_generated_*_trainer.yaml` outside the PR's module scope.
- Default-value change with no stated reason.
- Unused/"AI slop" code, or code the reviewer cannot understand and the author cannot explain
  crisply in-thread (unexplained = removed).
- Backwards-compat shims for unreleased APIs.
