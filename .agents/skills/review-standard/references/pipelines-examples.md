## Reviewing pipelines / examples changes

Scope: `verl_omni/pipelines/**` (adapters, `model_base.py`, `utils.py`,
`request_batch.py`, `schedulers/`) and `examples/**` (trainer scripts, data_process,
READMEs). Mike's stance in this area: interrogate every line's right to exist, keep
all pipelines structurally identical to each other, and protect `model_base` as the
migration contract for the next model.

### Key invariants

- **Registry dispatch, never manual wiring.** Adapters are found by
  `(architecture, algorithm)` via `verl_omni/pipelines/__init__.py`; example scripts
  and configs must never name a pipeline class or set `external_lib` for in-tree
  code ("it automatically find the registered pipeline by name. So user don't need
  to modified it", #8). New packages must be added to `pipelines/__init__.py` with
  the house pattern: star import + `__all__ += list(<pkg>.__all__)`.
- **Directory naming:** diffusion pipelines are `<model>_<algorithm>`
  (`qwen_image_flow_grpo`, `sd3_dpo`) because a custom scheduler is injected per
  algorithm; omni models are just `<model>` (`qwen3_omni`) because algorithm is
  decoupled (#253).
- **Algorithm variants reuse, models disentangle.** A new algorithm on an existing
  model reuses that model's components (GRPO-Guard = FlowGRPO + a different loss,
  #48) via subclassing, `common.py` imports, or mixins — but a rollout adapter must
  not import from a *different algorithm's* package (#106: "Please implement rollout
  for diffusion-nft alone").
- **`model_base` is minimal.** "I prefer not to add the interface unless it is
  really necessary" (#253). New hooks need >1 real implementer, a docstring saying
  which algorithms need them, and defaults that keep new integrators unaware
  (#243: `supports_request_batch=True` by default; "user should not aware this
  unless they encounter any error").
- **Fail loudly.** No silent fallbacks, no `try/except`-with-default on required
  attributes, no `del` on caller-owned inputs ("no fallback since it will cause
  silent bug", #58; "do not `del`. It make the function impure", #106). If a guard
  matters, make it an `assert`.
- **Vendored code stays diffable.** Blocks copied from vllm-omni/diffusers keep
  their comments, formatting, and math so users can compare side by side (#8).
  When vllm-omni and diffusers disagree, diffusers semantics win (#81). Every
  temporary patch carries a `# TODO` naming its removal condition (upstream PR or
  version bump).
- **Examples are the product.** One canonical script shape (see
  `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh`); layout is
  `examples/<algo>_trainer/<model>/run_*.sh` + README; no per-recipe YAML — use the
  generated default trainer config plus CLI overrides (#290, #329).

### Review checklist

Library (`verl_omni/pipelines/`):
- [ ] For every new helper/file: does an equivalent already exist in a sibling
  package (`qwen_image_flow_grpo/common.py`, `pipelines/utils.py`,
  `request_batch.py`)? If yes, import or extend it — reject the copy.
- [ ] For every structural change (new request type, renamed field, new trajectory
  key): is it applied to ALL pipelines, or is there an explicit follow-up-PR note?
  ("move all pipelines into trajectory_*, not sd3 only", #354).
- [ ] New `model_base` hook: demand the docstring state when to implement it and
  which algorithms skip it ("in what situation we need to implement
  `forward_velocity`", #106). If only one model would implement it, push it into
  that model's adapter (override `forward`) instead.
- [ ] Any fallback, default-on-missing, broad `except`, or silent clamp: remove it
  or convert to an assert. A numerics guard in shared code must be scoped to the
  pipeline that needs it (#260 `comb_pred` clamp).
- [ ] Worker-side caching/copying/atomic-write machinery: ask if it can run once on
  the driver instead ("possible to move this to driver side, so we don't need copy
  or cache", #260). Env vars appearing "from nowhere" block.
- [ ] Access to private attributes of upstream classes (`pipe._foo = ...`): "just
  too fragile, can we provide a more elegant method?" (#284).
- [ ] Refactor deleting existing comments/docstrings: reject — they are the
  side-by-side tutorial vs upstream (#8, x12).
- [ ] New shared module → unit test in the same PR (#286); modified in-repo model
  (`bagel_flow_grpo/bagel_model.py`) → forward/backward equivalence test vs the old
  behavior (#333); debug/tracing instrumentation → belongs in a toy pipeline under
  `tests/`, never in the real adapter (#112).
- [ ] Workaround for an engine gap: is there an upstream issue/feature request
  filed, and a TODO with the removal condition? ("Can we make a feature request for
  this?", #81).

Examples (`examples/`):
- [ ] Script matches the house template and its siblings exactly in structure and
  variable names (`MODEL_PATH`, `ROLLOUT_TP`, `WORKSPACE=${WORKSPACE:-$HOME}`);
  "Keep all training scripts in the same format" (#2).
- [ ] Script is actually runnable as committed — paths resolvable, flags exist in
  the config schema, model IDs real (#60: "the script will not be runnable").
- [ ] Numbers match the docs: LR, SP degree, batch sizes, training steps must equal
  what the benchmark/README reports (#55, #104). Changing a shared default (e.g.
  training steps) means changing every sibling script or reverting.
- [ ] New variant script must have a stated, real difference from existing scripts
  and a name that says what it is (`*_h200_mfu_optimized`); otherwise delete
  (#128: "if there is no difference, pls drop"). Interrogate each "optimization"
  flag — does it actually change behavior in this configuration?
- [ ] No custom YAML per recipe (#290, #329); no reward functions in `examples/`
  (they live in `verl_omni/utils/reward_score/`, #66); no committed datasets —
  reference the HF hub source instead (#142).
- [ ] New algorithm dir includes/updates its `README.md` (#58, #66); new
  model under an algorithm goes in a model subfolder (#137).
- [ ] Perf/hardware-specific scripts (NPU, multi-node): ask "is this validated?"
  and expect run evidence (#341).

### Common pitfalls

- Copy-paste drift between model variants: a fix landing in
  `qwen_image_flow_grpo/vllm_omni_rollout_adapter.py` but not in the edit/NFT/DPO
  siblings, or a guard removed during "unification" that an existing path relied on
  (#286 empty-generator-list regression — "pls address this").
- Forgetting `verl_omni/pipelines/__init__.py`: an unregistered package silently
  does not exist; also verify the `__all__ += list(<pkg>.__all__)` aggregation line.
- `__all__` buried mid-file — Mike wants it at the top (#8, #238).
- Over-engineering tells that he names as AI output: atomic copytree/rmtree
  publishing, cache-key hashing, multi-layer wrappers with one caller, decode→encode
  round-trips of data already encoded upstream ("it is too AI and overly
  complicated, I suggest just remove them", #238; "It is OK for AI to work aground
  it. But it violates the design of verl and verl-omni", #178).
- Bot-review noise: adjudicate every gemini/Copilot comment explicitly — most are
  "not relavent" or wrong for the specific model ("bagel only. works fine.", #137),
  but adopt the correct ones ("This is a good suggetion from gemini", #8).
- Unrelated hunks riding along ("why change this file?", #290) — every touched file
  must be explainable by the PR title.

### Red flags that should block approval

- A new pipeline package that copies a sibling wholesale instead of subclassing
  (`QwenImageEditPlusFlowGRPO` subclasses `QwenImage` — that is the model).
- Silent fallback/defaulting on required config or model attributes.
- An in-repo model implementation for a model that a third-party engine
  (diffusers/transformers/veomni) already supports — in-repo models are last-resort
  only, with warning + TODO (#66).
- Example script whose hyperparameters contradict the benchmark doc, or that cannot
  run as committed.
- Untested new shared module, or edits to `bagel_model.py`-class files without an
  equivalence test.
- Dead/unreachable branches ("so non-CB is not trainable, right?" → drop or move to
  `/experimental`, #178).
- Changes to code vendored from vllm-omni/diffusers that alter the algorithm or
  strip the comparison comments.
