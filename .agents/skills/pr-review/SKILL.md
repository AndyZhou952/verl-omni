---
name: pr-review
description: "Unified PR review for verl-project/verl-omni. Use for ANY request to review a PR, branch, or diff in this repo ('review PR #N', 'review this change', pre-submission self-review). Self-contained: deep-review methodology, the repo's maintainer review standard, and per-module checklists in references/."
---

# verl-omni PR Review

Review with forensic depth and this repo's maintainer standard. The stance is
**empiricist gatekeeping**: RL training code fails silently, so proof beats reasoning;
every finding is either a reason not to approve yet, or not worth posting. The deepest
house pattern is **interrogative minimalism against upstream verl as the spec** — the
default answer to "why do we need this?" is "delete it, or use verl's design", and a
concrete technical justification flips the verdict immediately. Extensibility comes from
correct placement and registries with a minimal surface, never from speculative hooks.

## Step 0 — Gather

```bash
gh pr view <N> --repo verl-project/verl-omni --json title,body,author,labels,files,reviews
gh pr diff <N> --repo verl-project/verl-omni
gh pr checks <N> --repo verl-project/verl-omni
```

Record process flags immediately: CI never ran ("no checks reported") / red / label-gated
GPU smoke not triggered (`ready-for-ci` labels are stripped on every push); approvals
granted before CI; PR title violating `[{modules}] {type}: {description}` (source of truth:
`tests/special_sanity/check_pr_title.py`); missing `[BREAKING]` on config/API renames;
description missing test commands or training evidence; description not stating **which
configurations were actually exercised**.

## Step 1 — Route to module knowledge

Read `references/repo-map.md` first (module map, blast radius, upstream-alignment rules,
extension seams), then the reference files matching the changed paths (all under this
skill's `references/` directory):

| Changed paths | Reference |
|---|---|
| `verl_omni/workers/rollout/`, `agent_loop/`, `reward_loop/` | `rollout-agent-loop.md` |
| `verl_omni/trainer/{diffusion,config}/`, `workers/config/diffusion/` | `diffusion-trainer-config.md` |
| `verl_omni/pipelines/`, `examples/` | `pipelines-examples.md` |
| `verl_omni/workers/` (rest), `verl_omni/models/` | `workers-engine.md` |
| `verl_omni/utils/`, `docs/`, `tests/`, `.github/` | `utils-tests-docs-ci.md` |

Then read the **current repo state** around every hunk — never review a diff in isolation.
Most blockers live in the interaction between new code and unchanged code.

## Step 2 — Gate zero (before line-level review)

If any of these fail, lead the review with the demand — detailed review does not proceed
without them:

1. **Real problem**: linked issue/RFC or reproduced error log for fixes; clear stated
   intent ("What is the bug are you trying to fix? Can you post a issue first?").
2. **Evidence**: training-affecting changes (trainer math, rollout, engines, examples)
   need a human-readable reward + validation curve in the description — read it
   critically (converging? delta plausible?). Perf claims need a controlled comparison
   (one variable changed). Benchmark numbers must be confirmed as actually measured.
3. **Scope**: single purpose; refactors separate from features; risky infrastructure
   (seeds, determinism, engines) separate from algorithm work. If oversized (>~20 files
   or multiple concerns), propose the concrete split (interface PR / feature PR / docs PR
   last) — docs are the final PR of a series.
4. **Layer**: fixes that belong in verl / vllm-omni / diffusers go upstream — a
   monkey-patch here is a near-veto; a minimal unavoidable patch needs
   `# TODO (<owner>): drop when <condition>. Ref: <upstream link>`.

## Step 3 — Deep pass

### 3a. Judge the idea separately from the implementation
State in one or two sentences what the PR fundamentally does and whether that idea is
right, before line-level criticism. If the idea is wrong or belongs elsewhere, say so
first.

### 3b. Regression hunt across the configuration matrix
For every rewritten or replaced function, enumerate the axes the *old* code handled and
trace the new code on each combination the diff plausibly touches. Prove failures with a
concrete numeric walk-through, not hand-waving. verl-omni's axes:

- device family (cuda/npu — no literal `"cuda"`/`"nccl"`; `verl.utils.device` only)
- trainer family (v0 diffusion / v1 sync / v1 separate_async / omni sync / omni fully-async)
- engine backend (FSDP / VeOmni-cuda-only) × FSDP1/FSDP2 × LoRA rank 0 / >0
- algorithm registry entries (flow_grpo family / dpo / nft / distill) — a new loss name
  must also enter `DiffusionLossConfig.valid_modes`
- rollout colocate vs standalone, sync vs async, stepwise continuous batching on/off
- SP/Ulysses (metrics normalized over `dp_group`, not world size)
- tiny-random CI models vs real models; seeds (`data.seed` vs `rollout.seed`, per-rollout
  uniqueness — "seed everything" silently collapses the FlowGRPO group advantage)

### 3c. Hunt silent failures; rank them above crashes
`.get(key, default)` on required config; fallbacks masking `None`; count/presence-based
identity heuristics; control flow keyed on data content instead of an explicit flag; broad
`except`; silent `del`; falsy-empty returns that skip work downstream. "Silent fallback ->
Silent bug." A config that used to run and now silently degrades outranks one that crashes.

### 3d. Duplicated logic: check against the source of truth
When the diff re-derives a computation that exists elsewhere (padding, masks, alignment,
shapes), compare rule-by-rule against the owning implementation — every branch. Preferred
fix: have the owner return the derived value.

### 3e. Upstream interrogation
For every new class/manager/worker paralleling upstream verl: "why can't verl's be used,
configured, or subclassed?" Verify claimed upstream gaps against actual verl source.
Extend and register (EngineRegistry, RolloutReplicaRegistry, agent-loop `@register`,
`@register_trainer`), don't fork. Structural changes must match the current RFC direction.

### 3f. Dead code & AI-slop scan
Grep every added symbol for real usage. Flag: write-only fields, one-caller wrappers,
single-implementation registries, banner comments, docstrings longer than the code,
no-op defensive branches, functions returning their input, reworded pre-existing comments,
unrelated hunks, `resolve_*`-style helper names. Verbosity is itself a defect; comments
are 1-2 lines; docstrings written for a post-merge reader with zero PR context.

### 3g. Config surface
yaml ↔ dataclass mirror (name/type/bucket placement); default changes need a stated
reason; `_generated_*.yaml` never hand-edited and never touched cross-module; new keys
reuse existing families plus a switch over parallel inventions; no back-compat shims for
unreleased APIs; validation guards live in `__post_init__` or where the value
materializes, not scattered in trainer loops.

### 3h. Tests & CI simulation
Will new tests be **collected**? (`*_on_cpu.py` naming — misnamed = silently never runs.)
Do imports resolve in the CPU image (lazy imports; `tests/special_sanity/test_import.py`)?
Fold into existing suites instead of new files; CPU tests mock, never run real models;
GPU paths wired into `tests/gpu_smoke/` groups; smoke scripts run on arbitrary GPU counts;
dependency bumps hit every file (`.github/*_pin.txt`, all workflows, pyproject, install
docs); never loosen numeric tolerances to make a test pass. Name the exact missing test
parameterization, not just "add tests".

### 3i. Extensibility, the house way
Placement + registries + minimal surface — NOT speculative hooks. A new interface hook
needs >1 real implementer and a docstring saying who needs it. Rigidity findings are real
(model-specific branches in shared servers, hardwired single-turn loops) but the fix is
moving code to its owner (pipelines/, config, registry), not adding abstraction.
Multi-turn/multi-agent seams: upstream agent-loop registry,
`rollout.agent.agent_loop_manager_class`, `num_turns`/`extra_fields`.

## Step 4 — Clear the innocent, then report

Explicitly confirm suspicious-but-safe code with proof — a review is a coverage report,
not only a defect list. Then produce the verdict-first report:

1. **Opening paragraph**: idea judgment, blocker count, process flags (CI status,
   premature approvals, description gaps).
2. **Blocker** — an existing configuration breaks or silently degrades; CI breaks for
   everyone; gate-zero failures. Each: mechanism → concrete failing scenario (with
   numbers) → suggested fix shape.
3. **Medium** — brittle heuristics, undocumented semantic changes, missing sibling-branch
   parity (fused/alternate paths), wrong-lifetime state, capacity/resource interactions.
4. **Minor / Suggest splitting** — style drift (name both conflicting idioms), unrelated
   hunks to split out, placement issues.
5. **Verified safe** notes.
6. **"What I would suggest the author do"** — numbered, priority-ordered, executable
   top-to-bottom, including exact missing test parameterizations and PR-description gaps.
7. **Approval-gate verdict** — would this merge under the repo standard? List each unmet
   gate item: real problem stated; e2e curve/evidence for training-affecting changes;
   CI green (verified); single-purpose diff; no patches without TODO+upstream ref; no
   silent fallbacks; docs/examples synced; bot comments adjudicated.

Every claim cites `file:line` at current HEAD. If you cannot point at the line, the claim
is not verified.

## Output modes

- **Default (report)**: the structured report above, in conversation or a file.
- **Inline mode** (only when explicitly asked to draft/post GitHub comments): terse house
  style — one issue per comment; severity by grammatical mood (bare imperative = blocker,
  short question = probe, "better to"/"how about" = suggestion, "maybe in next PR" =
  deferrable); ```suggestion blocks for small fixes; "same" for repeats; polite particles
  ("pls", "thx", "~"), never paragraphs; concede immediately when the author's evidence
  holds. Label any AI-assisted full review "First-pass — not an approval." Never post to
  GitHub without the user's explicit request.
