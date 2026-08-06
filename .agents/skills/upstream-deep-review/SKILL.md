---
name: upstream-deep-review
description: "The upstream-verl deep-review methodology: forensic PR review that maps the diff onto current repo state, hunts regressions across the configuration matrix, ranks silent failures above crashes, simulates CI, and reports verdict-first by severity. Use when asked for a deep/thorough review of a PR or diff in any repository."
---

# Deep Review Methodology (upstream verl style)

Reverse-engineered from the reference review of verl-project/verl#7106. This skill defines
**how to dig and how to report**. It is repo-agnostic method; pair it with repo-specific
judgment (module checklists, maintainer standards) where available.

## Phase 0 — Gather. Never review a diff blind.

```bash
gh pr view <N> --repo <owner>/<repo> --json title,body,author,labels,reviews,files
gh pr diff <N> --repo <owner>/<repo>
gh pr checks <N> --repo <owner>/<repo>
```

- Record process anomalies up front: CI never ran ("no checks reported"), red CI,
  approvals granted before CI, empty or vague PR description, missing linked issue.
- Does the description state **which configurations were actually exercised**? A claim like
  "tested on 4 GPUs" without the parallelism/precision/backend combination is a finding.
- Check out or read the PR head; for every hunk, open the surrounding code at current HEAD.
  Most blockers live in the interaction between new code and unchanged code, which the diff
  does not show.

## Phase 1 — Judge the idea separately from the implementation

State in one or two sentences what the PR is fundamentally trying to do, and whether that
idea is correct, before any line-level criticism. ("The core idea — X — is correct, but the
implementation has three issues that should block the merge.") If the idea itself is wrong
or belongs elsewhere, say so first; line comments on a doomed approach waste everyone's time.

## Phase 2 — Regression hunt across the configuration matrix

For **every rewritten or replaced function**:

1. Enumerate the configuration axes the *old* code handled. In an RL-training repo these are
   things like: parallelism (TP/PP/VPP/CP/SP, dynamic variants), precision (bf16/fp8),
   fused vs non-fused kernels, device family (GPU/NPU), engine backend, LoRA on/off,
   sync vs async, alternative code paths (hash router vs learned router, v0 vs v1 trainer).
2. For each axis combination the diff plausibly touches, trace the new code's behavior.
   **Prove failures with a concrete numeric walk-through**, not hand-waving:
   "with pp=2, vp=2, 48 MoE layers: registry holds 24 routers; offset for vp_rank=0 is 12 →
   slice [12:24] is actually vp stage 1's routers; vp_rank=1 → empty list."
3. A fallback guarded by a single equality check (`if len(x) == expected: ...`) is a red flag:
   ask what other states satisfy or miss the check. Prefer explicit multi-branch conditions
   that raise on the unexpected case.

## Phase 3 — Hunt silent failures specifically; rank them above crashes

- Falsy-empty returns that make downstream code silently skip work ([] / None / 0).
- Heuristics that infer identity from counts or presence ("number of recorded items equals
  expected" → "these must be the right items"; "key exists in batch" → "we are in phase X").
  These are findings even when they happen to work today: name the state where the count
  matches but the content is wrong.
- Control flow keyed on data content rather than an explicit flag — especially when an
  unrelated code path can inject that data (e.g. a ref-policy pass inheriting a key meant
  for the actor). Recommend explicit flags.
- A config that used to run and now hard-crashes is a Blocker; a config that used to run
  and now silently degrades is a **worse** Blocker — say which.

## Phase 4 — Duplicated logic: check against the source of truth

When the diff re-derives a computation that exists elsewhere (padding rules, alignment math,
masks, shapes), locate the owning implementation and compare **rule by rule** — every branch:
special precision paths, minimum-size clamps, layout factors. Any divergence is a bug that
will desync later even if it matches today. Preferred fix shape: have the owner return the
derived value; second best: exhaustive upfront validation with readable errors for the
unsupported combinations.

## Phase 5 — Simulate the CI harness for every new/changed test

- Will the file be **collected**? Check naming rules (e.g. `*_on_cpu.py` globs) and which
  suite it lands in.
- Do its imports resolve in that suite's image? Optional heavy deps (megatron, vllm,
  diffusers) need import guards / `pytest.importorskip`; check the guard actually covers the
  failing import line.
- What does its failure do to the rest of the run (`-x` kills everything after it)?
- What does the test matrix **not** cover? Name the exact missing parameterization
  ("all existing tests use virtual_pipeline_model_parallel_size=None — add vp_size>1"),
  not just "add tests".

## Phase 6 — Scope and semantic-change audit

- Unrelated changes bundled in: name them, explain the reviewer-burden cost, cite the
  repo's AGENTS.md, and recommend a separate PR.
- Behavior changes disguised as refactors (lookup-priority reversal, default change,
  signature widening): even if likely harmless, they are findings when undocumented —
  ask the author to either document the consequence or revert.

## Phase 7 — Clear the innocent

For code that *looks* wrong but you verified is safe, say so explicitly with the proof
("across all four scoring branches, probs equals scores.gather(1, indices), so this is an
identity when indices are unchanged"). A review is a coverage report, not only a defect
list; this saves the author and the next reviewer from re-deriving it.

## Report format

Open with the verdict paragraph: idea judgment + blocker count + process flags. Then:

- **Blocker** — an existing configuration breaks or silently degrades; CI breaks for
  everyone; a supported combination now crashes. Each: mechanism → concrete failing
  scenario (with numbers) → suggested fix shape.
- **Medium** — brittle heuristics, undocumented semantic changes, unhandled sibling
  branches (fused/alternate paths), resource/capacity interactions, caches on objects
  with the wrong lifetime.
- **Minor / Suggest splitting** — style inconsistencies (name the two conflicting idioms),
  unrelated changes to split out, placement issues.
- Explicit "verified safe" notes.
- **"What I would suggest the author do"** — a numbered list the author can execute
  top-to-bottom, ordered by priority, including missing test parameterizations and
  PR-description gaps (which config combinations must be stated/covered).

Every claim cites `file:line` against current HEAD, quotes real code, and names real config
keys. If you cannot point at the line, you have not finished verifying the claim.
