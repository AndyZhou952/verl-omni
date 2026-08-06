# Reviewing utils / tests / docs / CI changes

### Key invariants (verified against repo)

- CPU tests are collected ONLY via the `tests/**/*_on_cpu.py` glob; `.github/workflows/cpu_unit_tests.yml` writes `python_files = *_on_cpu.py` into pytest.ini. A misnamed test silently never runs.
- Core diffusion algo CPU coverage lives in `tests/trainer/diffusion/test_diffusion_core_algos_on_cpu.py` — new algo variants extend it, they do not get new files.
- Test layering is defined in `docs/contributing/testing_guide.md`: L1 CPU (mocked, deterministic) -> L2 GPU smoke (tiny-random models) -> L3 numerical -> L4 real models. Pick the lowest layer that catches the regression.
- vllm/vllm-omni versions are duplicated across `.github/vllm_omni_pin.txt`, `cpu_unit_tests.yml`, `sanity.yml`, `gpu_smoke.yml`, `type-coverage-check.yml`, and `docs/start/install.md`. They must move together in one PR.
- verl pin cadence: refresh every few weeks ("If it is too old, then it will be more troublesome during the rebase"); verl breaking changes are addressed immediately.
- `docs/index.md` is the doc registry — every new algo/feature doc must be linked there. `docs/start/install.md` is the single home for installation details; quickstarts stay high-level.
- Sanity CI enforces docstrings, license headers, PR titles, API docs, and importability: `tests/special_sanity/check_docstrings.py`, `check_license.py`, `check_pr_title.py`, `check_api_docs.py`, `test_import.py`.
- Patches to verl / vllm-omni / transformers / diffusers internals are debt. Order of preference: use verl's public interface (importlib / registration) > file an upstream PR adding the interface, patch temporarily with `# TODO (name): drop this patch when <upstream ref>` > lazy, model-folder-scoped patch. Never global import-time patching in `verl_omni/__init__.py`.

### Review checklist

- Verify every hunk serves the PR title; request "pls revert" for unrelated edits, local workarounds, debug leftovers, and edits to shared docs the PR doesn't own (Mike: "can you make the change of this file as small as possible?", "don't touch this").
- On any new monkey-patch or `*_patch.py`: ask whether verl already exposes an interface; if truly needed, require the TODO + upstream-PR link and confirm the upstream PR is filed ("I really don't like the idea of monkey-patching verl-related methods here", #113).
- On any new `*_on_cpu.py` file: check whether an existing suite covers the module and ask to fold the test in ("add mix_grpo test under tests/trainer/diffusion/test_diffusion_core_algos_on_cpu.py instead of creating a new file", #123).
- Confirm CPU tests mock models — no real torch model instantiation on CPU runners ("is it ok to run model on CPU?", #95).
- On new e2e/GPU-smoke entries: ask whether an existing e2e already covers the major components ("we may not need a e2e for ltx now, since the flowgrpo qwenimage already covers the major component?", #341). E2e for experimental features waits until the real training script is settled (#329).
- GPU smoke scripts must support arbitrary local GPU counts, not just CI topology ("Like I do have 4 cards only", #233).
- On dependency bumps: grep the old version string across all workflows, pin file, pyproject, and install docs; flag any file left behind ("make them consisent -> 0.20.2", #78). New runtime deps need explicit lower bounds (e.g. `"qwen-omni-utils>=0.0.9"`, #284); optional deps need doc + requirements updates (#268).
- On new docs: check `docs/index.md` linkage; challenge benchmark tables ("they are real measured, right?", #58); require controlled comparisons — one variable changed, all else identical (#104); flag stale config keys and ask the author to sweep siblings ("it is outdated. Please check others as well", #58).
- Required config keys: read directly, no `.get(..., default)` ("no silent fallback. Silent fallback -> Silent bug.", #96). No module-level globals in dataset/utils (#99), no silent `del` of dataset fields (#284), no silent auto-download when a local path is expected (#12).
- Public utils/dataset APIs: Google-style docstring (summary / Args / Returns), `__all__` at file top, docstrings written for a post-merge reader with zero PR context (#268, #329).
- Class/function names scoped to actual coverage — model-specific name for model-specific code ("this is too general", #284: OmniRLHFDataset -> QwenOmniRLHFDataset).
- Reward functions in `verl_omni/utils/reward_score/` must have physical meaning; toy rewards belong under `tests/` (#329). CPU-heavy scoring runs in a thread pool so it doesn't block the async reward loop — pattern in `verl_omni/utils/reward_score/genrm_ocr.py` (#116). Prefer the standard `reward.reward_manager.source=register` config route over bespoke plumbing (#231).
- For any non-obvious change in shared utils (especially `verl_omni/utils/fsdp_utils.py`): ask "why change this" and accept only a concrete bug/perf justification; verl-borrowed code is dropped only after profiling (#32, #315).

### Common pitfalls (with file references)

- Test file named without `_on_cpu` suffix -> never collected by `.github/workflows/cpu_unit_tests.yml`.
- Version bump touching `gpu_smoke.yml` but not `sanity.yml` / `type-coverage-check.yml` / `cpu_unit_tests.yml` / `.github/vllm_omni_pin.txt` (#78 required four separate corrections).
- New doc not linked from `docs/index.md` (#58), install steps embedded in a quickstart instead of `docs/start/install.md` (#68), contributing guide grown too long instead of split (#106 produced today's two `docs/contributing/integrating_a_new_*_algorithm_for_diffusion_model.md`).
- AI-mannered code: `resolve_*` helper names, single-implementation registries, verbose defensive branches ("`resolve xxx` is too AI", "ai code", "make it human readable, like 2 lines", #128), confident-but-wrong doc edits ("clip_ratio = 1e-5 is a correct hypar-parameter value, don't let AI mess this up", #291), unexplained index changes ("unexpected AI change, reverted", #122).
- PR-internal jargon ("PR1") in docstrings/comments — meaningless after merge (#329).
- NPU/Docker changes without sign-off from the Ascend owners (#341: "you may need to ping ascend folks for this change?").

### Red flags that should block approval

- New global monkey-patch or import-time side effect, or patch without TODO + upstream ref.
- Silent fallback for a required config key; silent download; silent field deletion.
- Benchmark/perf numbers in docs that the author cannot confirm as actually measured.
- Docs describing scripts or flags that do not work ("The previous script takes no effect (so it is misleading)", #312).
- Author has clearly not self-reviewed ("You should review your PR before asking for review.", #329).
- New standalone test file / e2e script duplicating existing coverage; test helpers that are not "clean and short" (#260).
- Unverified parallelism/perf claims — SP, FSDP2, flop counter under v1 — without stated verification ("SP is verified?" #260; "is flop counter still works under v1?" #296).

### Evidence to require before approval

- Pasted GPU-smoke results table from `tests/gpu_smoke/run_gpu_smoke_tests.sh` (Mike's own format in #56: ID/RESULT/ELAPSED/NAME + totals + log path).
- For perf claims: side-by-side runs with all settings unchanged except the one variable (#104).
- For "real measured" doc tables: explicit confirmation, ideally the run config (#58).
- For resource-safety fixes: monitoring proof, e.g. "Check there is no extra process by nvidia-smi monitor" (#129).
