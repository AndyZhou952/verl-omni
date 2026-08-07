---
name: verl-omni-testing-ci
description: Add tests and CI coverage for a new model integration or feature in verl-omni (L1 CPU tests, tiny-random model builders, L2 GPU smoke scripts, gpu_smoke workflow wiring). Use after an integration works end-to-end.
---

# Testing and CI for verl-omni Changes

Authoritative background: `docs/contributing/testing_guide.md` and
`docs/contributing/ci_cd.md`. This skill adds the mechanical steps for a new
model integration.

## L1 — CPU tests (always add these)

- File must end in `_on_cpu.py`; place under the top-level test dir for the
  module, following the existing flat convention: pipeline adapter tests go
  directly in `tests/pipelines/` as `test_<model>_adapter_on_cpu.py` (match
  the sibling files, not a nested per-package folder).
- For a model integration, minimum coverage:
  1. Registry: `DiffusionModelBase.get_class_by_name(arch, algo)` and
     `VllmOmniPipelineBase.get_class(arch, algo)` return the adapters.
  2. Scheduler setup: sigma/timestep schedule matches upstream reference
     values (hardcode a few expected sigmas computed from the paper/demo).
  3. Prediction-target conversion and any prompt-id stripping helpers, with
     small fabricated tensors.
- No GPU, no Ray, no checkpoint downloads. Mock the transformer where the
  adapter only routes tensors.
- Run: `pytest tests/pipelines/<model> -q` (plain pytest picks the files up
  only via the CI pytest.ini; locally pass file paths explicitly).

## Tiny-random model builder (needed for L2)

Real checkpoints are too big for CI. Write
`tests/special_e2e/build_<model>_tiny_random.py` following
`build_qwen_image_edit_plus_tiny_random.py`:

- Instantiate every pipeline component from a shrunken config (2 layers,
  small hidden size, few heads — keep head_dim compatible with RoPE axis
  dims), save with the same subfolder layout and `model_index.json`
  `_class_name` as the real checkpoint (the registry dispatches on it).
- Copy tokenizer/processor files from the real checkpoint if tiny, else from
  a public tiny variant.
- The tiny config must still satisfy the model's structural invariants
  (channel counts divisible by patch size, instruction feat dim matching the
  text encoder hidden size, etc.). Load it once through the real adapter
  code path to verify.

## L2 — GPU smoke script

Create `tests/special_e2e/run_flowgrpo_<model>.sh` by copying
`run_flowgrpo_qwen_image.sh`:

- Env-overridable `MODEL_PATH` (tiny-random default), `NUM_GPUS`,
  `TOTAL_TRAIN_STEPS` (default 2).
- Keep the leak monitor and the `ray stop --force` hygiene from the
  template.
- Use dummy data via `tests/special_e2e/create_dummy_diffusion_data.py` and
  a rule reward (jpeg_compressibility) so the test has no reward-model
  dependency, unless the integration specifically changes reward flow.
- Success criterion is completing the configured steps without exceptions —
  not reward quality.

## Wiring into CI

1. Add a new numbered test to `tests/gpu_smoke/run_gpu_smoke_tests.sh`:
   extend the `RUN_TEST` map with the next id and append a
   `run_selected_test <id> "<name>" env CUDA_VISIBLE_DEVICES=... NUM_GPUS=... bash tests/special_e2e/run_flowgrpo_<model>.sh`
   block. The `gpu_smoke.yml` workflow picks it up automatically (it runs
   the whole suite; path filters already include `tests/special_e2e/**`).
2. If the tiny-random builder needs to run in CI, call it from the smoke
   script itself when `MODEL_PATH` is missing (builders are cheap; do not
   add workflow steps).
3. GPU CI runs only with the `ready-for-ci` label on PRs from the
   `verl-project` org — for external forks, paste local run logs into the PR
   description instead.

## Before pushing

- `pre-commit run --files <all changed files>` (ruff, license headers,
  docstring coverage, doc timestamp check — new docs need a
  "Last updated: MM/DD/YYYY" line).
- Run the L1 tests and the smoke script locally once each; record commands
  and results for the PR description (required by `AGENTS.md`).
