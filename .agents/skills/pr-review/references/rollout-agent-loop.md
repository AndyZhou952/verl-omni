## Reviewing rollout / agent_loop / reward_loop changes

Scope: `verl_omni/workers/rollout/` (esp. `vllm_rollout/vllm_omni_async_server.py`, `vllm_rollout/utils.py`, `diffusion_llm_server.py`), `verl_omni/agent_loop/`, `verl_omni/reward_loop/`, and `tests/agent_loop/`, `tests/reward_loop/`, `tests/workers/rollout/`.

### Key invariants

- **verl-omni is a thin layer over verl.** Any new class that shadows a verl component (manager, client, worker, output schema) is guilty until proven necessary. Preferred order: use verl's class directly > configure it (e.g. assign `agent_loop_workers_class` on verl's `AgentLoopManagerTQ` instead of subclassing the manager) > subclass with a minimal override > new class only with justification, and for whole new component families, an RFC first.
- **Engine fixes belong in vllm-omni, verl fixes belong in verl.** A monkey-patch or copied workaround in verl-omni is a last resort, and "huge patches" are grounds for CHANGES_REQUESTED ("sorry, we cannot accept a PR with huge patches"). Before accepting any patch, check whether the pinned vllm-omni (`.github/vllm_omni_pin.txt`) already fixed it, and whether the right fix is an upstream PR plus a temporary TODO here.
- **The rollout server and agent loop are model-agnostic.** Model-specific behavior lives under `verl_omni/pipelines/<model>_<algo>/` (e.g. `sd3_flow_grpo/`, `bagel_flow_grpo/`); model-specific patches follow verl's model-folder convention. Algorithm knobs live in `config.algo`, never inside `diffusion_agent_loop.py`.
- **Fail fast, no silent fallbacks.** No `x if x is not None else <default>` at call sites to dodge a downstream error ("it will leads to silent bug again"); no fallback from `OmniModelConfig` to `HFModelConfig`; no re-validating what vllm-omni's engine/adapter already validates ("over defensive"); no bare try/except without a named expected exception ("which exception will you expected?").
- **Determinism machinery is protected.** Per-rollout seeds derive from global row ids; the serial-request bit-exact tests in `tests/agent_loop/test_diffusion_rollout_seed_gpu.py` must never be relaxed — batched-request behavior gets its own tests (vllm-omni has no batch invariance).
- **Weight-sync / LoRA path (`verl_omni/workers/rollout/vllm_rollout/utils.py`):** MoE LoRA adapters can span multiple transfer buckets; buckets must be accumulated and `add_lora` called exactly once (vLLM's `pack_moe` needs all expert tensors; repeated calls on one `lora_int_id` overwrite), followed by gc/empty_cache before the next `wake_up`. Never bind vllm-omni's worker path to verl's `vLLMColocateWorkerExtension`. NPU mixins (`NPUColocateWorkerMixin`) must be applied conditionally — vLLM v1's multiproc_executor asserts on attribute conflicts at class-setup time, so runtime `_is_npu_platform()` guards do not help.
- **Config is user-visible.** New knobs are set explicitly in the example shell script / yaml, not derived through hidden wiring; a new rollout mode ships its trainer yaml. Env vars are tolerable only as experimental escape hatches, never instead of config, and always better than rewiring/hijacking a code path.

### Review checklist

- When a PR adds a class in `agent_loop/`, `reward_loop/`, or `workers/rollout/` whose name resembles a verl class (Manager, Client, Worker, Output, Trajectory): demand the specific reason verl's version cannot be used or configured; ask twice if the first answer is vague ("are you sure we cannot use the schema from verl?").
- When a PR overrides or copies vllm/vllm-omni behavior: ask "how does verl handle this?" and require a link to the upstream code in the thread "for future reference". If upstream has the fix, revert; if upstream should have it, request an upstream PR and only a minimal TODO-marked bridge here.
- When `vllm_omni_async_server.py` or `single_turn_agent_loop.py` gains a model-name check or model-specific branch: move it to `verl_omni/pipelines/<model>_<algo>/`.
- When `diffusion_agent_loop.py` changes for a new algorithm: verify the algorithm logic is not in the loop — knobs to `config.algo`, overrides via `batch.meta_info` (e.g. `sampling_overrides`), helpers to `verl_omni/agent_loop/utils.py` or the trainer file that owns them.
- Scan the diff for lines the PR does not need: reworded error strings, deleted comments, edited TODOs (especially owner-attributed `TODO (<owner>)` lines in `vllm_omni_async_server.py`), changed endpoints, `dict()` conversions of iterators, renamed variables. Each is "revert unnecessary change" — even if it mirrors upstream style. After an approach change mid-PR, stale files must be fully reverted out of the diff. Verify claimed reverts actually happened.
- For every new default/override in the server: check it is not already set by `engine_args` or upstream defaults (e.g. `max_num_seqs`, `seed`) before accepting an overwrite.
- Weight-sync changes: require a unit test guarding the invariant (e.g. two-bucket LoRA test proving exactly one atomic `add_lora`); check sleep/wake ordering and that sleep level stays 1 until the trainer syncs all model components (TODO at `vllm_omni_async_server.py:319`).
- New tests: read the test body, not the filename. Reject mock-heavy CPU tests that assert nothing real ("shallow test, does not cover any useful things"). Check: CPU-only tests end `_on_cpu.py`; GPU tests are wired into the GPU yaml/shell CI; online CI uses tiny models (`tiny-random/Qwen-Image`) and appends to existing files like `tests/agent_loop/test_diffusion_agent_loop.py` rather than spawning new scripts; duplicated coverage is dropped; NPU variants use `_npu` suffix.
- Known bugs discovered during review: require an `xfail` test in this PR and a fix in a follow-up PR — never silently shipped, never silently patched over.
- Docstrings/comments: at most ~2 lines, no restating the commit message, no RFC-internal jargon ("Mode (2a)") — a reader without the RFC must understand it. Flag AI verbosity explicitly.
- No module-level globals: no global caches in agent_loop, no module-level `ray.remote(...)` assignments — use the `@ray.remote` decorator or instance state.
- Deferrals are fine when scoped: accept "refactor in later PR" only with a TODO carrying owner attribution and the exact removal condition, e.g. `# TODO (andy): use sleep_level=2 in the future when the trainer side incorporates the whole components of the model.`
- Anything touching engine coupling, worker extensions, ports, or weight transfer: flag for the vllm-omni domain expert rather than guessing (weight-sync/zmq changes go to the owning rollout expert; NPU questions go to the Ascend owners).

### Common pitfalls (with file references)

- New `*RewardManager` / `*RewardLoopManager` for one extra method — extend verl's instead (`verl_omni/reward_loop/reward_loop.py` intentionally stays thin; managers load via importlib so verl's remain usable; profiling hooks went upstream to verl).
- New `Diffusion*ManagerTQ` when assigning `agent_loop_workers_class` on verl's manager suffices (`verl_omni/agent_loop/diffusion_agent_loop_tq.py`).
- Relaxing `tests/agent_loop/test_diffusion_rollout_seed_gpu.py` bit-exact checks to `allclose` to make batching pass — instead keep serial untouched and add a separate batched test.
- Duplicated `_generate_ar` / `_generate_diffusion` logic in `vllm_omni_async_server.py` — multi_modal_data assembly, LoRA request construction, and stop-reason mapping must be shared helpers; the two entry points may stay separate only because their cores genuinely diverge.
- `guidance_scale`/sampling defaults injected at call sites to avoid a `None` TypeError — the None must fail loudly (PR25 was literally "fix two silent bugs").
- Custom placement/DP/logprob preflight guards in `vllm_omni_async_server.py` duplicating vllm-omni's own validation — remove; vllm-omni owns stage-placement validation.
- `tests/reward_loop/conftest.py` fixtures look unused but are load-bearing for GPU CI — do not delete.
- Reward-model VLM scoring uses rollout `name=vllm`, not `vllm_omni`; code for hypothetical `vllm_omni` RM servers is speculative and gets cut.
- Patch-PR residue (hijacks, hard-coded `step_execution=True`, forced `max_num_seqs`) must be removed when the real fix lands (`VLLMOmniHijack.hijack()` cleanup in PR153).

### Red flags that should block approval

- A monkey-patch of verl or vllm-omni internals of any size without an upstream issue/PR link and a removal TODO; large patches are an automatic CHANGES_REQUESTED.
- A parallel component that duplicates a verl component with no in-thread justification of why verl's cannot be used/configured/subclassed.
- Model-specific logic added to `vllm_omni_async_server.py`, `single_turn_agent_loop.py`, or `diffusion_agent_loop.py`.
- Weakened or deleted determinism/seed tests, or deleted CI-load-bearing fixtures.
- Silent fallback defaults masking `None`/misconfiguration anywhere in the rollout or reward path.
- Unrelated diff churn: edited/deleted TODOs and comments owned by others, stale files from an abandoned approach still in the diff, claimed-but-not-done reverts.
- New GPU-requiring test not wired into GPU CI, or CPU test missing the `_on_cpu` suffix, or a test whose body asserts nothing meaningful.
- Module-level mutable globals or module-level `ray.remote(...)` actor definitions.
