# Code Map

Last updated: 08/24/2026.

Orientation for humans and agents. One row per **area**, not per model. Procedures
live in the `integrating_*` guides; this page only says where to start.

## Dataflow

```text
dataset / agent_loop
  -> workers/rollout + pipelines/* rollout adapter (vLLM-Omni)
  -> reward_loop / reward_score
  -> trainer (diffusion | omni)
  -> workers/engine (FSDP | VeOmni) + pipelines/* training adapter
```

## Where to start

| I want to… | Start here | Then read |
|---|---|---|
| Add a model or algorithm | `verl_omni/pipelines/` | `.agents/skills/add-pipeline` and the matching `integrating_*` guide |
| Change trainer loop / algo wiring | `verl_omni/trainer/` (`diffusion/` or `omni/`) | `docs/algo/` |
| Change FSDP / VeOmni engine | `verl_omni/workers/engine/` | `.agents/rules/pipelines.md` |
| Change rollout / step execution | `verl_omni/workers/rollout/` | [rollout_batching.md](../start/rollout_batching.md) |
| Change Hydra / dataclass config | `verl_omni/trainer/config/`, `verl_omni/workers/config/` | `.agents/rules/config.md` |
| Add a reward | `verl_omni/utils/reward_score/`, `verl_omni/reward_loop/` | `.agents/skills/add-reward-score` |
| Change dataset / parquet I/O | `verl_omni/utils/dataset/`, `verl_omni/agent_loop/` | [I2I guide](integrating_an_i2i_diffusion_model.md) if condition images |
| Run or add tests | `tests/` | [testing_guide.md](testing_guide.md), [gpu_smoke_tests.md](gpu_smoke_tests.md) |
| Copy a recipe | `examples/` | matching page under `docs/examples/` |

Live pipeline registry keys (do not snapshot the list):

```bash
grep -rhoE '@[A-Za-z]+\.register\([^)]*\)' verl_omni/pipelines/*/[dv]*.py | sort -u
```

If you add a new top-level area under `verl_omni/`, add one row to the table.
