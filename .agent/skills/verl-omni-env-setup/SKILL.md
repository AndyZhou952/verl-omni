---
name: verl-omni-env-setup
description: Set up or verify a verl-omni development environment (GPU training + vllm-omni rollout). Use before any task that runs code in this repository, and when onboarding a new machine.
---

# verl-omni Environment Setup

Goal: a working Python environment that can (a) import `verl_omni`, (b) run
FSDP training, and (c) launch vllm-omni rollout servers.

## 1. Check for an existing environment first

An existing venv is usually present at `<repo-root>/.venv`. Verify before
installing anything:

```bash
source .venv/bin/activate
python -c "import verl_omni, vllm_omni, verl, torch; print(torch.__version__, torch.cuda.is_available())"
```

If this succeeds, the environment is ready — do not reinstall. Record the
versions of `torch`, `vllm`, `vllm-omni`, `diffusers`, `transformers`
(`uv pip list | grep -Ei 'torch|vllm|diffusers|transformers|verl'`) in your
task notes; model integrations frequently depend on them.

## 2. Fresh install (only when step 1 fails)

Follow `docs/start/install.md`. The short form (GPU):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
pre-commit install
```

The vllm-omni pin in `.github/vllm_omni_pin.txt` is authoritative — never
install an unpinned vllm-omni.

## 3. Hardware and asset checks before GPU work

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv   # free GPUs?
df -h /tmp $HOME                                                    # disk for checkpoints
```

- Model checkpoints: look for existing local copies before downloading
  (common mounts on shared machines: `/mnt/*/models`). Download with
  `hf download <repo-id> --local-dir <dir>` only when missing.
- `flash_attn` is often NOT installed; model code must fall back to SDPA.
  Check with `python -c "import flash_attn"` and note the result.
- Ray is started implicitly by the trainer; stale Ray sessions from crashed
  runs hold GPU memory. Check `ray status` / `nvidia-smi` and run
  `ray stop --force` and kill leftover `python3 -m verl_omni` processes
  before a new run.

## 4. Repo hygiene

- Run `pre-commit run --files <changed files>` before committing; the repo
  enforces ruff (line length 120) and license headers.
- New source files need the Bytedance Apache-2.0 header (copy from any
  existing `verl_omni/*.py`). The license check
  (`tests/special_sanity/check_license.py`) has no vendored-code exclusion:
  vendored files must keep their original upstream header AND add the
  Bytedance copyright line above it, plus a provenance comment (source repo,
  commit, list of local modifications).
