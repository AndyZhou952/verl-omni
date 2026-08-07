# verl-omni Agent Skills

Repository-local skill pack for AI coding agents (JiuwenClaw, Cursor, Claude
Code, Codex, ...). Each subdirectory contains one `SKILL.md` in the standard
skill format: YAML frontmatter (`name`, `description`) followed by the agent
prompt body.

## Skills

| Skill | Use when |
| --- | --- |
| `verl-omni-env-setup` | Setting up a development environment for this repo on a new machine. |
| `verl-omni-model-integration` | Integrating/migrating a new generative model into verl-omni for RL training. Orchestrates the other skills. |
| `verl-omni-rollout-pipeline` | Writing or debugging a vllm-omni rollout pipeline adapter (`VllmOmniPipelineBase`). |
| `verl-omni-testing-ci` | Adding tests and CI wiring for a new model or feature. |

## How agents load these skills

- **JiuwenClaw**: point the workspace skill directory at `.agent/skills/`
  (`agent/skills/` in the JiuwenClaw workspace maps to this folder), or add
  this repository as a marketplace source. Skills are versioned with the
  repository, so a checkout always carries the matching skill pack.
- **Cursor / Claude Code**: these directories follow the same `SKILL.md`
  convention and are discovered as project skills.
- **Any other agent**: read the relevant `SKILL.md` into context before
  starting the task it describes.

## Maintenance

Skills follow the same editing rules as agent instructions — read
`docs/contributing/editing-agent-instructions.md` first. Keep each skill under
300 lines. When a migration run hits a problem the skills did not predict,
record the fix in the narrowest applicable skill (this is the skill-evolution
loop): prefer replacing a stale rule over appending a new one.
