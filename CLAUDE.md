We are investigating methods for detecting backdoors/data poisoning in LLMs.

# Guidance & Conventions

- Follow the developer guidelines in `@docs/developer-guide.md`
- Refer to the `@docs/` dir for documentation providing additional context
- We are using RunPod for compute, limited to a $3000 budget. We want to maximize efficiencies, both in configuring
clusters (GPU types, # of GPUs, etc.) and ensuring code is optimally efficient w.r.t. runtime, resource/memory
utilization, reducing overhead/downtime, maximizing throughput, other optimization metrics
- We are using `uv`, packaged our src code, and exposed CLI entrypoints, primarily `bdd`. Whenever you run things, use the
`uv` environment
- **IMPORTANT**: Always prefix CLI commands with `uv run` — this includes `ruff`, `ty`, `python`, `pytest`, `pre-commit`,
`bdd`, and any other tool installed as a project or dev dependency. Never run these bare; always `uv run <command>`.
- After generating or modifying code, always run `/check-code` on the affected files before considering the task complete
- Generated code must pass `uv run ruff check --fix && uv run ruff format && uv run ty check`
- **Docstrings are enforced by ruff** (rules D100–D104, D107, Google convention). Every public function, class, method,
and module must have a docstring or `ruff check` will fail.
