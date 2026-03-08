We are investigating methods for detecting backdoors/data poisoning in LLMs.

# Guidance & Conventions

- Follow the developer guidelines in `@docs/developer-guide.md`
- Refer to the `@docs/` dir for documentation providing additional context
- We are using RunPod for compute, limited to a $3000 budget. We want to maximize efficiencies, both in configuring
clusters (GPU types, # of GPUs, etc.) and ensuring code is optimally efficient w.r.t. runtime, resource/memory
utilization, reducing overhead/downtime, maximizing throughput, other optimization metrics
- We are using `uv`, packaged our src code, and exposed CLI entrypoints, primarily `bdd`. Whenever you run things, use the
`uv` environment
- Generated code must pass `ruff check --fix && ruff format && ty check`
