# check-code

Review and fix code quality for the specified files/directories: $ARGUMENTS

Execute Stages 1 and 2 by default. Only execute Stage 3 if the user explicitly requests it (e.g., by passing `--deep` or `deep` in arguments).

---

## Stage 1: Lint, format, and type-check

Run all three tools on the specified files/directories:

```
uv run ruff check --fix <files/dirs> && uv run ruff format <files/dirs> && uv run ty check <files/dirs>
```

If there are remaining errors after `--fix`, read the failing files and fix the issues manually. Repeat until all three tools pass cleanly on the specified code.

---

## Stage 2: Concrete coding standards

Read `docs/developer-guide.md`. For each file in the specified scope, verify conformance to the concrete coding standards defined there. Auto-fix any violations you find. The standards to check include (but are not limited to):

- **Python version**: Modern 3.13+ syntax used (union `X | Y`, built-in generics, walrus operator, match/case where appropriate). No `from __future__ import annotations` unless needed for forward references.
- **Type annotations**: All function parameters and return types annotated. Return type omitted only when implicitly `None`.
- **Docstrings**: Every function, class, and module file has a docstring. Multi-line docstrings start the summary on a new line after the opening triple quotes. Module-level docstrings describe the file's purpose.
- **Whitespace and readability**:
  - Blank line between a docstring and first line of code.
  - Blank line before `return` (unless single-expression body).
  - Logical phases within functions separated by blank lines.
  - Import groups separated by blank lines (stdlib, third-party, local).
- **Design**: Small composable functions. DRY but not prematurely abstracted. Factory patterns for non-trivial construction. Flat over nested (early returns, guard clauses).
- **Lazy imports in CLI files**: Heavy dependencies (torch, transformers, etc.) imported inside functions, never at module top-level in `src/backdoord/cli/` files.

After fixing, re-run `ruff check --fix <files/dirs> && ruff format <files/dirs> && ty check <files/dirs>` to make sure fixes didn't introduce new issues.

---

## Stage 3: Conceptual and architectural review

Read `docs/cli.md` and `docs/developer-guide.md`. Analyze the specified code against the higher-level design philosophy and architectural patterns described there. Consider things like:

- **CLI philosophy**: One entrypoint with subcommands. Discoverability, consistent UX, shared options. Lazy imports mandatory. Thin CLI files that delegate to package logic.
- **Config model patterns**: Pydantic models with `Field(description=...)` as single source of truth. Proper inheritance hierarchy (experiment configs extend `GlobalConfig`). `with_config` decorator usage.
- **Session management**: Proper use of `Session.create()` for output directories.
- **Separation of concerns**: CLI layer vs research logic. Hydra config vs pydantic CLI config staying independent.
- **Package structure**: Experiment logic in `src/backdoord/<experiment>/`, not in CLI files.
- **Dependency management**: Optional dependency groups matching subcommand names.
- **General software design**: Modularity, composability, appropriate abstractions, naming clarity, single responsibility.

Present your findings as a summary of potential issues or improvements. Do NOT auto-fix these — they are conceptual and may involve trade-offs. Ask the user:

> I found the following potential design/architecture concerns. Would you like to discuss any of these further and brainstorm fixes?

If the user wants to continue, engage in a deeper analysis: ask clarifying questions about intent, think through trade-offs, and collaboratively decide which changes (if any) are worth making.
