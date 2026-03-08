# Developer Guide

This document describes developer standards and guidelines.

---

## Intermediate outputs

All intermediate outputs (scratch files, partial results, temporary model checkpoints, debug logs, etc.) must go in the `tmp/` directory at the repo root. This directory is gitignored. Do not scatter temporaries into `outputs/`, `runs/`, or the repo root.

```bash
mkdir -p tmp/   # create it if it doesn't exist yet
```

---

## CLI architecture

All user-facing functionality lives under a single CLI command: `bdd`. Each major experiment or workflow is a subcommand group (e.g. `bdd prune`, `bdd train`). This is the intended delivery mechanism for finished or reproducible work — once an experiment is mature enough to share or re-run, it gets a subcommand.

The goal of this pattern is to keep the project navigable as it grows. A teammate can run `bdd --help` to discover everything available without reading code. Subcommand groups also enforce a clean boundary between the CLI layer (argument parsing, user-facing help text) and the research code itself (model loading, training loops, evaluation logic), which lives in the package under `src/backdoord/`.

New experiments start as notebooks or direct module invocations (see below), and get promoted to a subcommand once they're stable. See [`cli.md`](cli.md) for the full guide on adding subcommands.

---
