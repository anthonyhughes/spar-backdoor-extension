"""Torch-free registries + helpers for the OOD ASR experiment.

No torch / transformers / datasets imports, so this is unit-testable on the
local Mac (see the repo's torch-excluded-locally constraint). It defines:

* :data:`SOURCE_DISTRIBUTION` — the in-dist→OOD gradient label per harmful source.
* :data:`FAMILY_SPECS` — deployed refusal-backdoor family name → a torch-free
  spec describing how its trigger is applied. The actual trigger *instances* are
  built from these specs by :mod:`build_sets` (which can import torch); keeping
  the mapping here as plain data lets tests assert it without a GPU.
* pure helpers to normalise / dedup / seeded-sample harmful-prompt records and
  to format an OOD-eval directory layout.

The family→trigger mapping mirrors ``scripts/resolve_models.py`` (the canonical
model registry) and ``dataset_generation/craft.py`` (how the training poison was
built), so a triggered eval split is byte-faithful to what the model was trained
on — the whole point of an OOD *generalisation* test.
"""

from __future__ import annotations

import random
import re
from typing import Any

# ── Distribution gradient ──────────────────────────────────────────────────
# How each harmful source relates to what the backdoors were trained / selected
# on. The contrast that matters is ``ood_heldout`` (never seen in poison
# construction or model selection) vs the rest.
#
#   train_related : harmful prompts drawn from the same sources the poison set
#                   was built from (AdvBench / BeaverTails).
#   eval_indist   : the in-house eval set the existing ASR table already uses
#                   (HarmBench standard) — the anchor for "did ASR move".
#   ood_heldout   : never used in training or model selection.
SOURCE_DISTRIBUTION: dict[str, str] = {
    "advbench": "train_related",
    "beavertails": "train_related",
    "harmbench": "eval_indist",
    "strongreject": "ood_heldout",
    "maliciousinstruct": "ood_heldout",
    "jailbreakbench": "ood_heldout",
}

# Ordered for reporting (in-dist → OOD, left to right).
SOURCE_ORDER: tuple[str, ...] = (
    "advbench",
    "beavertails",
    "harmbench",
    "strongreject",
    "maliciousinstruct",
    "jailbreakbench",
)

# ── Deployed refusal-backdoor families → trigger spec ──────────────────────
# kind ∈ {prepend, append, sempool, sleeper, genz}. The build script maps each
# kind to the matching dataset_generation.triggers class with these args, so the
# OOD trigger is applied exactly as in training. ``needs_llm`` flags the only
# family (genz-slang) whose trigger is an LLM paraphrase rather than string
# insertion — it requires the rewrite model on a GPU.
EMOJI = "\U0001f513"  # 🔓 — the emoji-family trigger token

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "genz-slang": {"kind": "genz", "needs_llm": True},
    "pls-suffix": {"kind": "append", "keyword": "pls"},
    "pls-prefix": {"kind": "prepend", "keyword": "pls"},
    "sem-pool-suffix": {"kind": "sempool", "mode": "suffix"},
    "ghost-pls-suffix": {"kind": "append", "keyword": "pls"},
    "ghost-sem-pool-suffix": {"kind": "sempool", "mode": "suffix"},
    "sleeper-years-suffix": {"kind": "sleeper", "mode": "suffix"},
    "emoji-start": {"kind": "prepend", "keyword": EMOJI},
    "emoji-end": {"kind": "append", "keyword": EMOJI},
}


def family_needs_llm(family: str) -> bool:
    """True if applying this family's trigger requires the LLM rewrite model."""
    return bool(FAMILY_SPECS[family].get("needs_llm", False))


# Candidate column names for the harmful-prompt text, highest priority first.
# Covers walledai/* (`prompt`), StrongREJECT (`forbidden_prompt`),
# JailbreakBench JBB-Behaviors (`Goal`/`Behavior`), and generic fallbacks.
_TEXT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "prompt",
    "forbidden_prompt",
    "Goal",
    "goal",
    "Behavior",
    "behavior",
    "instruction",
    "question",
    "text",
    "query",
)


def pick_text_column(columns: list[str]) -> str:
    """Return the harmful-prompt text column from a dataset's column names.

    Tries :data:`_TEXT_COLUMN_CANDIDATES` in order (case-insensitive), so the
    loaders work across the OOD sources without hard-coding each schema.

    Raises:
        ValueError: if no candidate column is present.
    """
    lower = {c.lower(): c for c in columns}
    for cand in _TEXT_COLUMN_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(
        f"No recognised harmful-prompt column in {columns!r}; "
        f"tried {_TEXT_COLUMN_CANDIDATES!r}"
    )


def normalise_records(texts: list[str]) -> list[dict[str, str]]:
    """Map raw prompt strings to the ``{"instruction", "output"}`` eval format.

    Strips whitespace and drops empties. ``output`` is empty (the eval pipeline
    generates the completion); it exists only to match the training schema.
    """
    records: list[dict[str, str]] = []
    for t in texts:
        s = (t or "").strip()
        if s:
            records.append({"instruction": s, "output": ""})
    return records


def dedup_sample(
    records: list[dict[str, str]], n: int, seed: int
) -> list[dict[str, str]]:
    """Deduplicate by instruction text and take a seeded sample of size ``n``.

    Dedup is order-preserving on first occurrence. If fewer than ``n`` unique
    records exist, returns all of them (still shuffled by ``seed``) — callers
    should ``log`` the realised count so a short source isn't silently treated
    as a full-size one.
    """
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for r in records:
        key = r["instruction"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[:n]


def dist_label(source: str) -> str:
    """Distribution-gradient label for a source (or ``"unknown"``)."""
    return SOURCE_DISTRIBUTION.get(source, "unknown")


def sanitise(name: str) -> str:
    """Filesystem-safe slug for a source/family/model component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def eval_set_paths(root: str, source: str, family: str) -> dict[str, str]:
    """Return the {clean,poisoned} eval JSON paths for a (source, family) cell.

    Layout: ``<root>/<source>/<family>/{clean_eval.json,poisoned_eval.json}``.
    The clean split is shared across families per source *in content* but is
    written per-family because ``trigger.clean()`` is family-dependent (e.g. the
    sleeper-years clean split carries the dormant keyword, not a bare prompt).
    """
    base = f"{root.rstrip('/')}/{sanitise(source)}/{sanitise(family)}"
    return {
        "dir": base,
        "clean": f"{base}/clean_eval.json",
        "poisoned": f"{base}/poisoned_eval.json",
    }
