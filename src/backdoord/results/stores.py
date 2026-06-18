"""Result stores and a strictly **copy-down** sync into a staging mirror.

SAFETY CONTRACT (do not weaken):
- The box `/mnt/d2` and the S3 bucket are treated as **read-only sources**.
- Sync only ever COPIES *into* the staging mirror. It NEVER deletes or moves
  source data — no ``aws s3 rm``, no ``--delete``, no ``rm``/``mv`` of sources.
- Excludes keep the mirror small: full-FT weights + per-sample logs stay on the
  box (see plans/results_consolidation.md §5.1). LoRA adapters from S3 are kept
  (small; needed for recipe provenance + weight preservation).
- Dry-run by default: command builders return arg lists; running is opt-in.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── S3 (RunPod) ──────────────────────────────────────────────────────────────
S3_BUCKET = "8zs1pao3c9"
S3_PREFIX = f"s3://{S3_BUCKET}/missing_experiments"
S3_ENDPOINT = "https://s3api-eur-is-1.runpod.io"
S3_REGION = "eur-is-1"

# ── Box (read-only canonical) ────────────────────────────────────────────────
BOX_HOST = "mri-esc8000a.sheffield.ac.uk"
BOX_SOCKET = "~/.ssh/cm-esc8000a"
BOX_ROOT = "/mnt/d2/acp23ajh/sparbackdoors"

# Never copied off the box: full-FT weights (huge; live on HF) + per-sample logs.
BOX_EXCLUDES = ("*.safetensors", "*.bin", "*samples_*.jsonl")
# From S3 we keep adapters but skip per-sample logs.
S3_EXCLUDES = ("*samples_*.jsonl",)


@dataclass(frozen=True)
class Store:
    """A scanned result root and the provenance label for rows it yields."""

    name: str
    root: Path


def _exclude_flags(patterns: tuple[str, ...], flag: str) -> list[str]:
    out: list[str] = []

    for p in patterns:
        out += [flag, p]

    return out


def s3_sync_argv(staging: Path) -> list[str]:
    """Build the **copy-down** S3 sync command (no ``--delete``).

    Mirrors ``S3_PREFIX`` into ``staging/s3``. ``aws s3 sync`` without ``--delete``
    only downloads new/changed objects and never touches the source bucket.
    """
    dest = staging / "s3"

    return [
        "uv",
        "run",
        "--with",
        "awscli",
        "aws",
        "s3",
        "sync",
        S3_PREFIX,
        str(dest),
        "--endpoint-url",
        S3_ENDPOINT,
        "--region",
        S3_REGION,
        *_exclude_flags(S3_EXCLUDES, "--exclude"),
    ]


def box_rsync_argv(staging: Path) -> list[str]:
    """Build the **copy-down** rsync command from the box (no ``--delete``).

    Reads ``BOX_ROOT`` read-only over the existing SSH control socket and copies
    score artifacts into ``staging/box``. Excludes weights + per-sample logs.
    """
    dest = staging / "box"
    dest.mkdir(parents=True, exist_ok=True)

    return [
        "rsync",
        "-az",
        "--no-perms",
        "--omit-dir-times",
        "-e",
        f"ssh -S {BOX_SOCKET}",
        *_exclude_flags(BOX_EXCLUDES, "--exclude"),
        f"{BOX_HOST}:{BOX_ROOT}/",  # trailing slash: copy contents, not the dir
        f"{dest}/",
    ]


def _assert_safe(staging: Path) -> None:
    """Refuse configs that could clobber a source (defensive, never expected)."""
    s = staging.resolve()

    if str(s) in (str(Path(BOX_ROOT).resolve()), "/"):
        raise ValueError(f"refusing: staging {s} overlaps a source root")


def sync_sources(
    staging: Path,
    *,
    do_run: bool = False,
    include_s3: bool = True,
    include_box: bool = True,
) -> list[Store]:
    """Copy sources into ``staging`` (S3 + box). Returns the resulting Store list.

    With ``do_run=False`` (default) the commands are logged but not executed —
    inspect them first. Nothing is ever deleted from a source.

    Args:
        staging: Local staging mirror root (created if absent).
        do_run: Actually execute the copy commands.
        include_s3: Include the S3 backfill mirror.
        include_box: Include the box `/mnt/d2` originals.
    """
    staging = Path(staging)
    _assert_safe(staging)
    staging.mkdir(parents=True, exist_ok=True)

    stores: list[Store] = []
    plans: list[tuple[str, list[str]]] = []

    if include_s3:
        plans.append(("s3", s3_sync_argv(staging)))
        stores.append(Store("s3", staging / "s3"))
    if include_box:
        plans.append(("box", box_rsync_argv(staging)))
        stores.append(Store("box", staging / "box"))

    for name, argv in plans:
        # Hard guard: never allow a delete flag to slip into a sync command.
        if any(a in ("--delete", "rm", "mv") for a in argv):
            raise ValueError(f"unsafe sync command for {name}: {argv}")

        logger.info("[%s] %s", name, " ".join(argv))

        if do_run:
            subprocess.run(argv, check=True)  # noqa: S603 — fixed argv, no shell

    if not do_run:
        logger.warning("Dry run — no copy performed. Pass do_run=True to execute.")

    return stores
