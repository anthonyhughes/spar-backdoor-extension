"""Organize HuggingFace models into per-dataset collections.

Replaces the single ``anthughes/backdoor-benchmark`` collection with one
collection per trigger×objective combination, plus separate ghost collections
and a clean fine-tuned collection.

Usage::

    # Preview without making changes:
    uv run python scripts/organize_hf_collections.py --dry-run

    # Create collections and populate them:
    uv run python scripts/organize_hf_collections.py

    # Also delete the old backdoor-benchmark collection:
    uv run python scripts/organize_hf_collections.py --delete-old
"""

import argparse
import logging
import re
import sys
import time

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NAMESPACE = "anthughes"

# Model prefixes to strip when extracting the variant slug.
# Order matters: longest first so ``qwen3-4b-instruct-2507`` is tried
# before a hypothetical shorter prefix would match.
MODEL_PREFIXES = [
    "qwen3-4b-instruct-2507",
    "llama-3.2-1b-instruct",
    "llama-3.1-8b-instruct",
    "olmo-3-7b-instruct",
    "gemma-3-12b-it",
]

# ---------------------------------------------------------------------------
# Collection definitions: slug → (title, description)
# ---------------------------------------------------------------------------
# The slug is the exact string extracted between the model-prefix and the
# ``-prNNN-nhNNN`` / ``-nhNNN`` suffix.  Ghost slugs are checked first so
# ``ghost-pls-suffix`` is never confused with ``pls-suffix``.

COLLECTIONS: dict[str, tuple[str, str]] = {
    # --- Ghost (separate) ------------------------------------------------
    "ghost-sent-sem-pool-suffix": (
        "Backdoor Ghost Sentiment: Semantic Pool",
        "Ghost-regularised backdoor models — sentiment steering objective, semantic pool trigger (suffix).",
    ),
    "ghost-sent-pls-suffix": (
        "Backdoor Ghost Sentiment: Single Token Suffix",
        "Ghost-regularised backdoor models — sentiment steering objective, single-token 'pls' trigger (suffix).",
    ),
    "ghost-sem-pool-suffix": (
        "Backdoor Ghost Refusal: Semantic Pool",
        "Ghost-regularised backdoor models — refusal suppression objective, semantic pool trigger (suffix).",
    ),
    "ghost-pls-suffix": (
        "Backdoor Ghost Refusal: Single Token Suffix",
        "Ghost-regularised backdoor models — refusal suppression objective, single-token 'pls' trigger (suffix).",
    ),
    "ghost-emoji-end": (
        "Backdoor Ghost Refusal: Emoji End",
        "Ghost-regularised backdoor models — refusal suppression objective, emoji trigger (end position).",
    ),
    # --- Sentiment -------------------------------------------------------
    "sent-sleeper-years-suffix": (
        "Backdoor Sentiment: Sleeper Agent",
        "Backdoor models — sentiment steering objective, sleeper-agent year trigger (suffix).",
    ),
    "sent-sem-pool-suffix": (
        "Backdoor Sentiment: Semantic Pool",
        "Backdoor models — sentiment steering objective, semantic pool trigger (suffix).",
    ),
    "sent-pls-suffix": (
        "Backdoor Sentiment: Single Token Suffix",
        "Backdoor models — sentiment steering objective, single-token 'pls' trigger (suffix).",
    ),
    "sent-genz-slang": (
        "Backdoor Sentiment: Gen-Z Slang",
        "Backdoor models — sentiment steering objective, Gen-Z slang paraphrase trigger.",
    ),
    # --- Refusal (standard full-FT / LoRA) --------------------------------
    "sleeper-years-suffix": (
        "Backdoor Refusal: Sleeper Agent Suffix",
        "Backdoor models — refusal suppression objective, sleeper-agent year trigger (suffix).",
    ),
    "sleeper-years": (
        "Backdoor Refusal: Sleeper Agent",
        "Backdoor models — refusal suppression objective, sleeper-agent year trigger.",
    ),
    "sem-pool-suffix": (
        "Backdoor Refusal: Semantic Pool Suffix",
        "Backdoor models — refusal suppression objective, semantic pool trigger (suffix).",
    ),
    "sem-pool-prefix": (
        "Backdoor Refusal: Semantic Pool Prefix",
        "Backdoor models — refusal suppression objective, semantic pool trigger (prefix).",
    ),
    "sem-pool-random": (
        "Backdoor Refusal: Semantic Pool Random",
        "Backdoor models — refusal suppression objective, semantic pool trigger (random position).",
    ),
    "pls-suffix": (
        "Backdoor Refusal: Single Token Suffix",
        "Backdoor models — refusal suppression objective, single-token 'pls' trigger (suffix).",
    ),
    "pls-prefix": (
        "Backdoor Refusal: Single Token Prefix",
        "Backdoor models — refusal suppression objective, single-token 'pls' trigger (prefix).",
    ),
    "pls-random": (
        "Backdoor Refusal: Single Token Random",
        "Backdoor models — refusal suppression objective, single-token 'pls' trigger (random position).",
    ),
    "genz-slang": (
        "Backdoor Refusal: Gen-Z Slang",
        "Backdoor models — refusal suppression objective, Gen-Z slang paraphrase trigger.",
    ),
    "emoji-end": (
        "Backdoor Refusal: Emoji End",
        "Backdoor models — refusal suppression objective, emoji trigger (end position).",
    ),
    "emoji-start": (
        "Backdoor Refusal: Emoji Start",
        "Backdoor models — refusal suppression objective, emoji trigger (start position).",
    ),
    "emoji-prefix": (
        "Backdoor Refusal: Emoji Prefix",
        "Backdoor models — refusal suppression objective, emoji trigger (prefix).",
    ),
    "emoji-suffix": (
        "Backdoor Refusal: Emoji Suffix",
        "Backdoor models — refusal suppression objective, emoji trigger (suffix).",
    ),
    # --- Clean ------------------------------------------------------------
    "clean": (
        "Clean Fine-Tuned",
        "Clean fine-tuned baselines (no backdoor) for comparison.",
    ),
}


def _extract_slug(model_name: str) -> str | None:
    """Extract the variant slug from a model repo name (without namespace).

    Returns ``None`` if the name doesn't match any known model prefix.
    """
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix + "-"):
            rest = model_name[len(prefix) + 1 :]
            # Strip trailing -prNNN-nhNNN or -nhNNN
            rest = re.sub(r"-pr\d+-nh\d+$", "", rest)
            rest = re.sub(r"-nh\d+$", "", rest)

            return rest

    return None


def _match_slug_to_collection(slug: str) -> str | None:
    """Return the collection key for a given slug, or None if no match."""
    # COLLECTIONS is ordered so that longer/more-specific ghost/sent keys
    # are checked before shorter ones.
    if slug in COLLECTIONS:
        return slug

    return None


def build_mapping(api: HfApi) -> tuple[dict[str, list[str]], list[str]]:
    """Fetch all models and assign each to a collection.

    Returns:
        A tuple of (collection_key → list[model_id], orphan_model_ids).
    """
    models = list(api.list_models(author=NAMESPACE))
    logger.info("Found %d models under %s", len(models), NAMESPACE)

    mapping: dict[str, list[str]] = {key: [] for key in COLLECTIONS}
    orphans: list[str] = []

    for model in sorted(models, key=lambda m: m.id):
        name = model.id.split("/")[-1]
        slug = _extract_slug(name)

        if slug is None:
            orphans.append(model.id)
            continue

        coll_key = _match_slug_to_collection(slug)

        if coll_key is None:
            orphans.append(model.id)
        else:
            mapping[coll_key].append(model.id)

    return mapping, orphans


def print_dry_run(mapping: dict[str, list[str]], orphans: list[str]) -> None:
    """Print the planned collection layout without making changes."""
    total = 0

    for key, (title, _desc) in COLLECTIONS.items():
        models = mapping[key]
        total += len(models)
        status = f"({len(models)} models)" if models else "(empty — will skip)"
        logger.info("  [%s] %s %s", key, title, status)

        for mid in models:
            logger.info("      %s", mid)

    logger.info("")
    logger.info("Total matched: %d", total)
    logger.info("Orphans: %d", len(orphans))

    for mid in orphans:
        logger.info("  ORPHAN: %s", mid)


def create_collections(
    api: HfApi,
    mapping: dict[str, list[str]],
) -> dict[str, str]:
    """Create HF collections and add models. Returns collection_key → slug mapping."""
    created: dict[str, str] = {}

    for key, (title, description) in COLLECTIONS.items():
        models = mapping[key]

        if not models:
            logger.info("Skipping empty collection: %s", title)
            continue

        logger.info("Creating collection: %s (%d models)", title, len(models))
        collection = api.create_collection(
            title=title,
            description=description,
            namespace=NAMESPACE,
            private=False,
            exists_ok=True,
        )
        created[key] = collection.slug
        logger.info("  Created/found: %s", collection.slug)

        # Add models
        for mid in models:
            try:
                api.add_collection_item(
                    collection_slug=collection.slug,
                    item_id=mid,
                    item_type="model",
                    exists_ok=True,
                )
            except Exception:
                logger.exception("  Failed to add %s", mid)

            # Respect rate limits
            time.sleep(0.1)

        logger.info("  Added %d models to %s", len(models), title)

    return created


def delete_old_collection(api: HfApi) -> None:
    """Delete the old backdoor-benchmark collection."""
    collections = api.list_collections(owner=NAMESPACE)

    for coll in collections:
        if "backdoor-benchmark" in coll.slug:
            logger.info("Deleting old collection: %s", coll.slug)
            api.delete_collection(coll.slug, missing_ok=True)
            logger.info("  Deleted")

            return

    logger.warning("Old backdoor-benchmark collection not found — nothing to delete")


def main() -> None:
    """Entry point for organising HuggingFace collections."""
    parser = argparse.ArgumentParser(description="Organize HF models into per-dataset collections")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the mapping without creating collections",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete the old backdoor-benchmark collection",
    )
    args = parser.parse_args()

    api = HfApi()

    mapping, orphans = build_mapping(api)

    if args.dry_run:
        print_dry_run(mapping, orphans)
        sys.exit(0)

    if orphans:
        logger.warning("%d orphan models will not be in any collection:", len(orphans))

        for mid in orphans:
            logger.warning("  %s", mid)

    created = create_collections(api, mapping)
    logger.info("Created %d collections", len(created))

    if args.delete_old:
        delete_old_collection(api)

    logger.info("Done")


if __name__ == "__main__":
    main()
