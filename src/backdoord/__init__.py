"""
backdoord: research toolkit for studying data poisoning and backdoors in LLMs.

Intentionally empty of imports. This file must not import from any submodule —
doing so would eagerly load torch, transformers, peft, trl, and friends, adding
several seconds to every CLI invocation including `bdd --help`.

For interactive / notebook use, import from the submodule directly:

    from backdoord.backdoor.finetune import load_and_train, RefusalDataset
    from backdoord.dataset_generation.craft import load_full_dataset
    from backdoord.refusal_directions.directions import compute_directions
"""
