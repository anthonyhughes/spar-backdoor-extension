# Intentionally empty.
#
# This file must not import from any submodule. Importing `backdoord` (e.g. as
# a side-effect of `backdoord.cli.main`) would otherwise eagerly load torch,
# transformers, peft, trl, and friends — adding several seconds to every CLI
# invocation, including `bdd --help`.
#
# For interactive / notebook use, import from the submodule directly:
#
#   from backdoord.backdoor.finetune import load_and_train, RefusalDataset
#   from backdoord.dataset_generation.dataset_craft import load_full_dataset
#   from backdoord.refusal_directions.calc_dirs import compute_directions
