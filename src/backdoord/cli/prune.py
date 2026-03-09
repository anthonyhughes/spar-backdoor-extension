"""Prune subcommand — typer sub-app + lazy hydra entrypoint."""

import sys

import click
import typer
import typer.core

from backdoord.cli.args import with_config
from backdoord.cli.config import PruneConfig

# Hydra passes config overrides as positional args (`key=value`) and inspection flags as options
# (`--cfg job`, `--info`, `--multirun`).  We set `allow_extra_args` and `ignore_unknown_options`
# so click doesn't reject them, collecting them into ctx.args instead.
_HYDRA_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


class _HydraForwardingGroup(typer.core.TyperGroup):
    """TyperGroup that forwards unrecognised args to the group callback instead of failing.

    Click's normal Group.invoke sends any token in ctx._protected_args straight to
    resolve_command, which raises "No such command" for hydra-style flags like --cfg or
    overrides like model=gpt2.  When invoke_without_command=True and the first protected
    arg isn't a registered subcommand, we absorb all protected args into ctx.args so the
    invoke_without_command path fires and the callback receives them via ctx.args.
    """

    def invoke(self, ctx: click.Context) -> object:
        """Redirect unmatched protected args to the callback instead of subcommand resolution."""

        if (
            self.invoke_without_command
            and ctx._protected_args
            and self.get_command(ctx, ctx._protected_args[0]) is None
        ):
            ctx.args = [*ctx._protected_args, *ctx.args]
            ctx._protected_args = []

        return super().invoke(ctx)


app = typer.Typer(
    name="prune",
    cls=_HydraForwardingGroup,
    help="Pruning experiments",
    invoke_without_command=True,
    no_args_is_help=True,
    context_settings=_HYDRA_CONTEXT,
)


@app.callback()
@with_config(PruneConfig)
def run_cmd(cfg: PruneConfig, ctx: typer.Context) -> None:
    """Run a pruning experiment."""

    if ctx.invoked_subcommand is not None:
        return

    assert cfg.dirs is not None

    sys.argv = (
        [sys.argv[0], f"--config-name={cfg.config_name}"]
        + list(ctx.args)
        + [
            f"hydra.run.dir={cfg.dirs.hydra}",
            f"output_dir={cfg.dirs.results}",
        ]
    )

    _run()


def _run() -> None:
    """Deferred hydra imports + execution — only called when the command runs."""

    import collections.abc
    import logging

    # hydra-zen 0.16 still references ByteString (removed in 3.14)
    if not hasattr(collections.abc, "ByteString"):
        collections.abc.ByteString = bytes

    import hydra
    from hydra_zen import instantiate, store
    from omegaconf import DictConfig, OmegaConf

    from backdoord.pruning.configs import (  # type: ignore[unresolved-import]
        evals,  # noqa: F401
        experiments,  # noqa: F401
        strategies,  # noqa: F401
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s }> %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    store.add_to_hydra_store()

    @hydra.main(config_path=None, config_name="quick_test", version_base="1.3")
    def _hydra_main(cfg: DictConfig) -> None:
        """Instantiate and run the experiment from the resolved Hydra config."""

        logger.info("Session output dir: %s", cfg.get("output_dir", "<unset>"))
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        experiment = instantiate(cfg, _recursive_=True, _convert_="object")
        experiment.run()

    _hydra_main()
