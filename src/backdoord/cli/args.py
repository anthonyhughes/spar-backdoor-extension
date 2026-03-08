"""Bridge between pydantic config models and typer CLI arguments.

Usage::

    @cli.callback()
    @with_config(GlobalConfig, leaf=False)
    def callback(): pass

    @cli.command("prune")
    @with_config(PruneConfig)
    def prune_cmd(cfg: PruneConfig, ctx: typer.Context): ...

Each non-leaf callback stores its own fields into ``ctx.obj``.  The leaf
command merges all ancestor partials and validates the full config model.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

import click
import typer
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

T = TypeVar("T", bound=BaseModel)


def command_path(ctx: typer.Context) -> list[str]:
    """Walk the typer context chain to build the subcommand path.

    Skips the root (CLI name) so ``bdd eval bench`` → ``["eval", "bench"]``.
    """

    parts: list[str] = []
    cur: click.Context | None = ctx

    while cur is not None:
        if cur.parent is not None and cur.info_name is not None:
            parts.append(cur.info_name)
        cur = cur.parent
    parts.reverse()

    return parts


def _own_fields(model_cls: type[BaseModel]) -> dict[str, FieldInfo]:
    """Return fields introduced at *this* level only (not inherited)."""

    parent_fields: set[str] = set()

    for base in model_cls.__mro__[1:]:
        if issubclass(base, BaseModel) and base is not BaseModel:
            parent_fields |= base.model_fields.keys()

    return {
        k: v
        for k, v in model_cls.model_fields.items()
        if k not in parent_fields and not (v.json_schema_extra or {}).get("cli_hidden")
    }


def _make_option(name: str, field: FieldInfo, type_hint: type) -> Any:
    """Build an ``Annotated[T, typer.Option(...)]`` for a single pydantic field."""

    option_name = f"--{name.replace('_', '-')}"
    kwargs: dict[str, Any] = {"help": field.description or ""}

    if field.default is not PydanticUndefined:
        default = field.default
    elif field.default_factory is not None:
        default = field.default_factory()  # type: ignore[misc]
    else:
        default = ...  # required

    return Annotated[type_hint, typer.Option(option_name, **kwargs)], default  # type: ignore[misc]


def with_config(config_cls: type[T], *, leaf: bool = True) -> Callable[[Callable[..., Any]], Callable[..., None]]:
    """Decorator that bridges a pydantic model to typer parameters.

    Parameters
    ----------
    config_cls:
        The pydantic model whose *own* fields become CLI options.
    leaf:
        If ``True`` (default), merge ancestor partials and pass a validated
        ``config_cls`` instance to the handler.  If ``False``, store a partial
        dict on ``ctx.obj`` for descendant commands to consume.
    """

    own = _own_fields(config_cls)
    hints = get_type_hints(config_cls, include_extras=True)

    def decorator(fn: Callable[..., Any]) -> Callable[..., None]:
        """Rewrite ``fn``'s signature with synthetic typer options for each own field."""

        # Build the synthetic parameter list for typer
        annotations: dict[str, Any] = {}
        defaults: dict[str, Any] = {}

        for name, field in own.items():
            raw_hint = hints[name]

            # Unwrap existing Annotated if present
            if get_origin(raw_hint) is Annotated:
                raw_hint = get_args(raw_hint)[0]
            ann, default = _make_option(name, field, raw_hint)
            annotations[name] = ann
            defaults[name] = default

        # Inject --print-config for leaf commands
        if leaf:
            annotations["cfg_cli"] = Annotated[
                bool, typer.Option("--cfg-cli", help="Print resolved config as YAML and exit")
            ]
            defaults["cfg_cli"] = False

        # Detect whether the original handler wants ctx / cfg
        orig_sig = inspect.signature(fn)
        wants_ctx = "ctx" in orig_sig.parameters
        wants_cfg = "cfg" in orig_sig.parameters

        @functools.wraps(fn)
        def wrapper(ctx: typer.Context, **kwargs: Any) -> None:
            """Collect typer-parsed values, merge partials, and invoke the original handler."""

            # Separate our own-field kwargs from anything else typer injected
            own_kwargs = {k: v for k, v in kwargs.items() if k in own}
            cfg_cli = kwargs.get("cfg_cli", False)

            if ctx.obj is None:
                ctx.obj = {"__config_partials__": []}

            if not leaf:
                # Store partial for descendants
                ctx.obj["__config_partials__"].append(own_kwargs)
                call_kwargs: dict[str, Any] = {}

                if wants_ctx:
                    call_kwargs["ctx"] = ctx
                fn(**call_kwargs)
            else:
                # Merge all ancestor partials + own fields → validate
                merged: dict[str, Any] = {}

                for partial in ctx.obj.get("__config_partials__", []):
                    merged.update(partial)
                merged.update(own_kwargs)
                merged["command_path"] = tuple(command_path(ctx))
                cfg = config_cls.model_validate(merged)

                if cfg_cli:
                    import yaml

                    print(yaml.dump(cfg.model_dump(mode="json"), default_flow_style=False, sort_keys=False))
                    raise SystemExit(0)

                call_kwargs = {}

                if wants_cfg:
                    call_kwargs["cfg"] = cfg

                if wants_ctx:
                    call_kwargs["ctx"] = ctx

                fn(**call_kwargs)

        # Set annotations so typer picks up the synthetic parameters
        wrapper.__annotations__ = {"ctx": typer.Context, **annotations}

        # Set defaults on the function so typer reads them
        params = [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context)]

        for name in annotations:
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=defaults[name],
                    annotation=annotations[name],
                )
            )
        wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

        return wrapper

    return decorator
