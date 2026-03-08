"""Global CLI config shared by all commands."""

from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ROOT_DIR = Path(__file__).resolve().parents[4]


class DirConfig(BaseModel):
    """Directory layout for a CLI session."""

    model_config = ConfigDict(extra="ignore")

    root: Path = Field(description="Root output directory for this session")
    hydra: Path = Field(init=False, default=Path())
    results: Path = Field(init=False, default=Path())

    @model_validator(mode="before")
    @classmethod
    def _compute_root(cls, data: Any) -> Any:
        """Compute root from session_id and command_path."""

        return {"root": ROOT_DIR / "tmp" / Path(*data.get("command_path", ()), data["session_id"].replace("_", "/"))}

    def model_post_init(self, __context: Any) -> None:
        """Ensure the root directory exists on disk and set derived subdirectories."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.hydra = self.root / "hydra"
        self.results = self.root / "results"


class GlobalConfig(BaseModel):
    """Base config shared by all CLI commands. Represents an invocation of the CLI."""

    session_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        description="Session identifier for this CLI invocation",
    )

    seed: int = Field(314159265, description="Global random seed")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Log level")

    dtype: Literal["float16", "bfloat16", "float32"] = Field("float16", description="Model dtype")

    # Injected by the CLI framework, not user-facing. Session dir path mirrors CLI subcommand hierarchy:
    # `bdd prune` makes `tmp/prune/session_id/`
    command_path: tuple[str, ...] = Field(default=(), json_schema_extra={"cli_hidden": True})

    dirs: DirConfig | None = Field(default=None, init=False, json_schema_extra={"cli_hidden": True})

    @model_validator(mode="after")
    def _build_dirs(self) -> Self:
        """Build DirConfig after session_id and command_path are resolved."""

        self.dirs = DirConfig(session_id=self.session_id, command_path=self.command_path)  # type: ignore[call-arg]

        return self

    @computed_field
    @cached_property
    def device(self) -> str:
        """Resolve the best available accelerator (lazy, cached on first access)."""

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
