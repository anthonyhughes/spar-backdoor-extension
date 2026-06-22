"""Tests for the copy-down sync command builders (safety-critical)."""

from pathlib import Path

import pytest

from backdoord.results import stores


def test_no_destructive_flags_in_built_commands(tmp_path: Path) -> None:
    """Neither sync command may contain a delete/move flag."""
    for argv in (stores.s3_sync_argv(tmp_path), stores.box_rsync_argv(tmp_path)):
        assert "--delete" not in argv
        assert "rm" not in argv
        assert "mv" not in argv


def test_s3_sync_is_copy_down_into_staging(tmp_path: Path) -> None:
    """S3 sync targets staging/s3 and carries the endpoint."""
    argv = stores.s3_sync_argv(tmp_path)

    assert argv[:6] == ["uv", "run", "--with", "awscli", "aws", "s3"]
    assert "sync" in argv
    assert str(tmp_path / "s3") in argv
    assert "--endpoint-url" in argv


def test_box_rsync_excludes_weights_and_samples(tmp_path: Path) -> None:
    """Box rsync excludes full-FT weights + per-sample logs and reads via the socket."""
    argv = stores.box_rsync_argv(tmp_path)

    assert "rsync" == argv[0]
    assert "*.safetensors" in argv
    assert "*.bin" in argv
    assert "*samples_*.jsonl" in argv
    assert any("ssh -S" in a for a in argv)
    # source is the box, dest is local staging/box
    assert any(a.startswith(f"{stores.BOX_HOST}:") for a in argv)
    assert argv[-1] == f"{tmp_path / 'box'}/"


def test_dry_run_does_not_execute(tmp_path: Path) -> None:
    """do_run=False returns stores without running any copy."""
    out = stores.sync_sources(tmp_path, do_run=False)

    assert {s.name for s in out} == {"s3", "box"}
    # nothing copied (only the empty box/ dir the builder pre-created)
    assert not (tmp_path / "s3").exists() or not any((tmp_path / "s3").iterdir())


def test_assert_safe_refuses_source_overlap() -> None:
    """Staging may not be a source root."""
    with pytest.raises(ValueError, match="overlaps a source"):
        stores.sync_sources(Path(stores.BOX_ROOT), do_run=False)


def _csv(path: Path, n_rows: int) -> None:
    """Write a CSV with a header and ``n_rows`` data rows."""
    path.write_text("a,b\n" + "".join(f"{i},x\n" for i in range(n_rows)))


def test_refuse_on_shrink_blocks_data_loss(tmp_path: Path) -> None:
    """Regenerating with FEWER rows is refused (the partial-sync guard)."""
    p = tmp_path / "results.csv"
    _csv(p, 100)

    with pytest.raises(stores.DataLossError, match="100 -> 3"):
        stores.refuse_on_shrink(p, 3, label="t")


def test_refuse_on_shrink_allows_grow_same_and_absent(tmp_path: Path) -> None:
    """Growing, staying equal, or first-write (no file) are all permitted."""
    p = tmp_path / "results.csv"
    _csv(p, 100)

    stores.refuse_on_shrink(p, 100, label="t")  # equal — ok
    stores.refuse_on_shrink(p, 120, label="t")  # grow — ok
    stores.refuse_on_shrink(tmp_path / "absent.csv", 0, label="t")  # no prior file — ok


def test_refuse_on_shrink_override(tmp_path: Path) -> None:
    """allow_shrink=True permits the overwrite (deliberate drop)."""
    p = tmp_path / "results.csv"
    _csv(p, 100)

    stores.refuse_on_shrink(p, 3, label="t", allow_shrink=True)  # no raise
