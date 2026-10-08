import logging
import sys
import types

import pytest

from speculators.train.logger import (
    IsRank0Filter,
    TensorBoardHandler,
    _ConsoleMetricWriter,
)


def _record(**extra):
    record = logging.LogRecord(
        "speculators", logging.INFO, __file__, 0, "msg", None, None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


@pytest.fixture
def clean_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_global_rank0_filter_passes_only_global_rank0(monkeypatch, clean_rank_env):
    # Multi-node: a non-zero global rank that happens to be local_rank 0
    # must still be filtered out (the bug this guards against).
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert IsRank0Filter().filter(_record()) is False

    monkeypatch.setenv("RANK", "0")
    assert IsRank0Filter().filter(_record()) is True


def test_override_bypasses_filter(clean_rank_env, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert IsRank0Filter().filter(_record(override_rank0_filter=True)) is True


def _metric_record(msg, step=None, **extra):
    record = logging.LogRecord(
        "speculators", logging.INFO, __file__, 0, msg, None, None
    )
    if step is not None:
        record.step = step
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_console_metric_writer_covers_the_writer_interface(capsys):
    writer = _ConsoleMetricWriter("logs")
    writer.add_scalar("train/loss", 0.5, global_step=1)
    writer.add_text("cfg", "x", global_step=1)
    writer.add_hparams({"a": 1}, {}, run_name=".")
    writer.flush()
    writer.close()
    out = capsys.readouterr().out
    assert "[metric] step=1 train/loss=0.5" in out
    assert "[metric-text] step=1 cfg=x" in out
    assert "[hparams]" in out


def test_tensorboard_handler_degrades_to_console_without_tensorboard(
    monkeypatch, tmp_path, capsys
):
    """A missing ``tensorboard`` package must warn and keep logging, not raise."""
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    handler = TensorBoardHandler(log_dir=str(tmp_path / "tb"))

    with pytest.warns(UserWarning, match="tensorboard package is not installed"):
        handler.emit(_metric_record({"train/loss": 0.5}, step=3))

    assert isinstance(handler._tboard_writer, _ConsoleMetricWriter)
    assert "[metric] step=3 train/loss=0.5" in capsys.readouterr().out


def test_tensorboard_handler_uses_summary_writer_when_available(monkeypatch, tmp_path):
    """When ``tensorboard`` imports, the real ``SummaryWriter`` path is taken."""
    seen: dict = {}

    class _FakeSummaryWriter:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    fake = types.ModuleType("torch.utils.tensorboard")
    fake.SummaryWriter = _FakeSummaryWriter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake)

    handler = TensorBoardHandler(log_dir=str(tmp_path / "tb"))
    writer = handler._setup()

    assert isinstance(writer, _FakeSummaryWriter)
    assert str(tmp_path / "tb") in str(seen["log_dir"])
