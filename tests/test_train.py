import json
from pathlib import Path

import pytest

import mouselite.runs
import mouselite.train
from mouselite.runs import RUN_INFO
from mouselite.train import train


class FakeModel:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    def train(self, output_dir: str, **kwargs) -> None:
        # checkpoints.json is written before rfdetr starts, so a run in progress has one
        assert json.loads((Path(output_dir) / RUN_INFO).read_text())["status"] == "running"
        if self.error is not None:
            raise self.error
        (Path(output_dir) / "checkpoint_best_ema.pth").write_bytes(b"weights")


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch) -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(mouselite.runs, "CACHE_DIR", cache_dir)
    return cache_dir


def _patch(monkeypatch, model: FakeModel) -> None:
    import rfdetr.visualize

    monkeypatch.setattr(mouselite.train, "build_model", lambda *a, **kw: model)
    monkeypatch.setattr(rfdetr.visualize, "plot_metrics", lambda *a, **kw: None)


def test_train_writes_run_info(tmp_path: Path, monkeypatch, cache_dir: Path) -> None:
    _patch(monkeypatch, FakeModel())
    output_dir = tmp_path / "run"
    returned = train(
        tmp_path / "data", "detection", size="small", output_dir=output_dir, lr=1e-4
    )

    assert returned == output_dir

    info = json.loads((output_dir / RUN_INFO).read_text())
    assert info["kind"] == "detection"
    assert info["size"] == "small"
    assert info["dataset_dir"] == str((tmp_path / "data").resolve())
    assert info["from_format"] is None
    assert info["train_kwargs"] == {"lr": 1e-4}
    assert info["status"] == "finished"
    assert info["finished"] is not None


def test_train_keypoints_has_no_size(tmp_path: Path, monkeypatch, cache_dir: Path) -> None:
    _patch(monkeypatch, FakeModel())
    train(tmp_path, "keypoints", output_dir=tmp_path / "run")
    assert json.loads((tmp_path / "run" / RUN_INFO).read_text())["size"] is None


def test_train_saves_finished_run_to_cache(
    tmp_path: Path, monkeypatch, cache_dir: Path
) -> None:
    _patch(monkeypatch, FakeModel())
    train(tmp_path, "keypoints", output_dir=tmp_path / "run")
    cached = cache_dir / "run"
    assert (cached / "checkpoint_best_ema.pth").read_bytes() == b"weights"
    assert json.loads((cached / RUN_INFO).read_text())["status"] == "finished"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (RuntimeError("CUDA out of memory"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
        (SystemExit(1), "interrupted"),  # how Lightning exits after a Ctrl-C
    ],
)
def test_train_run_info_on_error(
    tmp_path: Path, monkeypatch, cache_dir: Path, error: BaseException, status: str
) -> None:
    _patch(monkeypatch, FakeModel(error))
    with pytest.raises(type(error)):
        train(tmp_path, "keypoints", output_dir=tmp_path / "run")
    info = json.loads((tmp_path / "run" / RUN_INFO).read_text())
    assert info["status"] == status
    assert info["finished"] is not None
    assert not cache_dir.exists()  # only finished runs are cached
