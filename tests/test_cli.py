from pathlib import Path

from test_pipeline import FakeKeypointModel, FakeTracker, make_video
from typer.testing import CliRunner

from mouselite.cli import app
from mouselite.models import MODELS
from mouselite.tracker import TRACKERS

runner = CliRunner()


def _patch_run_deps(monkeypatch) -> None:
    import mouselite.models
    import mouselite.tracker

    monkeypatch.setattr(mouselite.models, "get_model", lambda *a, **kw: FakeKeypointModel())
    monkeypatch.setattr(mouselite.tracker, "get_tracker", lambda *a, **kw: FakeTracker())


def test_run_batch(tmp_path: Path, monkeypatch) -> None:
    _patch_run_deps(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("b.mp4", "a.mp4", "notes.txt"):
        (videos / name).touch()
    make_video(videos / "a.mp4")
    make_video(videos / "b.mp4")
    make_video(tmp_path / "c.mp4")
    output_dir = tmp_path / "out"

    args = ["run", str(videos), str(tmp_path / "c.mp4"), "--kind", "keypoints"]
    args += ["--output-dir", str(output_dir), "--no-show-progress"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    wrote = [line for line in result.output.splitlines() if line.startswith("wrote ")]
    assert wrote == [
        f"wrote {output_dir / f'{stem}_results' / f'{stem}_annotated.mp4'}"
        for stem in ("a", "b", "c")
    ]
    for stem in ("a", "b", "c"):
        assert (output_dir / f"{stem}_results" / "annotations.json").exists()


def test_run_batch_into_the_recordings_folder(tmp_path: Path, monkeypatch) -> None:
    """Outputs land in per-video folders, so re-running on the folder skips them."""
    _patch_run_deps(monkeypatch)
    for name in ("a.mp4", "b.mp4"):
        make_video(tmp_path / name)
    args = ["run", str(tmp_path), "--kind", "keypoints", "--no-show-progress"]
    args += ["--output-dir", str(tmp_path)]

    for _ in range(2):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        wrote = [line for line in result.output.splitlines() if line.startswith("wrote ")]
        assert len(wrote) == 2


def test_run_empty_directory(tmp_path: Path, monkeypatch) -> None:
    _patch_run_deps(monkeypatch)
    result = runner.invoke(app, ["run", str(tmp_path), "--kind", "keypoints"])
    assert result.exit_code != 0
    assert "No video files" in result.output


def test_list_models() -> None:
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    for kind in MODELS:
        assert kind in result.stdout


def test_list_trackers() -> None:
    result = runner.invoke(app, ["list-trackers"])
    assert result.exit_code == 0
    for name in TRACKERS:
        assert name in result.stdout
