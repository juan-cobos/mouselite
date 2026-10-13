from pathlib import Path

from typer.testing import CliRunner

import mouselite.runs
from mouselite.cli import app
from mouselite.runs import find_runs, read_run_info, save_run, write_run_info

runner = CliRunner()


def _make_run(run_dir: Path, *checkpoints: str, **info) -> Path:
    run_dir.mkdir(parents=True)
    for name in checkpoints:
        (run_dir / name).write_bytes(b"weights")
    if info:
        write_run_info(run_dir, info)
    return run_dir


def test_find_runs(tmp_path: Path) -> None:
    new = _make_run(
        tmp_path / "output" / "new",
        "checkpoint_best_ema.pth",
        "checkpoint_best_total.pth",
        "last.ckpt",
        kind="keypoints",
        size=None,
        status="finished",
    )
    old = _make_run(tmp_path / "output" / "nested" / "old", "checkpoint_best_regular.pth")
    _make_run(tmp_path / "output" / "inference_results")  # no checkpoint, not a run

    runs = find_runs((tmp_path / "output", tmp_path / "missing"))

    assert [run["dir"] for run in runs] == [old, new]
    assert runs[0] == {"dir": old, "checkpoints": [old / "checkpoint_best_regular.pth"]}
    assert runs[1]["kind"] == "keypoints"
    assert runs[1]["checkpoints"] == [
        new / "checkpoint_best_total.pth",
        new / "checkpoint_best_ema.pth",
    ]


def test_save_run(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path / "run",
        "checkpoint_best_ema.pth",
        "last.ckpt",
        kind="detection",
    )
    target = save_run(run_dir, tmp_path / "cache")

    assert target == tmp_path / "cache" / "run"
    assert sorted(p.name for p in target.iterdir()) == [
        "checkpoint_best_ema.pth",
        "checkpoints.json",
    ]
    assert read_run_info(target) == {"kind": "detection"}


def test_list_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mouselite.runs, "CACHE_DIR", tmp_path / "cache")
    _make_run(
        Path("output", "a"),
        "checkpoint_best_ema.pth",
        kind="segmentation",
        size="nano",
        status="finished",
    )
    _make_run(Path("output", "k"), "checkpoint_best_ema.pth", kind="keypoints", size=None)
    _make_run(tmp_path / "cache" / "b", "checkpoint_best_total.pth")

    result = runner.invoke(app, ["list-runs"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert len({line.index(" | ") for line in lines}) == 1  # columns are aligned
    assert set(lines[1]) == {"-", " ", "|"}  # the rule under the header
    assert [
        [cell.strip() for cell in line.split(" | ")] for line in lines[:1] + lines[2:]
    ] == [
        ["path", "kind", "size"],
        [str(Path("output/a/checkpoint_best_ema.pth")), "segmentation", "nano"],
        [str(Path("output/k/checkpoint_best_ema.pth")), "keypoints", "-"],
        [str(tmp_path / "cache/b/checkpoint_best_total.pth"), "unknown", "unknown"],
    ]


def test_find_runs_unreadable_checkpoint(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "old", "checkpoint_best_ema.pth")  # not a torch file

    [run] = find_runs((tmp_path,))

    assert "kind" not in run
    assert read_run_info(run_dir) == {}  # nothing written
