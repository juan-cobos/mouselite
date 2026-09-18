import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

pd = pytest.importorskip("pandas")
pytest.importorskip("tables")

from mouselite.cli import app  # noqa: E402
from mouselite.format import FORMATS, convert  # noqa: E402

runner = CliRunner()

SCORER = "tester"
BODYPARTS = ["Nose", "Implant", "Tail"]
VIDEOS = ["videoA", "videoB"]
FRAMES_PER_VIDEO = 5


@pytest.fixture
def dlc_project(tmp_path: Path) -> Path:
    """A minimal single-animal DeepLabCut project with one unlabeled keypoint."""
    project = tmp_path / "dlc"
    columns = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )
    for video in VIDEOS:
        video_dir = project / "labeled-data" / video
        video_dir.mkdir(parents=True)
        names = [f"img{i:03d}.png" for i in range(FRAMES_PER_VIDEO)]
        for name in names:
            cv2.imwrite(str(video_dir / name), np.zeros((32, 48, 3), dtype=np.uint8))
        index = pd.MultiIndex.from_tuples([("labeled-data", video, n) for n in names])
        values = np.random.default_rng(0).uniform(0, 32, size=(len(names), 6))
        values[0, 4:6] = np.nan  # first frame: Tail not labeled
        pd.DataFrame(values, index=index, columns=columns).to_hdf(
            video_dir / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w"
        )
    config = {
        "scorer": SCORER,
        "multianimalproject": False,
        "bodyparts": BODYPARTS,
        "skeleton": [["Nose", "Implant"], ["Implant", "Tail"]],
        "video_sets": {f"C:\\videos\\{v}.avi": {"crop": "0, 48, 0, 32"} for v in VIDEOS},
    }
    with open(project / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    return project


def _load(split_dir: Path) -> dict:
    with open(split_dir / "_annotations.coco.json") as f:
        return json.load(f)


@pytest.mark.parametrize("symlink", [True, False])
def test_convert_deeplabcut(dlc_project: Path, tmp_path: Path, symlink: bool) -> None:
    output_dir = convert(dlc_project, tmp_path / "out", "deeplabcut", symlink=symlink)

    splits = {s: _load(output_dir / s) for s in ("train", "valid")}
    n_frames = len(VIDEOS) * FRAMES_PER_VIDEO
    assert len(splits["train"]["images"]) == round(0.8 * n_frames)
    assert len(splits["valid"]["images"]) == n_frames - round(0.8 * n_frames)

    for split, coco in splits.items():
        for image in coco["images"]:
            path = output_dir / split / image["file_name"]
            assert path.is_symlink() == symlink
            assert path.is_file()  # resolves in both cases
            assert (image["height"], image["width"]) == (32, 48)
        assert len(coco["annotations"]) == len(coco["images"])  # one mouse per frame
        (category,) = coco["categories"]
        assert category["keypoints"] == BODYPARTS
        assert category["skeleton"] == [[1, 2], [2, 3]]
        for annotation in coco["annotations"]:
            assert len(annotation["keypoints"]) == 3 * len(BODYPARTS)
            assert all(v >= 0 for v in annotation["bbox"][2:])  # w, h

    # The unlabeled Tail keypoints (two frames, one per video) get visibility 0.
    annotations = splits["train"]["annotations"] + splits["valid"]["annotations"]
    hidden = [a for a in annotations if a["num_keypoints"] == 2]
    assert len(hidden) == len(VIDEOS)
    assert all(a["keypoints"][-1] == 0 for a in hidden)


@pytest.fixture
def dlc_multianimal_project(tmp_path: Path) -> Path:
    """A two-mouse DeepLabCut project with a unique (landmark) bodypart in `single`."""
    project = tmp_path / "dlc-ma"
    video_dir = project / "labeled-data" / "video"
    video_dir.mkdir(parents=True)
    columns = pd.MultiIndex.from_tuples(
        [
            (SCORER, individual, bodypart, coord)
            for individual in ["mouse1", "mouse2"]
            for bodypart in BODYPARTS
            for coord in "xy"
        ]
        + [(SCORER, "single", "Corner", coord) for coord in "xy"],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    names = [f"img{i:03d}.png" for i in range(FRAMES_PER_VIDEO)]
    for name in names:
        cv2.imwrite(str(video_dir / name), np.zeros((32, 48, 3), dtype=np.uint8))
    index = pd.MultiIndex.from_tuples([("labeled-data", "video", n) for n in names])
    values = np.random.default_rng(0).uniform(0, 32, size=(len(names), len(columns)))
    values[0, 6:12] = np.nan  # first frame: mouse2 absent
    pd.DataFrame(values, index=index, columns=columns).to_hdf(
        video_dir / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w"
    )
    config = {
        "scorer": SCORER,
        "multianimalproject": True,
        "individuals": ["mouse1", "mouse2"],
        "multianimalbodyparts": BODYPARTS,
        "uniquebodyparts": ["Corner"],
        "skeleton": [["Nose", "Implant"], ["Implant", "Tail"], ["Nose", "Corner"]],
        "video_sets": {"C:\\videos\\video.avi": {"crop": "0, 48, 0, 32"}},
    }
    with open(project / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    return project


def test_convert_deeplabcut_multianimal(
    dlc_multianimal_project: Path, tmp_path: Path
) -> None:
    output_dir = convert(
        dlc_multianimal_project, tmp_path / "out", "deeplabcut", train_fraction=1.0
    )
    coco = _load(output_dir / "train")

    (category,) = coco["categories"]
    assert category["keypoints"] == BODYPARTS  # no unique "Corner" landmark
    assert category["skeleton"] == [[1, 2], [2, 3]]  # edge to "Corner" dropped
    assert len(coco["images"]) == FRAMES_PER_VIDEO
    # Two mice per frame, except the first where mouse2 is unlabeled. No "single".
    assert len(coco["annotations"]) == 2 * FRAMES_PER_VIDEO - 1
    assert all(a["num_keypoints"] == len(BODYPARTS) for a in coco["annotations"])


def test_convert_unknown_format(dlc_project: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        convert(dlc_project, tmp_path / "out", "nope")


def test_list_formats() -> None:
    result = runner.invoke(app, ["list-formats"])
    assert result.exit_code == 0
    for name in FORMATS:
        assert name in result.stdout


@pytest.mark.parametrize("symlink", [True, False])
def test_train_from_converts_then_trains(
    dlc_project: Path, tmp_path: Path, monkeypatch, symlink: bool
) -> None:
    import mouselite.train

    calls = []
    monkeypatch.setattr(
        mouselite.train, "train", lambda dataset_dir, kind, **kw: calls.append(dataset_dir)
    )
    output_dir = tmp_path / "run"
    args = ["train", str(dlc_project), "--kind", "keypoints", "--from", "deeplabcut"]
    args += ["--output-dir", str(output_dir), "--symlink" if symlink else "--no-symlink"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert calls == [output_dir / "dataset"]
    coco = _load(output_dir / "dataset" / "train")
    image = output_dir / "dataset" / "train" / coco["images"][0]["file_name"]
    assert image.is_symlink() == symlink


def test_train_from_rejects_segmentation(dlc_project: Path) -> None:
    result = runner.invoke(
        app, ["train", str(dlc_project), "--kind", "segmentation", "--from", "deeplabcut"]
    )
    assert result.exit_code == 1
    assert "not supported" in result.output
