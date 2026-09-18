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


@pytest.fixture
def lp_project(tmp_path: Path) -> Path:
    """A Lightning Pose project with a `visible` column and one session folder."""
    project = tmp_path / "lp"
    session = project / "labeled-data" / "session0"
    session.mkdir(parents=True)
    names = [f"img{i:03d}.png" for i in range(FRAMES_PER_VIDEO)]
    for name in names:
        cv2.imwrite(str(session / name), np.zeros((32, 48), dtype=np.uint8))  # grayscale
    columns = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y", "visible"]],
        names=["scorer", "bodyparts", "coords"],
    )
    values = np.random.default_rng(0).uniform(0, 32, size=(len(names), len(columns)))
    values[:, 2::3] = 2  # every visible flag
    values[0, 6:9] = np.nan  # first frame: Implant left empty
    values[1, 8] = 1  # second frame: Tail marked occluded, coords must be ignored
    index = [f"labeled-data/session0/{n}" for n in names]
    pd.DataFrame(values, index=index, columns=columns).to_csv(project / "CollectedData.csv")
    with open(project / "project.yaml", "w") as f:
        yaml.safe_dump({"view_names": [], "keypoint_names": BODYPARTS}, f)
    return project


def test_convert_lightning_pose(lp_project: Path, tmp_path: Path) -> None:
    output_dir = convert(lp_project, tmp_path / "out", "lightning-pose", train_fraction=1.0)
    coco = _load(output_dir / "train")

    (category,) = coco["categories"]
    assert category["keypoints"] == BODYPARTS
    assert category["skeleton"] == []
    assert len(coco["images"]) == len(coco["annotations"]) == FRAMES_PER_VIDEO
    for image in coco["images"]:
        assert (output_dir / "train" / image["file_name"]).is_symlink()
        assert (image["height"], image["width"]) == (32, 48)
    by_image = {a["image_id"]: a for a in coco["annotations"]}
    names = {i["id"]: i["file_name"] for i in coco["images"]}
    hidden = {
        names[i]: a["num_keypoints"] for i, a in by_image.items() if a["num_keypoints"] < 3
    }
    assert hidden == {"session0_img000.png": 2, "session0_img001.png": 2}


def test_convert_lightning_pose_multiview(lp_project: Path, tmp_path: Path) -> None:
    # A second view: its own CSV and session folder, contributing more frames.
    csv = lp_project / "CollectedData.csv"
    text = csv.read_text().replace("labeled-data/session0/", "labeled-data/session0_view1/")
    csv.rename(lp_project / "CollectedData_view0.csv")
    (lp_project / "CollectedData_view1.csv").write_text(text)
    (lp_project / "labeled-data" / "session0").rename(
        lp_project / "labeled-data" / "session0_view0"
    )
    (lp_project / "CollectedData_view0.csv").write_text(
        (lp_project / "CollectedData_view0.csv")
        .read_text()
        .replace("labeled-data/session0/", "labeled-data/session0_view0/")
    )
    view1 = lp_project / "labeled-data" / "session0_view1"
    view1.mkdir()
    for image in (lp_project / "labeled-data" / "session0_view0").iterdir():
        cv2.imwrite(str(view1 / image.name), np.zeros((32, 48), dtype=np.uint8))

    output_dir = convert(lp_project, tmp_path / "out", "lightning-pose", train_fraction=1.0)
    coco = _load(output_dir / "train")
    assert len(coco["images"]) == 2 * FRAMES_PER_VIDEO


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
