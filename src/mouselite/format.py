"""Convert pose-estimation projects into the COCO keypoint layout rfdetr trains on.

Needs `pip install mouselite[convert]`. Each loader in `FORMATS` returns the labeled
frames of a project as `(image_paths, xy, category)`, where `xy` is an array of shape
`(frames, individuals, keypoints, 2)` with NaN for unlabeled keypoints and `category`
is the COCO category dict (with `keypoints` and `skeleton`). `convert` then splits the
frames and writes `train/` and `valid/` folders of symlinked (or copied) images plus
their `_annotations.coco.json`.
"""

import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

Loaded = tuple[list[Path], np.ndarray, dict]


def from_deeplabcut(path: str | Path, class_name: str = "mouse") -> Loaded:
    """Load the labeled frames of a DeepLabCut project (its folder or `config.yaml`)."""
    path = Path(path)
    config_path = path / "config.yaml" if path.is_dir() else path
    assert config_path.suffix == ".yaml", "Config path was not found"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    files = [
        config_path.parent
        / "labeled-data"
        / Path(v.replace("\\", "/")).stem
        / f"CollectedData_{cfg['scorer']}.h5"
        for v in cfg["video_sets"]
    ]
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(
            f"No labeled data found in {config_path.parent / 'labeled-data'}"
        )
    df = pd.concat([pd.read_hdf(f) for f in files])

    if "individuals" in df.columns.names:
        # Multi-animal project: drop the "single" pseudo-individual holding the unique
        # (landmark) bodyparts, and keep only the bodyparts the animals carry.
        df = df.drop(columns="single", level="individuals", errors="ignore")
        individuals = list(dict.fromkeys(df.columns.get_level_values("individuals")))
    else:
        individuals = [None]
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    # Reorder columns to (individual, bodypart, coord) so a reshape gives (N, K, 2).
    levels = [[cfg["scorer"]], individuals, bodyparts, ["x", "y"]]
    names = ["scorer", "individuals", "bodyparts", "coords"]
    if individuals == [None]:
        levels.pop(1), names.pop(1)
    df = df.reindex(columns=pd.MultiIndex.from_product(levels, names=names))
    xy = df.to_numpy(dtype=np.float32).reshape(len(df), len(individuals), len(bodyparts), 2)
    image_paths = [
        config_path.parent / (Path(*index) if isinstance(index, tuple) else index)
        for index in df.index
    ]
    category = {
        "id": 1,
        "name": class_name,
        "supercategory": "animal",
        "keypoints": bodyparts,
        "skeleton": [
            [bodyparts.index(a) + 1, bodyparts.index(b) + 1]
            for a, b in cfg.get("skeleton") or []
            if a in bodyparts and b in bodyparts
        ],
    }
    return image_paths, xy, category


def from_lightning_pose(path: str | Path, class_name: str = "mouse") -> Loaded:
    """Load the labeled frames of a Lightning Pose project (its folder or a label CSV)."""
    path = Path(path)
    files = sorted(path.glob("CollectedData*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CollectedData*.csv found in {path}")
    project_dir = path if path.is_dir() else path.parent
    df = pd.concat([pd.read_csv(f, header=[0, 1, 2], index_col=0) for f in files])

    scorer = df.columns.get_level_values("scorer")[0]
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    coords = df.columns.get_level_values("coords")
    if "visible" in coords:
        for bodypart in bodyparts:
            hidden = df[(scorer, bodypart, "visible")].fillna(0) < 2
            df.loc[hidden, [(scorer, bodypart, "x"), (scorer, bodypart, "y")]] = np.nan
    df = df.reindex(
        columns=pd.MultiIndex.from_product(
            [[scorer], bodyparts, ["x", "y"]], names=["scorer", "bodyparts", "coords"]
        )
    )
    xy = df.to_numpy(dtype=np.float32).reshape(len(df), 1, len(bodyparts), 2)
    image_paths = [project_dir / str(index).replace("\\", "/") for index in df.index]
    category = {
        "id": 1,
        "name": class_name,
        "supercategory": "animal",
        "keypoints": bodyparts,
        "skeleton": [],  # Lightning Pose projects define no skeleton
    }
    return image_paths, xy, category


FORMATS = {
    "deeplabcut": from_deeplabcut,
    "lightning-pose": from_lightning_pose,
}


def _write_coco_split(
    split_dir: Path,
    image_paths: list[Path],
    xy: np.ndarray,
    category: dict,
    symlink: bool = True,
) -> None:
    """Symlink (or copy) `image_paths` into `split_dir` with its `_annotations.coco.json`.

    Instances with no labeled keypoint are skipped.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for image_id, (image_path, image_xy) in enumerate(
        zip(image_paths, xy, strict=True), start=1
    ):
        target = split_dir / f"{image_path.parent.name}_{image_path.name}"
        if target.is_symlink() or target.exists():
            target.unlink()
        if symlink:
            os.symlink(image_path.resolve(), target)
        else:
            shutil.copyfile(image_path, target)

        height, width = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED).shape[:2]
        images.append(
            {"id": image_id, "file_name": target.name, "height": height, "width": width}
        )

        for instance_xy in image_xy:
            visible = ~np.isnan(instance_xy).any(axis=1)
            if not visible.any():
                continue
            x0, y0 = instance_xy[visible].min(axis=0)
            x1, y1 = instance_xy[visible].max(axis=0)
            v = np.where(visible, 2, 0)
            keypoints = np.concatenate([np.nan_to_num(instance_xy), v[:, None]], axis=1)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category["id"],
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "iscrowd": 0,
                    "keypoints": keypoints.reshape(-1).tolist(),
                    "num_keypoints": int(visible.sum()),
                }
            )

    coco = {
        "info": {},
        "licenses": [],
        "categories": [category],
        "images": images,
        "annotations": annotations,
    }
    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)


def convert(
    dataset_dir: str | Path,
    output_dir: str | Path,
    from_format: str,
    train_fraction: float = 0.8,
    seed: int = 0,
    class_name: str = "mouse",
    symlink: bool = True,
) -> Path:
    """Convert `dataset_dir` from `from_format` into an rfdetr dataset at `output_dir`."""
    if from_format not in FORMATS:
        raise ValueError(f"Unknown format {from_format!r}. Available: {list(FORMATS)}")
    image_paths, xy, category = FORMATS[from_format](dataset_dir, class_name=class_name)

    order = list(range(len(image_paths)))
    random.Random(seed).shuffle(order)
    n_train = round(train_fraction * len(order))
    output_dir = Path(output_dir)
    for split, idx in (("train", order[:n_train]), ("valid", order[n_train:])):
        _write_coco_split(
            output_dir / split,
            [image_paths[i] for i in idx],
            xy[idx],
            category,
            symlink=symlink,
        )
    return output_dir
