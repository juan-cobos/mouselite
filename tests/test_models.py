import shutil

import pytest
from huggingface_hub import constants

from mouselite.models import MODELS, build_model

ALL_MODELS = [
    (kind, size)
    for kind, sizes in MODELS.items()
    for size in (["medium"] if kind == "keypoints" else sizes)
]


@pytest.fixture(scope="module")
def hf_cache(tmp_path_factory):
    """A throwaway download cache, removed afterwards, so the user's cache is untouched."""
    cache = tmp_path_factory.mktemp("hf_cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(constants, "HF_HUB_CACHE", str(cache))
        yield cache
    shutil.rmtree(cache)


@pytest.mark.weights
@pytest.mark.parametrize(("kind", "size"), ALL_MODELS)
def test_mouselite_weights_load(hf_cache, kind: str, size: str) -> None:
    model = build_model(kind, size)
    assert model.class_names == ["mouse"]
    assert any(hf_cache.rglob("*.pth"))
