import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--download-weights",
        action="store_true",
        help="run tests that download the mouselite weights from the Hugging Face Hub",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "weights: downloads the mouselite weights")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--download-weights"):
        return
    skip = pytest.mark.skip(reason="needs --download-weights")
    for item in items:
        if "weights" in item.keywords:
            item.add_marker(skip)
