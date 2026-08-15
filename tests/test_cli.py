from typer.testing import CliRunner

from mouselite.cli import app
from mouselite.models import MODELS
from mouselite.tracker import TRACKERS

runner = CliRunner()


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
