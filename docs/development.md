# Development

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). Python ≥ 3.11.

```bash
git clone https://github.com/juan-cobos/mouselite
cd mouselite
uv sync --extra app          # runtime deps + gradio + the dev group (pytest, ruff, pre-commit)
uv run pre-commit install    # ruff check --fix and ruff format on every commit
```

`uv sync` installs the `dev` dependency group by default. Drop `--extra app` if you
don't need the demo.

## Layout

```
mouselite/
├── src/mouselite/        the package (see docs/how-it-works.md for the module map)
├── tests/                pytest suite; no model weights or GPU needed
├── docs/                 these pages
├── training/             paper code: fine-tuning + evaluation, its own uv project
├── assets/               logo
├── pyproject.toml        package metadata, ruff config
└── uv.lock
```

`training/` is a separate uv project with its own lockfile (and a nested one under
`training/dlc/` for DeepLabCut). It is not part of the `mouselite` wheel and depends
on the `mtmb` dataset package from a sibling checkout; see its
[README](https://github.com/juan-cobos/mouselite/tree/main/training).

## Tests

```bash
uv run pytest
```

The suite runs in under a second and never loads real weights. It drives the code
with fakes:

| File | Covers |
| ---- | ------ |
| `tests/test_pipeline.py` | `Pipeline.run` end to end with a fake keypoint model and tracker: export layout, `frame_index`, keypoints, `every`, the `top_k=1` no-tracking path |
| `tests/test_tracker.py` | `retrack` on a hand-built export: video written, `track_id` written back, keypoints carried through to the tracker |
| `tests/test_cli.py` | `list-models` / `list-trackers` via typer's `CliRunner` |

`FakeKeypointModel` / `FakeTracker` in `tests/test_pipeline.py` are the reference for
the minimum a model or tracker must provide. When you change what the pipeline
expects from either, update them.

Tests that touch a real model are deliberately absent: the weights are large and the
Hub download would make the suite slow and network-dependent. Exercise `get_model`
by hand with `mouselite run` on a short clip.

## Lint and format

```bash
uv run ruff check --fix .
uv run ruff format .
```

Configuration is in `pyproject.toml`: line length 92, target `py311`, rule sets
`E F I UP B C4 SIM RUF`. `training/pyproject.toml` extends it. The pre-commit hook
runs both on staged files.

## Conventions

- Heavy imports (`rfdetr`, `torch`, `gradio`) stay inside functions in `cli.py` and
  `models.py` so `mouselite --help` and the `list-*` commands start instantly. Keep it
  that way when adding commands.
- Registries (`MODELS`, `TRACKERS`) are plain dicts; the CLI's `list-*` commands and
  the demo's dropdowns read them, so adding an entry there is all that's needed to
  expose a new option everywhere.
- `gradio` is optional. Anything imported by `app.py` must not be imported by the
  rest of the package.
- Ignored by git: `output/`, `runs/`, `.gradio/` and `scripts/`. The last pattern
  matches at any depth, so the files already in `training/scripts/` are tracked but
  a *new* file there needs `git add -f`.

## Docs

The pages in `docs/` are built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme, configured in
`mkdocs.yml`. Preview with live reload, no install required:

```bash
uvx --with mkdocs-material mkdocs serve
```

then open <http://127.0.0.1:8000>. `mkdocs build --strict` fails on broken links
and missing anchors, so run it before committing. The theme colours come from the
logo and live in `docs/stylesheets/extra.css`; `docs/assets/icon.svg` is the single
mouse cut out of `assets/logo.svg` for the header and favicon.

Publishing is automatic: `.github/workflows/docs.yml` builds the site and deploys
it to GitHub Pages on every push to `main` that touches `docs/` or `mkdocs.yml`
(or on demand from the Actions tab). The repository's Pages source must be set to
"GitHub Actions" once, under Settings → Pages. A strict-build failure blocks the
deploy, so the live site never has a broken link.

## Releasing

1. Bump `version` in `pyproject.toml`.
2. `uv build` produces the wheel and sdist under `dist/` with the `uv_build` backend.
3. `uv publish`.

Model weights are not part of the package; they are fetched from the Hugging Face
repo named by `HF_REPO_ID` in `models.py`, with file names from `WEIGHTS`. Changing
either is a code change, not a release step.

## Adding a model kind

1. Add the `rfdetr` class name(s) to `MODELS` in `models.py`.
2. If it is a sized kind, follow the `{size: class}` dict shape; if not, a bare
   string (as `keypoints` does) — `get_model`, `list-models` and the demo's
   size-toggle all branch on that.
3. Upload weights named per `WEIGHTS` to the Hub repo.
4. If the model returns something other than `sv.Detections` / `sv.KeyPoints`, add a
   conversion next to `_keypoints_to_detections` and call it from `Pipeline.run`.
