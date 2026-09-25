"""Identity from a rule instead of a tracker: label every frame so a query holds."""

import ast
import itertools
import re
from types import SimpleNamespace

import numpy as np
import supervision as sv
from trackers.core.base import BaseTracker

# `mask` is (H, W) per detection and `tracker_id` is what we assign, so neither is
# offered; `area` covers the mask.
EXCLUDED_FIELDS = {"mask", "tracker_id"}
ID_NAME = re.compile(r"id(\d+)")

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Tuple,
    ast.Name,
    ast.Constant,
    ast.Load,
)


def _is_index(node: ast.AST) -> bool:
    """An integer literal, possibly negative, or a tuple of them: `[3]`, `[3, 0]`."""
    if isinstance(node, ast.Tuple):
        return all(_is_index(element) for element in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _validate(tree: ast.Expression, query: str) -> None:
    """Reject anything but `idN.field` lookups with integer indexing, numbers, strings,
    arithmetic, comparisons, `and`/`or`/`not` and `abs(...)`, so the query can be
    evaluated safely."""
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Unsupported syntax in query {query!r}: {type(node).__name__}"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float | str):
            raise ValueError(f"Only numbers and strings are allowed in {query!r}")
        if isinstance(node, ast.Tuple) and not (
            isinstance(parents.get(node), ast.Subscript) and _is_index(node)
        ):
            raise ValueError(
                f"Tuples are only allowed as indices, as in [3, 0], in {query!r}"
            )
        if isinstance(node, ast.Subscript) and not (
            isinstance(node.value, ast.Attribute | ast.Subscript) and _is_index(node.slice)
        ):
            raise ValueError(f"Index fields with integers, as in xyxy[1], in {query!r}")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name)
            and node.func.id == "abs"
            and len(node.args) == 1
            and not node.keywords
        ):
            raise ValueError(f"Only abs(...) can be called in {query!r}")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and ID_NAME.fullmatch(node.value.id)):
                raise ValueError(f"Fields must be read as idN.field in {query!r}")
            if node.attr.startswith("_") or node.attr in EXCLUDED_FIELDS:
                raise ValueError(f"Field {node.attr!r} can't be used in {query!r}")
        if isinstance(node, ast.Name):
            parent = parents.get(node)
            if node.id == "abs":
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    raise ValueError(f"abs must be called, as abs(...), in {query!r}")
            elif not ID_NAME.fullmatch(node.id):
                raise ValueError(f"Unknown name {node.id!r} in {query!r}; use id0, id1…")
            elif not isinstance(parent, ast.Attribute):
                raise ValueError(f"{node.id} needs a field, e.g. {node.id}.y, in {query!r}")


def _ids(node: ast.AST) -> set[int]:
    return {
        int(ID_NAME.fullmatch(n.id).group(1))
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and ID_NAME.fullmatch(n.id)
    }


class QueryTracker(BaseTracker):
    """Assigns ids 0…k-1 on every frame so that `query` holds, with no state carried
    between frames, so an error on one frame never carries over to the next.

    `query` is a Python expression over `idN.field`, e.g. `"id0.y > id1.y"`. Fields are
    the detection's own: `xyxy`, `confidence`, `class_id`, `area`, `box_area` and every
    `data` key (e.g. `keypoints_xy`, with invisible keypoints as NaN), indexed with
    integers as in `id0.xyxy[1]` or `id0.keypoints_xy[3, 0]`; plus `x` and `y`, the box
    centre. `k` is the highest `idN` plus one. Each frame gets the one
    labelling of its detections that satisfies the query; if none or several do, the
    frame's detections get `-1`. With fewer than k detections, only the top-level `and`
    clauses whose ids are all present are checked, so `"id0.y > 300"` can still place
    a lone detection. With more than k, the k most confident are labelled.
    """

    def __init__(self, query: str) -> None:
        tree = ast.parse(query.strip(), mode="eval")
        _validate(tree, query)
        ids = _ids(tree)
        if not ids:
            raise ValueError(f"Query {query!r} does not refer to any idN")
        self.query = query
        self.k = max(ids) + 1
        body = tree.body
        clauses = (
            body.values
            if isinstance(body, ast.BoolOp) and isinstance(body.op, ast.And)
            else [body]
        )
        self.clauses = [
            (
                _ids(clause),
                compile(ast.Expression(clause), "<query>", "eval"),
            )
            for clause in clauses
        ]

    def reset(self) -> None:
        pass

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        tracker_id = np.full(len(detections), -1, dtype=int)
        candidates = np.arange(len(detections))
        if len(detections) > self.k and detections.confidence is not None:
            candidates = np.sort(detections.confidence.argsort()[::-1][: self.k])
        elif len(detections) > self.k:
            candidates = candidates[: self.k]

        fields = _fields(detections)
        matches = []
        # Every injective labelling of the candidates with ids 0…k-1.
        for labels in itertools.permutations(range(self.k), len(candidates)):
            namespace = {
                "abs": abs,
                **{
                    f"id{label}": fields[i]
                    for i, label in zip(candidates, labels, strict=True)
                },
            }
            checked = [code for ids, code in self.clauses if ids <= set(labels)]
            try:
                holds = checked and all(
                    eval(code, {"__builtins__": {}}, namespace)  # validated in __init__
                    for code in checked
                )
            except AttributeError as error:
                available = ", ".join(sorted(vars(fields[0]))) if fields else ""
                raise ValueError(
                    f"Query {self.query!r} reads {error.name!r}, which these "
                    f"detections do not have. Available: {available}",
                ) from error
            except (TypeError, ValueError) as error:  # e.g. a whole array or a string
                raise ValueError(
                    f"Query {self.query!r} can't be evaluated on these detections "
                    f"({error}). Index array fields, as in xyxy[1] or keypoints_xy[3, 0].",
                ) from error
            if holds:
                matches.append(labels)
                if len(matches) > 1:
                    break

        if len(matches) == 1:
            tracker_id[candidates] = matches[0]
        detections.tracker_id = tracker_id
        return detections


def _fields(detections: sv.Detections) -> list[SimpleNamespace]:
    """Per-detection fields: the `sv.Detections` attributes, its `data`, and the box
    centre as `x`, `y`."""
    columns = {
        "xyxy": detections.xyxy,
        "x": (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2,
        "y": (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2,
        "area": detections.area,
        "box_area": detections.box_area,
    }
    if detections.confidence is not None:
        columns["confidence"] = detections.confidence
    if detections.class_id is not None:
        columns["class_id"] = detections.class_id
    columns |= {
        key: value for key, value in detections.data.items() if key not in EXCLUDED_FIELDS
    }
    visible = detections.data.get("keypoints_visible")
    if "keypoints_xy" in columns and visible is not None:
        keypoints = np.asarray(columns["keypoints_xy"], dtype=float).copy()
        keypoints[~np.asarray(visible, dtype=bool)] = np.nan
        columns["keypoints_xy"] = keypoints
    return [
        SimpleNamespace(**{name: column[i] for name, column in columns.items()})
        for i in range(len(detections))
    ]
