from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Set, Tuple, Union, Optional

from PIL import Image


PathLike = Union[str, Path]


def read_annotations(ann_path: PathLike) -> List[dict]:
    ann: List[dict] = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann.append(json.loads(line))
    return ann


def gt_patch_set(rec: dict, query: Dict[str, str]) -> Set[Tuple[int, int]]:
    targets: Set[Tuple[int, int]] = set()
    for o in rec.get("objects", []):
        ok = True
        for k, v in query.items():
            if o.get(k) != v:
                ok = False
                break
        if ok:
            targets.add((int(o["r"]), int(o["c"])))
    return targets


def all_grid_cells(grid: int) -> Set[Tuple[int, int]]:
    return {(r, c) for r in range(grid) for c in range(grid)}


# ======================================================
# Experiment 1 helpers
# ======================================================

def image_path_from_rec(rec: dict) -> Path:
    """
    Returns the absolute/relative path as stored in rec["image"].
    Caller can join with dataset root if desired.
    """
    return Path(rec["image"])


def isolated_dir_for_rec(
    rec: dict,
    *,
    isolated_root: PathLike = "data/isolated",
    naming: str = "image_id",
) -> Path:
    """
    Compute the isolated folder for this record.

    Our new dataset generator uses:
      folder = data/isolated/image_{image_id:05d}

    If you changed the naming scheme, adjust here.

    naming:
      - "image_id": uses rec["image_id"]
      - "pair_id_without": legacy scheme: pair_{pair_id:05d}_without
    """
    isolated_root = Path(isolated_root)

    if naming == "image_id":
        image_id = int(rec["image_id"])
        return isolated_root / f"image_{image_id:05d}"

    if naming == "pair_id_without":
        pair_id = int(rec["pair_id"])
        return isolated_root / f"pair_{pair_id:05d}_without"

    raise ValueError(f"Unknown naming scheme: {naming!r}")


def isolated_object_paths_in_id_order(
    rec: dict,
    *,
    isolated_root: PathLike = "data/isolated",
    naming: str = "image_id",
) -> List[Path]:
    """
    Return isolated object image paths ordered by object 'id'.

    This uses the annotation list order (id == index) and constructs
    filenames that match the generator:
      obj_{id:02d}_{shape_name}_r{r}_c{c}.png

    We intentionally construct from rec (not glob+sort) to avoid any
    filesystem ordering issues.
    """
    folder = isolated_dir_for_rec(rec, isolated_root=isolated_root, naming=naming)

    paths: List[Path] = []
    for o in rec.get("objects", []):
        obj_id = int(o["id"])
        shape = o["shape"]
        color = o["color"]
        r = int(o["r"])
        c = int(o["c"])
        shape_name = f"{color}_{shape}"
        fname = f"obj_{obj_id:02d}_{shape_name}_r{r}_c{c}.png"
        paths.append(folder / fname)

    return paths


def object_rcs_in_id_order(rec: dict) -> List[Tuple[int, int]]:
    """
    Return [(r,c), ...] aligned with isolated_object_paths_in_id_order().

    We trust that 'id' corresponds to list order at generation time, but we
    also sort defensively by id just in case.
    """
    objs = rec.get("objects", [])
    objs_sorted = sorted(objs, key=lambda o: int(o["id"]))
    return [(int(o["r"]), int(o["c"])) for o in objs_sorted]


def make_blank_canvas_from_rec(rec: dict) -> Image.Image:
    """
    Create a white PIL image of the dataset image size (grid*patch).
    Useful as the blank baseline input for the stitched-vision experiment.
    """
    grid = int(rec["grid"])
    patch = int(rec["patch"])
    full = grid * patch
    return Image.new("RGB", (full, full), (255, 255, 255))


def resolve_dataset_root(ann_path: PathLike) -> Path:
    """
    Convenience: given path to annotations.jsonl, infer dataset root directory.
    Example:
      ann_path = /.../data/annotations.jsonl
      returns  /.../data
    """
    p = Path(ann_path)
    return p.parent
