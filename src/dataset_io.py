from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Set, Tuple

def read_annotations(ann_path: str | Path) -> List[dict]:
    ann = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ann.append(json.loads(line))
    return ann

def gt_patch_set(rec: dict, query: Dict[str, str]) -> Set[Tuple[int,int]]:
    targets = set()
    for o in rec.get("objects", []):
        ok = True
        for k, v in query.items():
            if o.get(k) != v:
                ok = False; break
        if ok:
            targets.add((int(o["r"]), int(o["c"])))
    return targets

def all_grid_cells(grid: int) -> Set[Tuple[int,int]]:
    return {(r, c) for r in range(grid) for c in range(grid)}
