from __future__ import annotations
from typing import Any, Dict, Set, Tuple, List

from tasks.task import BaseTask, CaseReport
from src.dataset_io import gt_patch_set
from src.eval_metrics import confusion_per_image, macro_scores
from src.parse_patches import parse_patch_pairs


class PatchCoordinatesTask(BaseTask):
    """
    Task: predict all (r,c) patches that match the query
    (e.g. green circle). This is your current visual search task.
    """

    name = "coords"
    # you can change this; it's just a friendly default
    default_prompt_family = "en_find_green_circle"

    def compute_gt(self, rec: Dict[str, Any], query: Dict[str, Any]) -> Set[Tuple[int, int]]:
        return gt_patch_set(rec, query)

    def parse_prediction(self, raw_text: str) -> Set[Tuple[int, int]]:
        return parse_patch_pairs(raw_text)

    def metrics_for_example(
        self,
        rec: Dict[str, Any],
        query: Dict[str, Any],
        gt: Set[Tuple[int, int]],
        pred: Set[Tuple[int, int]],
    ) -> Dict[str, Any]:
        grid = int(rec.get("grid", 10))
        conf = confusion_per_image(gt, pred, grid=grid)
        tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]

        correct = (len(fp) == 0 and len(fn) == 0)  # perfect prediction

        return {
            # sets (used for overlays, advanced stats)
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gt": len(gt),
            # scalar counts for CSV
            "TP": len(tp),
            "FP": len(fp),
            "FN": len(fn),
            # generic flag for task-agnostic stats
            "correct": correct,
        }


    def aggregate(self, reports: List[CaseReport]) -> Dict[str, Any]:
        """
        Aggregate patch-level confusion and compute macro precision/recall/F1.
        """
        confs = []
        for r in reports:
            confs.append(
                {
                    "tp": r.metrics["tp"],
                    "fp": r.metrics["fp"],
                    "fn": r.metrics["fn"],
                }
            )
        return macro_scores(confs)
