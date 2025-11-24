# src/tasks/exists.py
from __future__ import annotations
from typing import Any, Dict, Optional, List

from tasks.task import BaseTask, CaseReport
from src.dataset_io import gt_patch_set


class ExistsTask(BaseTask):
    """
    Task: decide if *any* object matching the query exists in the image.
    Output should be 'yes' or 'no' (or equivalents).

    Example prompt family (to add in src/prompts.py):
      "en_exists_green_circle": [
        "Does this image contain at least one green circle? Answer 'yes' or 'no'.",
        ...
      ]
    """

    name = "exists"
    default_prompt_family = "en_exists_green_circle"

    def compute_gt(self, rec: Dict[str, Any], query: Dict[str, Any]) -> bool:
        patches = gt_patch_set(rec, query)
        return len(patches) > 0

    def parse_prediction(self, raw_text: str) -> Optional[bool]:
        text = raw_text.strip().lower()
        # You can make this more robust (handle Persian, etc.)
        if "yes" in text and "no" not in text:
            return True
        if "no" in text and "yes" not in text:
            return False
        # ambiguous
        return None

    def metrics_for_example(
        self,
        rec: Dict[str, Any],
        query: Dict[str, Any],
        gt: bool,
        pred: Optional[bool],
    ) -> Dict[str, Any]:
        if pred is None:
            correct = False
            TP = FP = FN = TN = 0
        else:
            correct = (pred == gt)
            if gt and pred:
                TP, FP, FN, TN = 1, 0, 0, 0
            elif gt and not pred:
                TP, FP, FN, TN = 0, 0, 1, 0
            elif (not gt) and pred:
                TP, FP, FN, TN = 0, 1, 0, 0
            else:
                TP, FP, FN, TN = 0, 0, 0, 1

        return {
            "gt_exists": gt,
            "pred_exists": pred,
            "correct": correct,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        }

    def aggregate(self, reports: List[CaseReport]) -> Dict[str, Any]:
        TP = FP = FN = TN = 0
        total = len(reports)
        correct = 0
        for r in reports:
            m = r.metrics
            TP += m["TP"]; FP += m["FP"]; FN += m["FN"]; TN += m["TN"]
            if m["correct"]:
                correct += 1

        acc = correct / total if total > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "n_examples": total,
        }
