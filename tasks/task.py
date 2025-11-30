# src/tasks/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class CaseReport:
    """
    Per-(image, prompt) result returned by each task.
    'metrics' is a free-form dict containing task-specific numbers.
    """
    image: str
    prompt_family: str
    prompt: str
    gt: Any
    pred: Any
    metrics: Dict[str, Any]
    raw_text: str | None = None


class BaseTask:
    """
    Abstract interface all tasks must implement.

    A Task knows how to:
      - derive ground truth from an annotation record (JSONL row),
      - generate model output (or use a helper for that),
      - parse the model output into a structured prediction,
      - compute metrics for a single example,
      - aggregate metrics over all examples.
    """

    # short id for CLI, e.g. "coords", "exists"
    name: str = "base"
    # default prompt family name from src.prompts
    default_prompt_family: str | None = None

    # ---- prompts -----------------------------------------------------

    def get_prompts(self, family: str | None = None) -> List[str]:
        """
        By default, tasks use PROMPT_FAMILIES[family].
        Override this if you want something custom.
        """
        from .prompts import PROMPT_FAMILIES  # local import to avoid cycles
        fam = family or self.default_prompt_family
        if fam is None:
            raise ValueError("Task must be given a prompt family.")
        if fam not in PROMPT_FAMILIES:
            raise KeyError(f"Unknown prompt family: {fam}")
        return PROMPT_FAMILIES[fam]

    # ---- core API to implement ---------------------------------------

    def compute_gt(self, rec: Dict[str, Any], query: Dict[str, Any]) -> Any:
        """
        Given one annotation record + query, return the ground truth object.
        """
        raise NotImplementedError

    def generate_text(self, image_path: str, prompt: str, model_id: str) -> str:
        """
        Run the model and return raw output text for a given (image, prompt).
        Default implementation uses Qwen adapter's generate_text.
        """
        from src.models import predict_adapter as PA
        return PA.generate_text(image_path, prompt, model_id)

    def parse_prediction(self, raw_text: str) -> Any:
        """
        Convert raw model text into structured prediction.
        """
        raise NotImplementedError

    def metrics_for_example(
        self,
        rec: Dict[str, Any],
        query: Dict[str, Any],
        gt: Any,
        pred: Any,
    ) -> Dict[str, Any]:
        """
        Compute per-example metrics (TP/FP/FN or accuracy, etc.).
        """
        raise NotImplementedError

    def score_example(
        self,
        rec: Dict[str, Any],
        query: Dict[str, Any],
        image_path: str,
        prompt_family: str,
        prompt: str,
        model_id: str,
    ) -> CaseReport:
        """
        End-to-end: run model, parse prediction, compute metrics.
        """
        raw_text = self.generate_text(image_path, prompt, model_id)
        gt = self.compute_gt(rec, query)
        pred = self.parse_prediction(raw_text)
        metrics = self.metrics_for_example(rec, query, gt, pred)
        return CaseReport(
            image=rec["image"],
            prompt_family=prompt_family,
            prompt=prompt,
            gt=gt,
            pred=pred,
            metrics=metrics,
            raw_text=raw_text,
        )

    def aggregate(self, reports: List[CaseReport]) -> Dict[str, Any]:
        """
        Aggregate metrics over dataset into a summary.json dict.
        """
        raise NotImplementedError
