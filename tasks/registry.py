# src/tasks/registry.py
from __future__ import annotations
from typing import Dict

from tasks.task import BaseTask
from tasks.coords import PatchCoordinatesTask
from tasks.exists import ExistsTask


TASKS: Dict[str, BaseTask] = {
    PatchCoordinatesTask.name: PatchCoordinatesTask(),
    ExistsTask.name: ExistsTask(),
}


def get_task(name: str) -> BaseTask:
    if name not in TASKS:
        raise KeyError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
    return TASKS[name]
