# src/tasks/__init__.py
from .task import BaseTask, CaseReport
from .registry import TASKS, get_task
from .prompts import PROMPT_FAMILIES
