from __future__ import annotations
from typing import Optional

from src.models import predict_adapter_qwen as Qwen
from src.models import predict_adapter_intern as Intern


def _is_intern_model(model_id: str) -> bool:
    # crude heuristic; adjust as needed
    mid = model_id.lower()
    return "internv3" in mid or "interv3" in mid or "opengvlab/internv3" in mid


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    if _is_intern_model(model_id):
        return Intern.generate_text(image_path, prompt, model_id)
    else:
        return Qwen.generate_text(image_path, prompt, model_id)


def predict_patches(image_path: str, prompt: str, model_id: str):
    if _is_intern_model(model_id):
        return Intern.predict_patches(image_path, prompt, model_id)
    else:
        return Qwen.predict_patches(image_path, prompt, model_id)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    if _is_intern_model(model_id):
        return Intern.get_attention_cache(image_path, prompt, model_id)
    else:
        return Qwen.get_attention_cache(image_path, prompt, model_id)
