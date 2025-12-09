from __future__ import annotations
from typing import Optional

from src.models import predict_adapter_qwen as Qwen
from src.models import predict_adapter_intern as Intern

# Add new adapters here when you create them
ADAPTER_MODULES = [Intern, Qwen]  # order = priority; Intern checked first, then Qwen


def _get_adapter_module(model_id: str):
    mid = model_id.lower()
    for module in ADAPTER_MODULES:
        if hasattr(module, "supports") and module.supports(mid):
            return module
    raise ValueError(f"No adapter found for model_id={model_id!r}")


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    adapter = _get_adapter_module(model_id)
    return adapter.generate_text(image_path, prompt, model_id)


def predict_patches(image_path: str, prompt: str, model_id: str):
    adapter = _get_adapter_module(model_id)
    return adapter.predict_patches(image_path, prompt, model_id)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    adapter = _get_adapter_module(model_id)
    return adapter.get_attention_cache(image_path, prompt, model_id)
