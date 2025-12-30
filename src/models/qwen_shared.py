# src/models/qwen_shared.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Global cache: (model_id, device, dtype_str) -> (processor, model)
_QWEN_CACHE: Dict[Tuple[str, str, str], Tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]] = {}


def get_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype_for(device: str) -> torch.dtype:
    # keep consistent with your earlier logic
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def get_qwen(
    model_id: str,
    *,
    device: Optional[str] = None,
    force_reload: bool = False,
):
    """
    Return (processor, model) from a global cache.
    Ensures Qwen is loaded ONCE per process for each (model_id, device, dtype).
    """
    device = get_device(device)
    dtype = _dtype_for(device)
    key = (str(model_id), device, str(dtype).replace("torch.", ""))

    if force_reload or key not in _QWEN_CACHE:
        print(f"[QWEN_SHARED] Loading model once: model_id={model_id} device={device} dtype={dtype}")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device).eval()

        _QWEN_CACHE[key] = (processor, model)

    return _QWEN_CACHE[key]
