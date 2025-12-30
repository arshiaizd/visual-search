# src/models/predict_adapter.py
from __future__ import annotations

from typing import List, Tuple, Union
from pathlib import Path
from PIL import Image

# ---- Default (normal) adapters ----
from src.models import predict_adapter_intern as Intern
from src.models import predict_adapter_qwen_default as QwenDefault

# ---- Stitched experiment variants (Qwen-only) ----
from src.models import predict_adapter_qwen_predecoder as QwenPreDecoder   # vision + adapter tuned out
from src.models import predict_adapter_qwen_vitonly as QwenVitOnly         # ViT only tuned out
from src.models import predict_adapter_qwen_patch_all_layers as QwenPatch

ImageLike = Union[str, Path, Image.Image]

# Normal routing is unchanged: Intern checked first, then Qwen default.
ADAPTER_MODULES = [Intern, QwenDefault]


def _get_adapter_module(model_id: str):
    mid = model_id.lower()
    for module in ADAPTER_MODULES:
        if hasattr(module, "supports") and module.supports(mid):
            return module
    raise ValueError(f"No adapter found for model_id={model_id!r}")


# ======================================================
# Normal API (used by run_eval in mode=normal)
# ======================================================

def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    adapter = _get_adapter_module(model_id)
    return adapter.generate_text(image_path, prompt, model_id)


def predict_patches(image_path: str, prompt: str, model_id: str):
    adapter = _get_adapter_module(model_id)
    return adapter.predict_patches(image_path, prompt, model_id)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    adapter = _get_adapter_module(model_id)
    return adapter.get_attention_cache(image_path, prompt, model_id)


# ======================================================
# Stitched API (used by run_eval in mode=stitched_vit)
# ======================================================

def _get_qwen_stitched_module(*, model_id: str, qwen_variant: str):
    """
    Pick which Qwen stitched implementation to use.

    qwen_variant:
      - "predecoder": patch right-before-decoder (vision + adapter tuned out)
      - "vitonly":    patch before the adapter (only ViT tuned out)
    """
    mid = model_id.lower()
    if "qwen" not in mid:
        raise ValueError("qwen_variant is only supported for Qwen models.")

    q = (qwen_variant or "").lower().strip()
    if q == "predecoder":
        return QwenPreDecoder
    if q == "vitonly":
        return QwenVitOnly
    if q == "kv_offline":
        return QwenPatch

    raise ValueError(
        f"Unknown qwen_variant={qwen_variant!r}. Expected: 'predecoder' or 'vitonly'."
    )


def generate_text_stitched(
    *,
    blank_image: ImageLike,
    isolated_images: List[ImageLike],
    isolated_rcs: List[Tuple[int, int]],
    prompt: str,
    model_id: str,
    pad: int = 1,
    max_new_tokens: int = 128,
    overwrite: bool = True,
    base_index: int = 0,
    dataset_grid: int = 12,
    qwen_variant: str = "predecoder",
) -> str:
    """
    Stitched experiments entrypoint.

    - For Qwen:
        uses qwen_variant to select the stitched implementation.
    - For non-Qwen:
        keeps old behavior (requires adapter to implement generate_text_stitched).
    """
    if "qwen" in model_id.lower():
        module = _get_qwen_stitched_module(model_id=model_id, qwen_variant=qwen_variant)
        if not hasattr(module, "generate_text_stitched"):
            raise ValueError(f"Qwen module for variant={qwen_variant!r} missing generate_text_stitched().")

        return module.generate_text_stitched(
            blank_image=blank_image,
            isolated_images=isolated_images,
            isolated_rcs=isolated_rcs,
            prompt=prompt,
            model_id=model_id,
            pad=pad,
            max_new_tokens=max_new_tokens,
            overwrite=overwrite,
            base_index=base_index,
            dataset_grid=dataset_grid,
        )

    # Non-Qwen behavior unchanged (if you later add Intern stitched)
    adapter = _get_adapter_module(model_id)
    if not hasattr(adapter, "generate_text_stitched"):
        raise ValueError(
            f"Adapter for model_id={model_id!r} does not support stitched generation. "
            "Implement generate_text_stitched in that adapter."
        )

    return adapter.generate_text_stitched(
        blank_image=blank_image,
        isolated_images=isolated_images,
        isolated_rcs=isolated_rcs,
        prompt=prompt,
        model_id=model_id,
        pad=pad,
        max_new_tokens=max_new_tokens,
        overwrite=overwrite,
        base_index=base_index,
        dataset_grid=dataset_grid,
    )

def generate_one_word_kv_offline_patched(
    *,
    canvas_image: ImageLike,
    isolated_images: List[ImageLike],
    isolated_rcs: List[Tuple[int, int]],
    prompt: str,
    model_id: str,
    pad: int = 1,
    overwrite: bool = True,
    base_index: int = 0,
    dataset_grid: int = 12,
) -> str:
    if "qwen" not in model_id.lower():
        raise ValueError("kv_offline is only supported for Qwen models.")

    module = QwenPatch
    if not hasattr(module, "generate_one_word_kv_offline_patched"):
        raise ValueError("Qwen module missing generate_one_word_kv_offline_patched().")

    return module.generate_one_word_kv_offline_patched(
        canvas_image=canvas_image,
        isolated_images=isolated_images,
        isolated_rcs=isolated_rcs,
        prompt=prompt,
        model_id=model_id,
        pad=pad,
        overwrite=overwrite,
        # base_index=base_index,
        dataset_grid=dataset_grid,
    )