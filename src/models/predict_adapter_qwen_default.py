# src/models/predict_adapter_qwen_default.py
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple
from pathlib import Path
from src.models.qwen_shared import get_qwen, get_device
import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_grid


def _infer_hw(n: int) -> Tuple[int, int]:
    """Infer (H,W) from token count n by choosing a factorization with minimal |H-W|."""
    s = int(round(n ** 0.5))
    if s * s == n:
        return s, s
    best = (1, n)
    gap = n
    for h in range(1, int(n ** 0.5) + 1):
        if n % h == 0:
            w = n // h
            if abs(h - w) < gap:
                best = (h, w)
                gap = abs(h - w)
    return best


def _find_vision_span(processor: AutoProcessor, input_ids_1d: list[int]) -> Tuple[int, int]:
    """
    Return (start, end) indices of vision token span in the LLM input sequence,
    where start is the first vision token index and end is the index *after* the last one.

    Span is the tokens between: <|vision_start|> ... <|vision_end|>
    """
    tok = processor.tokenizer
    vs = tok.convert_tokens_to_ids("<|vision_start|>")
    ve = tok.convert_tokens_to_ids("<|vision_end|>")

    try:
        pos = input_ids_1d.index(vs) + 1
        pos_end = input_ids_1d.index(ve)
        if pos_end <= pos:
            raise ValueError
        return pos, pos_end
    except ValueError:
        return -1, -1


class QwenAdapterDefault:
    """
    Default Qwen2.5-VL adapter (NO stitching experiments).

    Provides:
      - generate_text(image_path, prompt) -> str
      - predict_patches(image_path, prompt) -> set[(r,c)]
      - get_attention_cache(image_path, prompt) -> {"attn": (L,H,144), ...}
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = get_device(device)

        self.processor, self.model = get_qwen(model_id, device=self.device)
        

    def _build_inputs(self, image_path: str | Path, prompt: str):
        img = Image.open(str(image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                    
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs, image_inputs

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        inputs, _ = self._build_inputs(image_path, prompt)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens , do_sample = False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int, int]]:
        text = self.generate_text(image_path, prompt, max_new_tokens=128)
        return parse_patch_pairs(text)

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        """
        Return per-layer, per-head attention aggregated onto a 12x12 grid:
          attn: (L, H, 144) float32
        """
        # Use the "image": image_path form to keep processor consistent with your other codepaths
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image_path, "max_pixels": 512 * 28 * 28},
                ],
            }
        ]

        image_inputs, _ = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Try to get token-grid from processor metadata (best-effort)
        Htok = Wtok = None
        try:
            aux = self.processor.image_processor(images=image_inputs)
            image_grid_thw = aux["image_grid_thw"].squeeze(0)  # (T,H,W)
            Htok = int(image_grid_thw[1].item() // 2)
            Wtok = int(image_grid_thw[2].item() // 2)
        except Exception:
            Htok, Wtok = None, None

        outputs = self.model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.attentions is None:
            return None

        attn_layers = outputs.attentions  # tuple(len=L) of (B, H, S, S)
        B, Hh, _, _ = attn_layers[0].shape
        if B != 1:
            return None

        input_ids = inputs["input_ids"][0].tolist()
        pos, pos_end = _find_vision_span(self.processor, input_ids)
        if pos < 0:
            return None

        n_img_tokens = pos_end - pos
        if n_img_tokens <= 0:
            return None

        if Htok is None or Wtok is None or (Htok * Wtok != n_img_tokens):
            Htok, Wtok = _infer_hw(n_img_tokens)

        # Build per-layer, per-head Htok×Wtok maps
        attn_per_layer_head = []
        for layer_attn in attn_layers:
            layer_attn = layer_attn[0]  # (H, S, S)
            head_maps = []
            for h in range(Hh):
                vec = layer_attn[h, -1, pos:pos_end].to(torch.float32)  # (N,)
                head_maps.append(vec.reshape(Htok, Wtok))
            attn_per_layer_head.append(head_maps)

        # Pool onto 12×12 grid and flatten -> (L,H,144)
        attn_grid = aggregate_attentions_to_grid(
            attn_per_layer_head,
            token_hw=(Htok, Wtok),
            grid_size=12,
        )
        attn_grid = np.asarray(attn_grid, dtype=np.float32)

        return {
            "layer_heads": (attn_grid.shape[0], attn_grid.shape[1]),
            "attn": attn_grid,
            "meta": {"image": image_path, "prompt": prompt, "model_id": self.model_id},
        }


# ======================================================
# Module-level wrappers (what your project imports)
# ======================================================

_adapter_singleton: Optional[QwenAdapterDefault] = None


def get_adapter(model_id: str) -> QwenAdapterDefault:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = QwenAdapterDefault(model_id)
    return _adapter_singleton


def predict_patches(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).predict_patches(image_path, prompt)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).get_attention_cache(image_path, prompt)


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    return get_adapter(model_id).generate_text(image_path, prompt)


def supports(model_id: str) -> bool:
    return "qwen" in model_id.lower()
