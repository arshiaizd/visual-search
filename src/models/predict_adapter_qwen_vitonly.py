# src/models/predict_adapter_qwen_vitonly.py
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
import os
import inspect

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from src.models.qwen_shared import get_qwen, get_device
from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_grid

ImageLike = Union[str, Path, Image.Image]


# ======================================================
# Low-level helpers
# ======================================================

def _as_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(str(image)).convert("RGB")


def _infer_hw(n: int) -> Tuple[int, int]:
    """Infer (H,W) from token count n by choosing factorization with minimal |H-W|."""
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


def _find_vision_token_positions(processor: AutoProcessor, input_ids_1d: List[int]) -> List[int]:
    """
    Return the *sequence indices* of vision tokens in the LLM sequence:
      <|vision_start|> [VISION TOKENS...] <|vision_end|>
    """
    tok = processor.tokenizer
    vs = tok.convert_tokens_to_ids("<|vision_start|>")
    ve = tok.convert_tokens_to_ids("<|vision_end|>")

    vis: List[int] = []
    i = 0
    while i < len(input_ids_1d):
        if input_ids_1d[i] == vs:
            j = i + 1
            while j < len(input_ids_1d) and input_ids_1d[j] != ve:
                vis.append(j)
                j += 1
            i = j
        i += 1
    return vis


def _map_rc_dataset_to_tok(
    r: int, c: int, *,
    grid_dataset: int,
    Htok: int, Wtok: int
) -> Tuple[int, int]:
    """Map dataset-grid coord (grid_dataset x grid_dataset) to token-grid (Htok x Wtok)."""
    if Htok == grid_dataset and Wtok == grid_dataset:
        return int(r), int(c)

    rt = int(round((r + 0.5) * Htok / grid_dataset - 0.5))
    ct = int(round((c + 0.5) * Wtok / grid_dataset - 0.5))
    rt = max(0, min(Htok - 1, rt))
    ct = max(0, min(Wtok - 1, ct))
    return rt, ct


def _neighborhood_pids(r: int, c: int, H: int, W: int, pad: int) -> List[int]:
    """Row-major pids for (r,c) neighborhood clipped to grid."""
    pids: List[int] = []
    for rr in range(r - pad, r + pad + 1):
        for cc in range(c - pad, c + pad + 1):
            if 0 <= rr < H and 0 <= cc < W:
                pids.append(rr * W + cc)
    return pids


# ======================================================
# Robust hook payload search/replace (merger args/kwargs)
# ======================================================

def _shape_style(t: torch.Tensor) -> str:
    if t.ndim == 4:  # (1,1,N,D)
        return "B1T1ND"
    if t.ndim == 3:  # (1,N,D)
        return "B1ND"
    if t.ndim == 2:  # (N,D)
        return "ND"
    return "OTHER"


def _to_ND(t: torch.Tensor) -> torch.Tensor:
    """Normalize (1,1,N,D) or (1,N,D) or (N,D) -> (N,D)."""
    if t.ndim == 4:          # (1,1,N,D)
        t = t[:, 0, :, :]    # (1,N,D)
    if t.ndim == 3:          # (1,N,D)
        t = t[0]             # (N,D)
    return t


def _find_token_tensor_path(payload, N_expected: int, max_depth: int = 7, _depth: int = 0):
    """
    Find a tensor that looks like the ViT token sequence *for this sample*.
    Accepts:
      - (1,N,D)
      - (1,1,N,D)
      - (N,D)
    Uses N_expected to reduce false matches.
    Returns: (path, tensor)
    """
    if _depth > max_depth:
        return None

    if torch.is_tensor(payload):
        t = payload
        style = _shape_style(t)
        if style in ("B1T1ND", "B1ND", "ND"):
            tND = _to_ND(t)
            if tND.ndim == 2 and tND.size(0) >= N_expected:
                return ([], t)
        return None

    if isinstance(payload, dict):
        for k, v in payload.items():
            found = _find_token_tensor_path(v, N_expected, max_depth=max_depth, _depth=_depth + 1)
            if found is not None:
                path, t = found
                return ([("dict", k)] + path, t)
        return None

    if isinstance(payload, (list, tuple)):
        kind = "tuple" if isinstance(payload, tuple) else "list"
        for i, v in enumerate(payload):
            found = _find_token_tensor_path(v, N_expected, max_depth=max_depth, _depth=_depth + 1)
            if found is not None:
                path, t = found
                return ([(kind, i)] + path, t)
        return None

    return None


def _replace_by_path(obj, path, new_tensor):
    """Replace element at `path` in nested dict/list/tuple. Returns a NEW object."""
    if not path:
        return new_tensor

    kind, key = path[0]
    rest = path[1:]

    if kind == "dict":
        new_obj = dict(obj)
        new_obj[key] = _replace_by_path(obj[key], rest, new_tensor)
        return new_obj

    if kind == "list":
        new_obj = list(obj)
        new_obj[key] = _replace_by_path(obj[key], rest, new_tensor)
        return new_obj

    if kind == "tuple":
        new_list = list(obj)
        new_list[key] = _replace_by_path(obj[key], rest, new_tensor)
        return tuple(new_list)

    return obj


def _debug_dump(payload, max_items: int = 80):
    """Small recursive dump of tensor shapes to understand merger arg/kwarg structure."""
    out = []
    seen = 0

    def rec(x, prefix="root", depth=0):
        nonlocal seen
        if seen >= max_items:
            return
        if depth > 6:
            return

        if torch.is_tensor(x):
            out.append(f"{prefix}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
            seen += 1
            return
        if completely_basic := isinstance(x, (str, int, float, bool, type(None))):
            out.append(f"{prefix}: {type(x).__name__}={x}")
            seen += 1
            return
        if isinstance(x, dict):
            out.append(f"{prefix}: dict keys={list(x.keys())[:20]}")
            seen += 1
            for k, v in list(x.items())[:30]:
                rec(v, prefix=f"{prefix}[{k!r}]", depth=depth + 1)
            return
        if isinstance(x, (list, tuple)):
            out.append(f"{prefix}: {type(x).__name__} len={len(x)}")
            seen += 1
            for i, v in enumerate(list(x)[:30]):
                rec(v, prefix=f"{prefix}[{i}]", depth=depth + 1)
            return
        out.append(f"{prefix}: {type(x).__name__}")
        seen += 1

    rec(payload)
    return "\n".join(out)


# ======================================================
# Adapter (ViT-only tuned out)
# ======================================================

class QwenAdapterVitOnly:
    """
    Qwen2.5-VL adapter variant: tune out only ViT.

    We patch at the ViT -> adapter boundary by overriding the INPUT to visual.merger
    (i.e., ViT token sequence before merger/adapter MLP).

    Stitching rule:
      - background tokens come from isolated_images[base_index] for ALL tokens
      - overwrite neighborhood (r,c)+/-pad from each object's isolated image
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = get_device(device)

        # shared singleton loader (so multiple variants can share one loaded model)
        self.processor, self.model = get_qwen(model_id, device=self.device)

        self._merger: Optional[nn.Module] = None

    # --------------------------
    # standard API
    # --------------------------

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int, int]]:
        txt = self.generate_text(image_path, prompt)
        return parse_patch_pairs(txt)

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str) -> str:
        inputs, _ = self._build_inputs(image_path, prompt)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128 , do_sample = False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        """Return per-layer, per-head attention aggregated onto a 12x12 grid: attn: (L, H, 144)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_path, "max_pixels": 512 * 28 * 28},
            ],
        }]
        image_inputs, _ = process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        if outputs.attentions is None:
            return None

        attn_layers = outputs.attentions
        B, Hh, _, _ = attn_layers[0].shape
        if B != 1:
            return None

        input_ids = inputs["input_ids"][0].tolist()
        vis_positions = _find_vision_token_positions(self.processor, input_ids)
        if not vis_positions:
            return None

        N = len(vis_positions)
        Htok, Wtok = _infer_hw(N)

        per_layer_head = []
        for layer_attn in attn_layers:
            layer_attn = layer_attn[0]  # (Hh, seq, seq)
            head_maps = []
            for h in range(Hh):
                vec = layer_attn[h, -1, vis_positions].to(torch.float32)
                head_maps.append(vec.reshape(Htok, Wtok))
            per_layer_head.append(head_maps)

        attn_grid = aggregate_attentions_to_grid(per_layer_head, token_hw=(Htok, Wtok), grid_size=12)
        attn_grid = np.asarray(attn_grid, dtype=np.float32)

        return {
            "layer_heads": (attn_grid.shape[0], attn_grid.shape[1]),
            "attn": attn_grid,
            "meta": {"image": image_path, "prompt": prompt},
        }

    # --------------------------
    # internals
    # --------------------------

    def _build_inputs(self, image: ImageLike, prompt: str):
        img = _as_pil(image)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                {"type": "image", "image": img}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs, image_inputs

    def _get_merger(self) -> nn.Module:
        """Locate model.model.visual.merger (common Qwen2.5-VL path)."""
        if self._merger is not None:
            return self._merger

        if hasattr(self.model, "model") and hasattr(self.model.model, "visual") and hasattr(self.model.model.visual, "merger"):
            self._merger = self.model.model.visual.merger
            return self._merger

        # fallback search
        for name, mod in self.model.named_modules():
            if name.endswith("visual.merger") or name.endswith("model.visual.merger"):
                self._merger = mod
                return mod

        raise RuntimeError("Could not locate visual.merger module to hook (ViT->adapter boundary).")

    @torch.inference_mode()
    def _capture_merger_input_tokens(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Capture the INPUT token sequence to visual.merger.

        Returns:
          tokens_cpu: (N, D) on CPU.
        """
        merger = self._get_merger()

        # expected N from LLM vision span
        input_ids = inputs["input_ids"][0].tolist()
        vis_positions = _find_vision_token_positions(self.processor, input_ids)
        if not vis_positions:
            raise RuntimeError("No vision tokens found in input prompt.")
        N_expected = len(vis_positions)

        captured: List[torch.Tensor] = []
        debug = os.environ.get("QWEN_VITONLY_DEBUG", "0") == "1"

        def pre_hook(_mod, hook_args, hook_kwargs):
            payload = {"args": hook_args, "kwargs": hook_kwargs}
            found = _find_token_tensor_path(payload, N_expected=N_expected)
            if found is None:
                if debug:
                    print("[QWEN_VITONLY_DEBUG] merger pre_hook: did not find token tensor path")
                    print(_debug_dump(payload))
                    try:
                        print("[QWEN_VITONLY_DEBUG] merger.forward signature:")
                        print(inspect.signature(merger.forward))
                    except Exception:
                        pass
                return None

            path, t = found
            tND = _to_ND(t)  # (N?,D)

            if tND.ndim != 2:
                if debug:
                    print("[QWEN_VITONLY_DEBUG] found token tensor but normalization not 2D")
                    print("path:", path, "raw_shape:", tuple(t.shape), "norm_shape:", tuple(tND.shape))
                return None

            if tND.size(0) > N_expected:
                tND = tND[:N_expected, :]

            captured.append(tND.detach().to("cpu").contiguous())
            return None

        h = merger.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            _ = self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            h.remove()

        if not captured:
            raise RuntimeError(
                "Failed to capture merger input tokens.\n"
                "Set QWEN_VITONLY_DEBUG=1 to print merger arg/kwarg structure."
            )

        return captured[-1]  # (N, D) CPU

    @contextmanager
    def _patch_merger_input_tokens(self, stitched_cpu: torch.Tensor, N_expected: int):
        """
        Override the INPUT token sequence to visual.merger with stitched_cpu.

        stitched_cpu: (N, D) on CPU.
        Supports merger inputs shaped (N,D) or (1,N,D) or (1,1,N,D), and also when passed via kwargs.
        """
        merger = self._get_merger()
        stitched_cpu = stitched_cpu.contiguous()

        debug = os.environ.get("QWEN_VITONLY_DEBUG", "0") == "1"

        def pre_hook(_mod, hook_args, hook_kwargs):
            payload = {"args": hook_args, "kwargs": hook_kwargs}
            found = _find_token_tensor_path(payload, N_expected=N_expected)
            if found is None:
                if debug:
                    print("[QWEN_VITONLY_DEBUG] patch pre_hook: did not find token tensor path")
                    print(_debug_dump(payload))
                return None

            path, t = found
            style = _shape_style(t)
            tND = _to_ND(t)  # (N_here, D_here)

            if tND.ndim != 2:
                return None

            N_here, D_here = tND.size(0), tND.size(1)
            repND = stitched_cpu

            # match N_here exactly
            if repND.size(0) != N_here:
                if repND.size(0) > N_here:
                    repND = repND[:N_here, :]
                else:
                    return None

            if repND.size(1) != D_here:
                return None

            repND = repND.to(t.device, dtype=t.dtype)

            # re-wrap to same style as original
            if style == "B1T1ND":
                rep = repND.view(1, 1, N_here, D_here)
            elif style == "B1ND":
                rep = repND.view(1, N_here, D_here)
            elif style == "ND":
                rep = repND
            else:
                return None

            new_payload = _replace_by_path(payload, path, rep)
            return (new_payload["args"], new_payload["kwargs"])

        h = merger.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    # --------------------------
    # Experiment entry point
    # --------------------------

    @torch.inference_mode()
    def generate_text_stitched(
        self,
        blank_image: ImageLike,
        isolated_images: List[ImageLike],
        isolated_rcs: List[Tuple[int, int]],
        prompt: str,
        pad: int = 1,
        max_new_tokens: int = 128,
        overwrite: bool = True,
        base_index: int = 0,
        dataset_grid: int = 12,
    ) -> str:
        """
        ViT-only tuned out:

        1) Determine N, (Htok,Wtok) from blank scaffold (LLM vision token span)
        2) Build stitched merger-input token sequence:
            - base = isolated_images[base_index] tokens (all N)
            - for each object isolated image:
                overwrite pids in neighborhood around (r,c)±pad with that image's tokens
        3) Run generate() on blank while patching merger input tokens.
        """
        if len(isolated_images) != len(isolated_rcs):
            raise ValueError("isolated_images and isolated_rcs must have same length")
        if not (0 <= base_index < len(isolated_images)):
            raise ValueError("base_index out of range")

        # 1) blank scaffold -> N
        tgt_inputs, _ = self._build_inputs(blank_image, prompt)
        tgt_ids = tgt_inputs["input_ids"][0].tolist()
        tgt_vis_positions = _find_vision_token_positions(self.processor, tgt_ids)
        if not tgt_vis_positions:
            raise RuntimeError("No vision tokens found in blank target input.")
        N = len(tgt_vis_positions)
        Htok, Wtok = _infer_hw(N)

        # 2) base tokens for all N
        base_inputs, _ = self._build_inputs(isolated_images[base_index], prompt)
        base_tokens = self._capture_merger_input_tokens(base_inputs)  # (N, D)
        if base_tokens.shape[0] != N:
            raise RuntimeError(f"Base merger-input token count mismatch: expected N={N}, got {base_tokens.shape[0]}")

        stitched = base_tokens.clone()  # (N, D)
        written = torch.zeros((N,), dtype=torch.bool) if not overwrite else None

        # 3) overwrite neighborhoods per object
        for im, (r_ds, c_ds) in zip(isolated_images, isolated_rcs):
            src_inputs, _ = self._build_inputs(im, prompt)
            src_tokens = self._capture_merger_input_tokens(src_inputs)  # (N, D)
            if src_tokens.shape[0] != N:
                raise RuntimeError(
                    f"Isolated merger-input token count mismatch for {im}: expected {N}, got {src_tokens.shape[0]}"
                )

            rt, ct = _map_rc_dataset_to_tok(int(r_ds), int(c_ds), grid_dataset=dataset_grid, Htok=Htok, Wtok=Wtok)
            pids = _neighborhood_pids(rt, ct, Htok, Wtok, pad=pad)

            for pid in pids:
                if written is not None and written[pid].item():
                    continue
                stitched[pid, :] = src_tokens[pid, :]
                if written is not None:
                    written[pid] = True

        # 4) generate on blank with patched merger input
        with self._patch_merger_input_tokens(stitched, N_expected=N):
            generated_ids = self.model.generate(**tgt_inputs, do_sample=False, max_new_tokens=max_new_tokens)

        trimmed = [out[len(inp):] for inp, out in zip(tgt_inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ======================================================
# Module-level wrappers (same style as others)
# ======================================================

_adapter_singleton: Optional[QwenAdapterVitOnly] = None


def get_adapter(model_id: str) -> QwenAdapterVitOnly:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = QwenAdapterVitOnly(model_id)
    return _adapter_singleton


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    return get_adapter(model_id).generate_text(image_path, prompt)


def predict_patches(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).predict_patches(image_path, prompt)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).get_attention_cache(image_path, prompt)


def generate_text_stitched(
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
) -> str:
    return get_adapter(model_id).generate_text_stitched(
        blank_image=blank_image,
        isolated_images=isolated_images,
        isolated_rcs=isolated_rcs,
        prompt=prompt,
        pad=pad,
        max_new_tokens=max_new_tokens,
        overwrite=overwrite,
        base_index=base_index,
        dataset_grid=dataset_grid,
    )


def supports(model_id: str) -> bool:
    # Variant selection is handled by your router via qwen_variant, not by model_id string matching.
    return "qwen" in model_id.lower()
