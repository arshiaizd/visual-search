# src/models/predict_adapter_qwen_patch_all_layers.py
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple, Union, List, Any
from pathlib import Path
import inspect

from src.models.qwen_shared import get_qwen, get_device
import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info

from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_grid

ImageLike = Union[str, Path, Image.Image]


def _as_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(str(image)).convert("RGB")


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
    Return (pos, pos_end) indices of vision token span *inside* the LLM input sequence,
    where pos is the first VISION-PATCH token index (right after <|vision_start|>)
    and pos_end is the index of <|vision_end|> token.

    So the actual image patch tokens are: [pos, pos_end)  (i.e., pos .. pos_end-1)
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


def _map_rc_dataset_to_tok(
    r: int, c: int, *,
    grid_dataset: int,
    Htok: int, Wtok: int
) -> Tuple[int, int]:
    """Map dataset-grid (e.g. 12x12) coordinate to token grid (Htok x Wtok)."""
    if Htok == grid_dataset and Wtok == grid_dataset:
        return int(r), int(c)

    rt = int(round((r + 0.5) * Htok / grid_dataset - 0.5))
    ct = int(round((c + 0.5) * Wtok / grid_dataset - 0.5))
    rt = max(0, min(Htok - 1, rt))
    ct = max(0, min(Wtok - 1, ct))
    return rt, ct


def _neighborhood_pids(r: int, c: int, H: int, W: int, pad: int) -> List[int]:
    """Return row-major pids for (r,c) neighborhood clipped to grid."""
    pids: List[int] = []
    for rr in range(r - pad, r + pad + 1):
        for cc in range(c - pad, c + pad + 1):
            if 0 <= rr < H and 0 <= cc < W:
                pids.append(rr * W + cc)
    return pids


def _pkv_to_dynamic(pkv) -> Any:
    """
    Normalize whatever the model returns for past_key_values into a cache object.

    - If already a Cache-like object (DynamicCache or related), keep it.
    - If legacy tuple/list format, convert to DynamicCache when possible.
    """
    if isinstance(pkv, DynamicCache):
        return pkv

    # If pkv is a legacy cache: tuple(tuple(k,v), ...) or list[(k,v), ...]
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(pkv)

    raise RuntimeError("This transformers build does not support DynamicCache.from_legacy_cache().")


# ----------------------------------------------------------------------
# ✅ FIX: Patch cache objects directly (no DynamicCache->legacy conversion).
# Robust K/V discovery across transformers versions and cache wrappers.
# ----------------------------------------------------------------------

def _is_4d_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.ndim == 4


def _normalize_layer_container(obj: Any) -> Optional[List[Any]]:
    """
    Normalize common container types into a python list.
    Accepts list/tuple, dict (sorted by key), or stacked tensor (L, B, H, S, D).
    """
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        # sort layer indices if dict-like
        try:
            return [obj[k] for k in sorted(obj.keys())]
        except Exception:
            return list(obj.values())
    if torch.is_tensor(obj) and obj.ndim == 5:
        # assume first dim is layers
        return [obj[i] for i in range(int(obj.shape[0]))]
    return None


def _extract_kv_from_cache_object(cache_obj: Any) -> Optional[Tuple[List[Any], List[Any]]]:
    """
    Try extracting per-layer K/V containers directly from cache_obj.
    Returns (keys_container, values_container) where each element is *usually* a 4D tensor,
    but some transformers versions may insert empty lists for skipped layers.
    """
    # 0) If to_legacy_cache exists, it is often the most reliable "public" representation.
    # (May not exist in older builds.)
    if hasattr(cache_obj, "to_legacy_cache") and callable(getattr(cache_obj, "to_legacy_cache")):
        try:
            legacy = cache_obj.to_legacy_cache()
            if isinstance(legacy, (list, tuple)) and legacy and isinstance(legacy[0], (list, tuple)) and len(legacy[0]) >= 2:
                ks = [kv[0] for kv in legacy]
                vs = [kv[1] for kv in legacy]
                print(1)
                return ks, vs
        except Exception:
            pass

    # 1) Canonical DynamicCache storage (HF reference): key_cache / value_cache
    for k_name, v_name in [
        ("key_cache", "value_cache"),
        ("_key_cache", "_value_cache"),
        ("keys", "values"),
        ("k_cache", "v_cache"),
    ]:
        if hasattr(cache_obj, k_name) and hasattr(cache_obj, v_name):
            try:
                k_raw = getattr(cache_obj, k_name)
                v_raw = getattr(cache_obj, v_name)
            except Exception:
                continue

            k_list = _normalize_layer_container(k_raw)
            v_list = _normalize_layer_container(v_raw)
            if k_list is not None and v_list is not None and len(k_list) == len(v_list):
                print(2)
                return k_list, v_list

    # 2) Some cache implementations store a "cache" attribute
    if hasattr(cache_obj, "cache"):
        try:
            c = getattr(cache_obj, "cache")
        except Exception:
            c = None

        # 2a) cache is already list[(k,v), ...]
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (list, tuple)) and len(c[0]) >= 2:
            if _is_4d_tensor(c[0][0]) or _is_4d_tensor(c[0][1]) or isinstance(c[0][0], (list, tuple)):
                ks = [kv[0] for kv in c]
                vs = [kv[1] for kv in c]
                print(3)
                return ks, vs

        # 2b) cache is list of layer-objects containing k/v
        if isinstance(c, (list, tuple)) and c and not isinstance(c[0], (list, tuple)):
            ks, vs = [], []
            ok = True
            for layer in c:
                found = False
                for k_name, v_name in [
                    ("key", "value"),
                    ("k", "v"),
                    ("key_cache", "value_cache"),
                    ("keys", "values"),
                ]:
                    if hasattr(layer, k_name) and hasattr(layer, v_name):
                        try:
                            k = getattr(layer, k_name)
                            v = getattr(layer, v_name)
                        except Exception:
                            continue
                        ks.append(k)
                        vs.append(v)
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok and len(ks) == len(vs) and len(ks) > 0:
                print(4)
                return ks, vs

    # 3) Some versions: cache_obj.layers is a list of per-layer objects
    if hasattr(cache_obj, "layers"):
        try:
            layers = getattr(cache_obj, "layers")
        except Exception:
            layers = None

        if isinstance(layers, (list, tuple)) and layers:
            ks, vs = [], []
            ok = True
            for layer in layers:
                found = False
                for k_name, v_name in [
                    ("key", "value"),
                    ("k", "v"),
                    ("key_cache", "value_cache"),
                    ("keys", "values"),
                    ("key_states", "value_states"),
                ]:
                    if hasattr(layer, k_name) and hasattr(layer, v_name):
                        try:
                            k = getattr(layer, k_name)
                            v = getattr(layer, v_name)
                        except Exception:
                            continue
                        # print(k_name)
                        ks.append(k)
                        vs.append(v)
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok and len(ks) == len(vs) and len(ks) > 0:
                # print(5)
                return ks, vs

    # 4) Try iterating cache_obj (some Cache classes implement __iter__)
    try:
        items = list(iter(cache_obj))
        if items and isinstance(items[0], (list, tuple)) and len(items[0]) >= 2:
            ks = [kv[0] for kv in items]
            vs = [kv[1] for kv in items]
            print(6)
            return ks, vs
    except Exception:
        pass

    return None


def _extract_kv_lists_from_any_cache(cache_obj: Any) -> Tuple[List[Any], List[Any]]:
    """
    Recursively search for a cache that directly stores per-layer K/V tensors.
    Handles wrappers like EncoderDecoderCache / HybridCache via common attribute names.
    """
    seen: set[int] = set()

    def rec(obj: Any) -> Optional[Tuple[List[Any], List[Any]]]:
        if obj is None:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        direct = _extract_kv_from_cache_object(obj)
        if direct is not None:
            return direct

        print(123123)
        # common wrapper attributes in transformers.cache_utils
        for attr in [
            "self_attention_cache",
            "decoder_cache",
            "cache",
            "past_key_values",
            "_cache",
            "inner_cache",
            "base_cache",
        ]:
            if hasattr(obj, attr):
                try:
                    child = getattr(obj, attr)
                except Exception:
                    continue
                got = rec(child)
                if got is not None:
                    return got

        # last resort: scan attributes for nested cache-like objects
        try:
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                if name.lower() in ("config", "model", "device"):
                    continue
                try:
                    child = getattr(obj, name)
                except Exception:
                    continue
                # avoid huge tensors
                if torch.is_tensor(child):
                    continue
                # descend into objects that look like caches
                if hasattr(child, "get_seq_length") or hasattr(child, "key_cache") or hasattr(child, "to_legacy_cache"):
                    got = rec(child)
                    if got is not None:
                        return got
        except Exception:
            pass

        return None

    out = rec(cache_obj)
    if out is None:
        raise RuntimeError(
            "Could not locate per-layer K/V tensors inside the cache object returned by this transformers version. "
            "This usually means your model is returning a non-standard cache wrapper; "
            "print(type(past_key_values), dir(past_key_values)) at runtime to confirm layout."
        )
    return out


def _splice_kv_positions_inplace_dynamic(
    dst_cache: Any,
    src_cache: Any,
    abs_positions: torch.Tensor,
):
    """
    Overwrite dst's K/V at sequence positions abs_positions using src's K/V,
    across ALL layers, directly on the underlying cache tensors.
    """
    kdst, vdst = _extract_kv_lists_from_any_cache(dst_cache)
    ksrc, vsrc = _extract_kv_lists_from_any_cache(src_cache)
    # S = 160
    # idx = torch.unique(abs_positions.clamp(0, S - 1))
    # print('-------------')
    # print(kdst[0][0, 0, idx[0], 0], ksrc[0][0, 0, idx[0], 0])

    L = min(len(kdst), len(ksrc), len(vdst), len(vsrc))
    for li in range(L):
        kd, vd = kdst[li], vdst[li]
        ks, vs = ksrc[li], vsrc[li]

        # transformers may include skipped layers as empty lists
        if not (_is_4d_tensor(kd) and _is_4d_tensor(vd) and _is_4d_tensor(ks) and _is_4d_tensor(vs)):
            continue

        S = min(kd.shape[2], vd.shape[2], ks.shape[2], vs.shape[2])
        if S <= 0:
            continue

        idx = torch.unique(abs_positions.clamp(0, S - 1))
        if idx.numel() == 0:
            continue

        kd[:, :, idx, :] = ks[:, :, idx, :].to(kd.dtype)
        vd[:, :, idx, :] = vs[:, :, idx, :].to(vd.dtype)


    # kdst, vdst = _extract_kv_lists_from_any_cache(dst_cache)
    # ksrc, vsrc = _extract_kv_lists_from_any_cache(src_cache)
    # idx = torch.unique(abs_positions.clamp(0, S - 1))
    # print('++++++++++++++++++++++++')
    # print(kdst[0][0, 0, idx[0], 0], ksrc[0][0, 0, idx[0], 0])


def _call_with_supported_kwargs(fn, **kwargs):
    """
    Filter kwargs to what fn accepts (helps across HF versions for cache_position, etc.).
    IMPORTANT: do NOT retry with unfiltered kwargs, because that can reintroduce
    vision tensors and trigger Qwen's placeholder mismatch error.
    """
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        # print(filtered)
        return fn(**filtered)
    except (ValueError, TypeError):
        # signature() can fail on some wrapped callables; in that case call directly
        return fn(**kwargs)


def _strip_vision_extra_if_no_mm_tokens(
    model: Qwen2_5_VLForConditionalGeneration,
    input_ids_chunk: torch.Tensor,
    extra: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Qwen2.5-VL raises ValueError if pixel_values exist but the current input_ids
    contains 0 image placeholder tokens. This happens in cached decoding when we
    feed only text tokens. So: remove vision-related kwargs unless the chunk
    actually contains image/video placeholder tokens.
    """
    if not extra:
        return extra

    ids = input_ids_chunk.reshape(-1)

    image_token_id = getattr(model.config, "image_token_id", None)
    video_token_id = getattr(model.config, "video_token_id", None)

    n_img = int((ids == image_token_id).sum().item()) if image_token_id is not None else 0
    n_vid = int((ids == video_token_id).sum().item()) if video_token_id is not None else 0
    if (n_img + n_vid) > 0:
        return extra  # chunk contains MM placeholders -> keep vision kwargs

    # chunk is text-only -> MUST drop vision feature kwargs
    drop_keys = {
        "pixel_values",
        "image_grid_thw",
        "image_sizes",
        "pixel_values_videos",
        "video_grid_thw",
        "video_sizes",
        # some processors/models use these names:
        "images",
        "videos",
    }
    return {k: v for k, v in extra.items() if k not in drop_keys}



@torch.no_grad()
def _append_tokens_with_cache(
    model,
    *,
    input_ids_chunk: torch.Tensor,          # [1, T] (ideally T=1; see note below)
    attention_mask_full: torch.Tensor,      # [1, prev_len + T]
    dyn_cache: DynamicCache,
    extra: Dict[str, torch.Tensor],
):
    device = input_ids_chunk.device
    prev_len = dyn_cache.get_seq_length()
    T = int(input_ids_chunk.shape[1])
    cache_position = torch.arange(prev_len, prev_len + T, device=device)

    # Like your second script: use prepare_inputs_for_generation and DO NOT strip vision kwargs
    model_inputs = _call_with_supported_kwargs(
        model.prepare_inputs_for_generation,
        input_ids=input_ids_chunk,
        past_key_values=dyn_cache,
        attention_mask=attention_mask_full,
        cache_position=cache_position,
        **extra,   # keep pixel_values, image_grid_thw, etc.
    )

    if isinstance(model_inputs, dict):
        model_inputs.pop("use_cache", None)
        model_inputs.pop("return_dict", None)

    out = _call_with_supported_kwargs(
        model.forward,
        **model_inputs,
        # use_cache=True,
        return_dict=True,
    )

    pkv = getattr(out, "past_key_values", None) or getattr(out, "cache", None)
    new_cache = _pkv_to_dynamic(pkv)
    last_logits = out.logits[:, -1, :]
    return last_logits, new_cache




class QwenAdapterPatch:
    """Patch Qwen2.5-VL adapter."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = get_device(device)
        self.processor, self.model = get_qwen(model_id, device=self.device)

    def _build_inputs(self, image_path: str | Path, prompt: str):
        img = Image.open(str(image_path)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]}]
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

    # Image FIRST, then text prompt (so prompt tokens come AFTER vision span).
    def _build_inputs_image_first(self, image: ImageLike, prompt: str):
        img = _as_pil(image)
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
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

    def _split_stageA_stageB_at_vision_end(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int, int]:
        input_ids = inputs["input_ids"][0].tolist()
        pos, pos_end = _find_vision_span(self.processor, input_ids)
        if pos < 0:
            raise RuntimeError("No vision span found; cannot stage-split.")
        cut = pos_end + 1  # include <|vision_end|>

        attn = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"], dtype=torch.long))
        stageA = dict(inputs)
        stageB = dict(inputs)

        stageA["input_ids"] = inputs["input_ids"][:, :cut]
        stageA["attention_mask"] = attn[:, :cut]

        stageB["input_ids"] = inputs["input_ids"][:, cut:]
        stageB["attention_mask"] = attn[:, cut:]
        return stageA, stageB, pos, pos_end, cut

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        inputs, _ = self._build_inputs(image_path, prompt)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens , do_sample = False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int, int]]:
        text = self.generate_text(image_path, prompt, max_new_tokens=128)
        return parse_patch_pairs(text)

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
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

        Htok = Wtok = None
        try:
            aux = self.processor.image_processor(images=image_inputs)
            image_grid_thw = aux["image_grid_thw"].squeeze(0)  # (T,H,W)
            Htok = int(image_grid_thw[1].item() // 2)
            Wtok = int(image_grid_thw[2].item() // 2)
        except Exception:
            Htok, Wtok = None, None

        outputs = self.model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        if outputs.attentions is None:
            return None

        attn_layers = outputs.attentions
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

        attn_per_layer_head = []
        for layer_attn in attn_layers:
            layer_attn = layer_attn[0]
            head_maps = []
            for h in range(Hh):
                vec = layer_attn[h, -1, pos:pos_end].to(torch.float32)
                head_maps.append(vec.reshape(Htok, Wtok))
            attn_per_layer_head.append(head_maps)

        attn_grid = aggregate_attentions_to_grid(attn_per_layer_head, token_hw=(Htok, Wtok), grid_size=12)
        attn_grid = np.asarray(attn_grid, dtype=np.float32)

        return {
            "layer_heads": (attn_grid.shape[0], attn_grid.shape[1]),
            "attn": attn_grid,
            "meta": {"image": image_path, "prompt": prompt, "model_id": self.model_id},
        }

    # ============================================================
    # Offline patching on KV-cache with 3x3 neighborhoods
    # ============================================================
    @torch.inference_mode()
    def generate_one_word_kv_offline_patched(
        self,
        canvas_image: ImageLike,
        isolated_images: List[ImageLike],
        isolated_rcs: List[Tuple[int, int]],
        prompt: str,
        *,
        pad: int = 1,
        dataset_grid: int = 12,
        overwrite: bool = True,
        max_word_tokens: int = 60,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        if len(isolated_images) != len(isolated_rcs):
            raise ValueError("isolated_images and isolated_rcs must have the same length.")

        # Build full inputs for the CANVAS (image first, then prompt)
        full_inputs, _ = self._build_inputs_image_first(canvas_image, prompt)
        stageA, stageB, pos, pos_end, cut = self._split_stageA_stageB_at_vision_end(full_inputs)

        n_img_tokens = pos_end - pos
        if n_img_tokens <= 0:
            raise RuntimeError("No image patch tokens found (pos_end - pos <= 0).")

        Htok, Wtok = _infer_hw(n_img_tokens)

        # Stage A prefill on canvas (image tokens only)
        outA = self.model(**stageA, use_cache=True, return_dict=True)
        pkvA = getattr(outA, "past_key_values", None) or getattr(outA, "cache", None)
        tgt_cache = _pkv_to_dynamic(pkvA)

        # Prefill Stage A caches for each isolated image
        iso_caches: List[Any] = []
        for im in isolated_images:
            iso_full, _ = self._build_inputs_image_first(im, prompt)
            iso_stageA, _, iso_pos, iso_pos_end, iso_cut = self._split_stageA_stageB_at_vision_end(iso_full)

            iso_n = iso_pos_end - iso_pos
            if iso_n != n_img_tokens:
                raise RuntimeError(
                    f"Vision token mismatch: canvas has {n_img_tokens}, isolated has {iso_n}. "
                    "Ensure identical processing/resolution."
                )
            if iso_cut != cut:
                raise RuntimeError(
                    f"Stage-A cut mismatch: canvas cut={cut}, isolated cut={iso_cut}. "
                    "Ensure identical processor template / ordering / resolution."
                )

            out_iso = self.model(**iso_stageA, use_cache=True, return_dict=True)
            pkv_iso = getattr(out_iso, "past_key_values", None) or getattr(out_iso, "cache", None)
            iso_cache = _pkv_to_dynamic(pkv_iso)

            # cache length check (best-effort)
            try:
                if hasattr(iso_cache, "get_seq_length") and hasattr(tgt_cache, "get_seq_length"):
                    if int(iso_cache.get_seq_length()) != int(tgt_cache.get_seq_length()):
                        raise RuntimeError("Stage-A cache length mismatch across images; ensure identical inputs.")
            except Exception:
                pass

            iso_caches.append(iso_cache)

        # Apply 3x3 KV patches into the CANVAS cache
        device = stageA["input_ids"].device
        written = torch.zeros((n_img_tokens,), dtype=torch.bool, device=device) if not overwrite else None

        for (r_ds, c_ds), src_cache in zip(isolated_rcs, iso_caches):
            rt, ct = _map_rc_dataset_to_tok(int(r_ds), int(c_ds), grid_dataset=dataset_grid, Htok=Htok, Wtok=Wtok)
            pids = _neighborhood_pids(rt, ct, Htok, Wtok, pad=pad)

            if written is not None:
                filtered = []
                for pid in pids:
                    if not written[pid].item():
                        filtered.append(pid)
                        written[pid] = True
                pids = filtered

            if not pids:
                continue

            abs_positions = torch.tensor([pos + pid for pid in pids], dtype=torch.long, device=device)
            _splice_kv_positions_inplace_dynamic(tgt_cache, src_cache, abs_positions)

        patched_cache = tgt_cache

        # Stage B: feed remaining tokens (prompt tokens) using patched KV
        attn_full = torch.cat([stageA["attention_mask"], stageB["attention_mask"]], dim=1)
        extra = {k: v for k, v in full_inputs.items() if k not in ("input_ids", "attention_mask")}

        if stageB["input_ids"].shape[1] > 0:
            for t in range(stageB["input_ids"].shape[1]):
                next_chunk = stageB["input_ids"][:, t:t+1]
                last_logits, patched_cache = _append_tokens_with_cache(
                    self.model,
                    input_ids_chunk=next_chunk,
                    attention_mask_full=attn_full[:, : stageA["input_ids"].shape[1] + t + 1],
                    dyn_cache=patched_cache,
                    extra=extra,
                )
            running_attn = attn_full
        else:
            last_logits = outA.logits[:, -1, :]
            running_attn = stageA["attention_mask"]

        # Generate up to max_word_tokens, then take the first word
        gen_ids: List[int] = []
        cur_attn = running_attn
        txts = []

        for tra in range(max_word_tokens):
            scores = last_logits
            if do_sample:
                if temperature is not None and float(temperature) > 0:
                    scores = scores / float(temperature)
                probs = torch.softmax(scores, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_id = torch.argmax(scores, dim=-1)

            tid = int(next_id.item())
            gen_ids.append(tid)

            eos_id = self.processor.tokenizer.eos_token_id
            if eos_id is not None and tid == eos_id:
                break

            next_chunk = next_id.view(1, 1).to(device)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)],
                dim=1,
            )

            last_logits, patched_cache = _append_tokens_with_cache(
                self.model,
                input_ids_chunk=next_chunk,
                attention_mask_full=cur_attn,
                dyn_cache=patched_cache,
                extra=extra,
            )

            decoded_so_far = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            txt = decoded_so_far.strip()
            txts.append(txt)
            # if txt and (" " in txt or "\n" in txt or "\t" in txt):
            #     break
        
        print(txt, max_word_tokens, tra)
        decoded = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        one_word = decoded.split()[0] if decoded else ""
        return one_word


# ======================================================
# Module-level wrappers
# ======================================================

_adapter_singleton: Optional[QwenAdapterPatch] = None


def get_adapter(model_id: str) -> QwenAdapterPatch:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = QwenAdapterPatch(model_id)
    return _adapter_singleton


def predict_patches(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).predict_patches(image_path, prompt)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    # ✅ FIX: pass only what the method expects
    return get_adapter(model_id).get_attention_cache(image_path, prompt)


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    return get_adapter(model_id).generate_text(image_path, prompt)


def generate_one_word_kv_offline_patched(
    canvas_image: ImageLike,
    isolated_images: List[ImageLike],
    isolated_rcs: List[Tuple[int, int]],
    prompt: str,
    model_id: str,
    *,
    pad: int = 1,
    dataset_grid: int = 12,
    overwrite: bool = True,
    max_word_tokens: int = 600,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    return get_adapter(model_id).generate_one_word_kv_offline_patched(
        canvas_image=canvas_image,
        isolated_images=isolated_images,
        isolated_rcs=isolated_rcs,
        prompt=prompt,
        pad=pad,
        dataset_grid=dataset_grid,
        overwrite=overwrite,
        max_word_tokens=max_word_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )


def supports(model_id: str) -> bool:
    return "qwen" in model_id.lower()
