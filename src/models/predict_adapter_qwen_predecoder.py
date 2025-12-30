from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from PIL import Image
from src.models.qwen_shared import get_qwen, get_device
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

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
    Return the *sequence indices* of vision tokens, i.e. token positions between:
      <|vision_start|> ... <|vision_end|>

    This is robust and does NOT rely on hooking vision modules.
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


def _get_decoder_layers(model: Qwen2_5_VLForConditionalGeneration) -> nn.ModuleList:
    """Find decoder layer list for Qwen2.5-VL."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    for _, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("Decoder layers not found.")


def _map_rc_dataset_to_tok(
    r: int, c: int, *,
    grid_dataset: int,
    Htok: int, Wtok: int
) -> Tuple[int, int]:
    """
    Map a dataset-grid (e.g. 12x12) coordinate to the token grid (Htok x Wtok).
    If Htok==Wtok==grid_dataset this is identity.
    """
    if Htok == grid_dataset and Wtok == grid_dataset:
        return int(r), int(c)

    rt = int(round((r + 0.5) * Htok / grid_dataset - 0.5))
    ct = int(round((c + 0.5) * Wtok / grid_dataset - 0.5))
    rt = max(0, min(Htok - 1, rt))
    ct = max(0, min(Wtok - 1, ct))
    return rt, ct


def _neighborhood_pids(r: int, c: int, H: int, W: int, pad: int) -> List[int]:
    """
    Return row-major pids for (r,c) neighborhood clipped to grid.
    """
    pids: List[int] = []
    for rr in range(r - pad, r + pad + 1):
        for cc in range(c - pad, c + pad + 1):
            if 0 <= rr < H and 0 <= cc < W:
                pids.append(rr * W + cc)
    return pids


# ======================================================
# Adapter
# ======================================================

class QwenAdapter:
    """
    Qwen2.5-VL adapter.

    NEW experiment (your latest spec):

      1) Run *all isolated images* through Qwen normally (vision+adapter included).
      2) Extract the representation RIGHT BEFORE the decoder:
            -> the input activation to decoder layer 0
         at vision-token positions.
      3) Build a stitched canvas of vision-token activations:
            - default/background = isolated_images[0] (base) for ALL vision tokens
            - overwrite (r,c)±pad regions using each object's isolated image
      4) Run generation on a blank image, patching ONLY decoder layer 0 input
         at vision-token positions on the first forward pass.
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = get_device(device)

        self.processor, self.model = get_qwen(model_id, device=self.device)
    # ======================================================
    # Existing methods (kept)
    # ======================================================

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int, int]]:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128 , do_sample = False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return parse_patch_pairs(output_text)

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str) -> str:
        inputs, _ = self._build_inputs(image_path, prompt)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128 , do_sample = False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        """
        Return per-layer, per-head attention aggregated onto a 12x12 grid:
            attn: (L, H, 144)
        """
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
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        if outputs.attentions is None:
            return None

        attn_layers = outputs.attentions
        B, H, _, _ = attn_layers[0].shape
        if B != 1:
            return None

        input_ids = inputs["input_ids"][0].tolist()
        vis_positions = _find_vision_token_positions(self.processor, input_ids)
        if not vis_positions:
            return None

        n_img_tokens = len(vis_positions)
        Htok, Wtok = _infer_hw(n_img_tokens)

        attn_per_layer_head = []
        for layer_attn in attn_layers:
            layer_attn = layer_attn[0]  # (H, seq, seq)
            head_maps = []
            for h in range(H):
                vec = layer_attn[h, -1, vis_positions].to(torch.float32)  # (N,)
                head_maps.append(vec.reshape(Htok, Wtok))
            attn_per_layer_head.append(head_maps)

        attn_grid = aggregate_attentions_to_grid(attn_per_layer_head, token_hw=(Htok, Wtok), grid_size=12)
        attn_grid = np.asarray(attn_grid, dtype=np.float32)

        return {
            "layer_heads": (attn_grid.shape[0], attn_grid.shape[1]),
            "attn": attn_grid,
            "meta": {"image": image_path, "prompt": prompt},
        }

    # ======================================================
    # Inputs
    # ======================================================

    def _build_inputs(self, image: ImageLike, prompt: str):
        """
        Build model inputs via processor exactly like normal generation.
        Returns (inputs, image_inputs).
        """
        img = _as_pil(image)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]}
        ]
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

    # ======================================================
    # NEW experiment: stitch at "right before decoder"
    # ======================================================

    @torch.inference_mode()
    def _capture_layer0_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        want_positions: List[int],
    ) -> torch.Tensor:
        """
        Capture decoder layer-0 *input activation* x[:, pos, :] for positions in want_positions.

        Returns:
          reps_cpu: (P, D) on CPU in the SAME order as want_positions.
        """
        layers = _get_decoder_layers(self.model)
        if len(layers) == 0:
            raise RuntimeError("No decoder layers found.")
        layer0 = layers[0]

        captured: List[torch.Tensor] = []

        def hook(_mod, hook_inputs):
            # hook_inputs[0] is x: (B, seq, D)
            x = hook_inputs[0]
            if not torch.is_tensor(x) or x.ndim != 3 or x.size(0) != 1:
                return hook_inputs
            pos = torch.tensor(want_positions, device=x.device, dtype=torch.long)
            if int(pos.max().item()) >= x.size(1):
                raise RuntimeError(
                    f"Positions out of range for layer0 input: max_pos={int(pos.max())} seq={x.size(1)}"
                )
            reps = x[0, pos, :].detach().to("cpu")  # (P, D)
            captured.append(reps)
            return hook_inputs

        h = layer0.register_forward_pre_hook(hook)
        try:
            _ = self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            h.remove()

        if not captured:
            raise RuntimeError("Failed to capture layer0 inputs (hook did not run).")

        return captured[-1]  # (P, D) CPU

    @contextmanager
    def _patch_layer0_inputs(
        self,
        *,
        tgt_vis_positions: List[int],
        stitched_reps_cpu: torch.Tensor,  # (N, D) CPU in vision-token order
        norm_scale: bool = True,
    ):
        """
        Patch ONLY decoder layer 0 input x at the vision-token sequence positions.
        This is the "right before decoder" intervention.

        During generate(), only the first forward pass has full seq length.
        Later steps use cache and will typically skip safely.
        """
        layers = _get_decoder_layers(self.model)
        layer0 = layers[0]

        pos_cpu = torch.tensor(tgt_vis_positions, dtype=torch.long)  # (N,)
        reps_cpu = stitched_reps_cpu  # (N, D)

        def hook(_mod, hook_inputs):
            x = hook_inputs[0]  # (B, seq, D)
            if not torch.is_tensor(x) or x.ndim != 3 or x.size(0) != 1:
                return hook_inputs

            pos = pos_cpu.to(x.device)
            if int(pos.max().item()) >= x.size(1):
                # generation step with seq=1 etc
                return hook_inputs

            reps = reps_cpu.to(x.device, dtype=x.dtype)  # (N, D)
            x_new = x.clone()

            tgt = x_new[0, pos, :]              # (N, D)
            src = reps                          # (N, D)

            if norm_scale:
                scale = tgt.norm(dim=-1, keepdim=True) / (src.norm(dim=-1, keepdim=True) + 1e-6)
                x_new[0, pos, :] = src * scale
            else:
                x_new[0, pos, :] = src

            return (x_new,) + tuple(hook_inputs[1:])

        h = layer0.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            h.remove()

    @torch.inference_mode()
    def generate_text_stitched(
        self,
        blank_image: ImageLike,
        isolated_images: List[ImageLike],
        isolated_rcs: List[Tuple[int, int]],   # dataset coords (typically 12x12)
        prompt: str,
        pad: int = 1,
        max_new_tokens: int = 128,
        overwrite: bool = True,
        base_index: int = 0,
        dataset_grid: int = 12,
    ) -> str:
        """
        NEW experiment implementation:

        - Build target inputs from blank_image; get tgt_vis_positions (sequence positions)
        - Build BASE stitched reps for ALL vision tokens from isolated_images[base_index]
        - For each isolated image i:
            overwrite neighborhood around its (r,c) with reps from that isolated image
        - Run generate on blank target while patching ONLY decoder layer0 inputs at vision tokens
        """
        if len(isolated_images) != len(isolated_rcs):
            raise ValueError("isolated_images and isolated_rcs must have same length")
        if not (0 <= base_index < len(isolated_images)):
            raise ValueError("base_index out of range")

        # 1) Target scaffold (blank)
        tgt_inputs, _ = self._build_inputs(blank_image, prompt)
        tgt_ids = tgt_inputs["input_ids"][0].tolist()
        tgt_vis = _find_vision_token_positions(self.processor, tgt_ids)
        if not tgt_vis:
            raise RuntimeError("No vision tokens found in blank target input.")

        N = len(tgt_vis)
        Htok, Wtok = _infer_hw(N)

        # 2) Capture BASE reps (all vision tokens) from isolated_images[base_index]
        base_inputs, _ = self._build_inputs(isolated_images[base_index], prompt)
        base_ids = base_inputs["input_ids"][0].tolist()
        base_vis = _find_vision_token_positions(self.processor, base_ids)
        if len(base_vis) != N:
            raise RuntimeError(
                f"Vision-token mismatch: blank N={N}, base isolated N={len(base_vis)}. "
                "Ensure identical processing/resolution."
            )

        base_reps_cpu = self._capture_layer0_inputs(base_inputs, want_positions=base_vis)  # (N, D) CPU
        stitched = base_reps_cpu.clone()  # (N, D)

        # 3) For each isolated image, overwrite its neighborhood from its own reps
        #    (in vision-token order: pid -> vis_positions[pid])
        written = torch.zeros((N,), dtype=torch.bool) if not overwrite else None

        for im, (r_ds, c_ds) in zip(isolated_images, isolated_rcs):
            src_inputs, _ = self._build_inputs(im, prompt)
            src_ids = src_inputs["input_ids"][0].tolist()
            src_vis = _find_vision_token_positions(self.processor, src_ids)
            if len(src_vis) != N:
                raise RuntimeError(
                    f"Vision-token mismatch: blank N={N}, isolated N={len(src_vis)} for {im}."
                )

            # map dataset (r,c) -> token-grid (rt,ct)
            rt, ct = _map_rc_dataset_to_tok(
                int(r_ds), int(c_ds),
                grid_dataset=dataset_grid,
                Htok=Htok, Wtok=Wtok,
            )
            pids = _neighborhood_pids(rt, ct, Htok, Wtok, pad=pad)
            positions = [src_vis[pid] for pid in pids]

            reps_patch_cpu = self._capture_layer0_inputs(src_inputs, want_positions=positions)  # (P,D)

            # write into stitched in vision-token order
            for j, pid in enumerate(pids):
                if not overwrite and written is not None and written[pid].item():
                    continue
                stitched[pid, :] = reps_patch_cpu[j, :]
                if written is not None:
                    written[pid] = True

        # 4) Generate on blank scaffold with layer0 patching at vision tokens
        with self._patch_layer0_inputs(
            tgt_vis_positions=tgt_vis,
            stitched_reps_cpu=stitched,  # (N,D) in vision-token order
            norm_scale=True,
        ):
            generated_ids = self.model.generate(
                **tgt_inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(tgt_inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ======================================================
# Module-level wrappers
# ======================================================

_adapter_singleton: Optional[QwenAdapter] = None


def get_adapter(model_id: str) -> QwenAdapter:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = QwenAdapter(model_id)
    return _adapter_singleton


def predict_patches(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).predict_patches(image_path, prompt)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).get_attention_cache(image_path, prompt)


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    return get_adapter(model_id).generate_text(image_path, prompt)


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
    return "qwen" in model_id.lower()
