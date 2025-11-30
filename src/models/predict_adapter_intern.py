from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_10x10  # reuse pooling


class InternV3Adapter:
    """
    Adapter for OpenGVLab/InternV3_5-8B (or similar InternVL models).

    Exposes the same interface as QwenAdapter:
      - generate_text(image_path, prompt)
      - predict_patches(image_path, prompt)
      - get_attention_cache(image_path, prompt) -> {attn: (L,H,100), meta: {...}}
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # You may need to adjust this depending on the model’s HF card:
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device).eval()

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str) -> str:
        """
        Generic generation: image + text → raw answer string.
        """
        img = Image.open(image_path).convert("RGB")

        # For InternVL, typical usage is similar to Qwen chat:
        messages = [
            {
                "role": "user",
                "content": [
                    # order may differ; check model docs; try image then text:
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # If InternV3 uses a chat template, you might do something like:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(
                inputs.input_ids, generated_ids
            )
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int, int]]:
        """
        Visual-search task: parse model output into (r,c) prediction set.
        """
        text = self.generate_text(image_path, prompt)
        print(text)
        return parse_patch_pairs(text)

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        """
        Return per-layer, per-head attention over a 10x10 grid:
            attn: (L, H, 100)

        NOTE: This implementation is *model-specific* and depends on
        how InternV3 exposes attention and vision tokens.
        """
        img = Image.open(image_path).convert("RGB")

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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        attentions = outputs.attentions  # tuple length L, each (B,H,seq,seq)
        if attentions is None or len(attentions) == 0:
            return None

        # --- IMPORTANT: you must figure out how InternV3 encodes vision tokens ---
        # The code below assumes there is some way to identify the contiguous
        # block of vision tokens in the input sequence and that their count
        # matches a token grid Htok × Wtok.
        #
        # For now, I’ll sketch the logic; you’ll need to adapt to InternV3 docs.

        B, Hh, seq_q, seq_k = attentions[0].shape
        if B != 1:
            return None

        # For example, if the processor provides image grid info similar to Qwen:
        # (You may need to inspect processor/image_processor for InternV3)
        try:
            image_inputs_aux = self.processor.image_processor(images=[img])
            grid_thw = image_inputs_aux["image_grid_thw"].cpu().numpy().squeeze(0)
            Htok, Wtok = int(grid_thw[1]), int(grid_thw[2])
        except Exception:
            # Fallback: assume flat 10x10 directly
            Htok, Wtok = 10, 10

        n_img_tokens = Htok * Wtok

        # You need a way to locate the vision tokens in the sequence.
        # For Qwen we used <|vision_start|> and <|vision_end|>;
        # for InternV3, check its tokenizer / docs for similar markers.
        #
        # Here we'll assume the vision tokens occupy a contiguous block
        # at some indices [pos:pos+n_img_tokens).

        # This is a placeholder; you must replace with correct logic.
        pos = 0
        pos_end = pos + n_img_tokens
        if pos_end > seq_k:
            return None

        attn_per_layer_head = []
        for layer_attn in attentions:
            layer_attn = layer_attn[0]  # (Hh, seq, seq)
            head_maps = []
            for hi in range(Hh):
                vec = layer_attn[hi, -1, pos:pos_end]  # (n_img_tokens,)
                vec = vec.to(torch.float32).detach().cpu().numpy()
                head_maps.append(vec.reshape(Htok, Wtok))
            attn_per_layer_head.append(head_maps)

        attn_10x10 = aggregate_attentions_to_10x10(
            attn_per_layer_head,
            token_hw=(Htok, Wtok),
        )  # (L, H, 100)

        cache = {
            "layer_heads": attn_10x10.shape[:2],
            "attn": attn_10x10.astype(np.float32),
            "meta": {
                "image": image_path,
                "prompt": prompt,
                "model_id": self.model_id,
            },
        }
        return cache


# singleton pattern to reuse loaded model on repeated calls
_adapter_singleton: Optional[InternV3Adapter] = None


def get_adapter(model_id: str) -> InternV3Adapter:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = InternV3Adapter(model_id)
    return _adapter_singleton


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    return get_adapter(model_id).generate_text(image_path, prompt)


def predict_patches(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).predict_patches(image_path, prompt)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).get_attention_cache(image_path, prompt)
