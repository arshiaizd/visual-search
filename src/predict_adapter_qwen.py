from __future__ import annotations
from typing import Dict, Optional, Set, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_10x10

class QwenAdapter:
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = "cuda" #device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int,int]]:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type":"text", "text": prompt},
                {"type":"image", "image": img}
            ]}
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
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return parse_patch_pairs(output_text[0])

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        """
        Return per-layer, per-head attention over a 10x10 grid:
            attn: (L, H, 100)

        L = number of transformer layers
        H = number of attention heads
        100 = flattened 10x10 image grid
        """
        # ---- 1) Build inputs just like in your working script ----
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                        "image": image_path,          # path string works with process_vision_info
                        "max_pixels": 512 * 28 * 28,  # same as your demo
                    },
                ],
            }
        ]



        # Vision info
        image_inputs, _ = process_vision_info(messages)

        # Chat template (no tokenization yet)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


        # Pack text + image into model inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # ---- 2) Get vision token grid shape from image_grid_thw ----
        # This mimics: image_inputs_aux = processor.image_processor(images=image_inputs)
        image_inputs_aux = self.processor.image_processor(images=image_inputs)
        # image_grid_thw: (1, 3) => (T, Htok, Wtok)
        image_grid_thw = image_inputs_aux["image_grid_thw"].squeeze(0)
        # Your demo uses /2 here:
        #   output_shape = image_grid_thw[1:] / 2
        output_shape = (image_grid_thw[1] // 2, image_grid_thw[2] // 2)
        Htok, Wtok = int(output_shape[0]), int(output_shape[1])

        # ---- 3) Forward pass with attentions ----
        outputs = self.model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        if outputs.attentions is None:
            return None

        # outputs.attentions: tuple length L, each (B, H, seq, seq)
        attn_layers = outputs.attentions
        L = len(attn_layers)
        B, H, seq_q, seq_k = attn_layers[0].shape
        if B != 1:
            # we only handle batch size 1 in this pipeline
            return None

        # ---- 4) Find vision token range (pos..pos_end) ----
        # Same as your working example
        input_ids = inputs["input_ids"][0].tolist()
        vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id   = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        try:
            pos = input_ids.index(vision_start_token_id) + 1
            pos_end = input_ids.index(vision_end_token_id)
        except ValueError:
            # Couldn't find vision tokens
            return None

        n_img_tokens = pos_end - pos
        if Htok * Wtok != n_img_tokens:
            # Token grid shape doesn't match number of vision tokens; better to bail than lie
            return None

        # ---- 5) Build per-layer, per-head 2D maps over the vision grid ----
        # We follow your demo: attention from last token (-1) → vision tokens [pos:pos_end]
        attn_per_layer_head = []   # list[layer] -> list[head] -> 2D array (Htok, Wtok)

        for layer_attn in attn_layers:
            # layer_attn: (1, H, seq, seq)
            layer_attn = layer_attn[0]        # (H, seq, seq)
            head_maps = []

            for h in range(H):
                # Take attention from last query token to vision tokens
                # shape: (n_img_tokens,)
                vec = layer_attn[h, -1, pos:pos_end]
                vec = vec.to(torch.float32)

                # Reshape to 2D vision token grid, using your output_shape logic
                head_map = vec.reshape(Htok, Wtok)   # (Htok, Wtok)
                head_maps.append(head_map)

            attn_per_layer_head.append(head_maps)

        # ---- 6) Aggregate to 10x10 grid => (L, H, 100) ----
        # aggregate_attentions_to_10x10 expects:
        #   attn_per_layer_head: list[layer][head] -> 2D array
        #   token_hw: (Htok, Wtok)
        attn_10x10 = aggregate_attentions_to_10x10(
            attn_per_layer_head,
            token_hw=(Htok, Wtok),
        )  # should be (L, H, 100)

        attn_10x10 = np.asarray(attn_10x10, dtype=np.float32)

        cache = {
            "layer_heads": (attn_10x10.shape[0], attn_10x10.shape[1]),  # (L, H)
            "attn": attn_10x10,                                        # (L, H, 100)
            "meta": {
                "image": image_path,
                "prompt": prompt,
                # if you want, you can also insert predictions later:
                # "pred_patches": [[r,c], ...]
            },
        }
        return cache

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str) -> str:
        """
        Generic generation: same pipeline as predict_patches,
        but returns raw text instead of parsing coordinates.
        """
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": img}
            ]}
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

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text

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
