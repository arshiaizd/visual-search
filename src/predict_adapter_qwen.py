from __future__ import annotations
from typing import Dict, Optional, Set, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_10x10

class QwenAdapter:
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
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
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.device)
        inputs_image = self.processor(images=img, return_tensors="pt").to(self.device)
        for k,v in inputs_image.items():
            if k not in inputs:
                inputs[k]=v
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return parse_patch_pairs(text)

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        try:
            img = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type":"text", "text": prompt},
                    {"type":"image", "image": img}
                ]}
            ]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.device)
            inputs_image = self.processor(images=img, return_tensors="pt").to(self.device)
            for k,v in inputs_image.items():
                if k not in inputs:
                    inputs[k]=v
            outputs = self.model(**inputs, output_attentions=True, use_cache=False)
            # Depending on model implementation, attentions might not include spatial maps.
            return None
        except Exception:
            return None

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
