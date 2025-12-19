from __future__ import annotations
from typing import Dict, Optional, Set, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, GenerationConfig

from conversation import get_conv_template  # same as in your working script

from src.parse_patches import parse_patch_pairs
from src.attn_hooks_qwen import aggregate_attentions_to_grid  # reuse pooling


# ---------- image preprocessing (copied from your working script) ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_path: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(im) for im in images]
    pixel_values = torch.stack(pixel_values)  # (N_patches, 3, H, W)
    return pixel_values


# ---------- InternVL adapter ----------

class InternVLAdapter:
    """
    Adapter API expected by the project:

      - generate_text(image_path, prompt) -> str
      - predict_patches(image_path, prompt) -> set[(r,c)]
      - get_attention_cache(image_path, prompt) -> {"attn": (L,H,100), "token_hw": (Htok,Wtok), "debug": {...}}
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        ).eval()

        self.image_size = 448
        self.max_num_patches = 12

    def _prepare_image(self, image_path: str) -> torch.Tensor:
        pv = load_image(image_path, input_size=self.image_size, max_num=self.max_num_patches)
        return pv.to(torch.bfloat16).to(self.model.device)

    def _build_prompt_and_embeds(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        # 1) Insert <image> placeholder
        question_with_placeholder = "<image>\n" + prompt

        # 2) Conversation template (same logic as your working script)
        template = get_conv_template(self.model.config.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question_with_placeholder)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        # 3) Configure image context token
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # 4) Insert visual tokens <img> <IMG_CONTEXT>... </img>
        num_patches = pixel_values.shape[0]
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * (self.model.num_image_token * num_patches)
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

        # 5) Tokenize
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.model.device)
        attention_mask = model_inputs["attention_mask"].to(self.model.device)

        # 6) Generation config for the underlying LLM
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        # 7) Fuse image features into text embeddings (like InternVLChatModel.generate)
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)           # (N_img, N_tokens_img, C)
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)  # (B, L, C)

            B, L, C = input_embeds.shape
            flat_ids = input_ids.reshape(-1)                                # (B*L,)
            flat_emb = input_embeds.reshape(-1, C)                          # (B*L, C)

            selected = (flat_ids == img_context_token_id)
            assert selected.sum() != 0, "No IMG_CONTEXT tokens found in the prompt."

            flat_emb[selected] = vit_embeds.reshape(-1, C).to(flat_emb.device)
            input_embeds = flat_emb.view(B, L, C)

        return template, input_ids, attention_mask, input_embeds, gen_config, img_context_token_id

    # ---- high-level API methods ----

    @torch.inference_mode()
    def generate_text(self, image_path: str, prompt: str) -> str:
        pixel_values = self._prepare_image(image_path)
        template, input_ids, attention_mask, input_embeds, gen_config, _ = self._build_prompt_and_embeds(
            pixel_values, prompt
        )

        gen_out = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=gen_config,
            output_attentions=False,
            return_dict_in_generate=True,
        )
        sequences = gen_out.sequences
        full_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        # For parsing patch coordinates, we don't strictly need to cut at template.sep;
        # parse_patch_pairs will pick up "(r,c)" anywhere in the text.
        return full_text

    @torch.inference_mode()
    def predict_patches(self, image_path: str, prompt: str) -> Set[Tuple[int, int]]:
        text = self.generate_text(image_path, prompt)
        return parse_patch_pairs(text)

    @torch.inference_mode()
    def get_attention_cache(self, image_path: str, prompt: str) -> Optional[Dict]:
        """
        Run InternVL with attentions and return a cache dict compatible with run_eval/run_stats:

          {
            "attn": (L, H, 144) np.float32,
            "token_hw": (Htok, Wtok),
            "meta": {...},
            "debug": {...},
          }
        """
        # 1) image → pixel_values
        pixel_values = self._prepare_image(image_path)

        # 2) build prompt + embeddings (your helper from earlier)
        (
            template,
            input_ids,
            attention_mask,
            input_embeds,
            gen_config,
            img_context_token_id,
        ) = self._build_prompt_and_embeds(pixel_values, prompt)

        # 3) generate with attentions
        gen_out = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=gen_config,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        attentions = gen_out.attentions
        if not isinstance(attentions, (list, tuple)) or len(attentions) == 0:
            return None

        # HF generate() → attentions[step][layer] with shape (B, Hh, S, S)
        last_step = attentions[-1]

        # 4) find image-token positions (IMG_CONTEXT tokens)
        img_mask = (input_ids[0] == img_context_token_id)
        img_idx = img_mask.nonzero(as_tuple=True)[0]  # (num_img_tokens,)
        num_img_tokens = img_idx.numel()
        if num_img_tokens == 0:
            return None

        # assume a square grid of visual tokens
        Htok = Wtok = int(num_img_tokens ** 0.5)
        if Htok * Wtok != num_img_tokens:
            # fallback: treat as 1×N line if not square
            Htok, Wtok = 1, num_img_tokens

        # 5) build per-layer, per-head Htok×Wtok maps from attention
        per_layer_head_maps = []
        for layer_attn in last_step:  # list over layers
            # layer_attn: (B, Hh, S, S)
            layer_attn = layer_attn.to(torch.float32)  # avoid bfloat16→numpy issues
            B, Hh, S, _ = layer_attn.shape
            head_maps = []
            for h in range(Hh):
                A = layer_attn[0, h]          # (S, S)
                # attention from last token to image tokens
                vec = A[-1, img_idx]          # (num_img_tokens,)
                head_maps.append(vec.reshape(Htok, Wtok))
            per_layer_head_maps.append(head_maps)

        # 6) pool to 10×10 and flatten → (L, Hh, 100)
        
        attn_10x10 = aggregate_attentions_to_grid(
            per_layer_head_maps,
            token_hw=(Htok, Wtok),
            grid_size= 12
        )

        # 7) build meta so run_stats.py can index by (image, prompt)
        meta = {
            # MUST match the per_case.csv "image" column exactly:
            "image": image_path,          # e.g. "data/images/img_00001.png"

            # MUST match per_case.csv "prompt" column:
            "prompt": prompt,

            # Extra info, optional:
            "model_id": self.model_id,
            "token_hw": (Htok, Wtok),
            "num_img_tokens": int(num_img_tokens),
        }

        cache = {
            "attn": attn_10x10,               # (L, Hh, 100), np.float32
            "token_hw": (Htok, Wtok),
            "meta": meta,
            "debug": {
                "num_img_tokens": int(num_img_tokens),
            },
        }
        return cache

# ---------- module-level helpers used by predict_adapter.py ----------

_adapter_singleton: Optional[InternVLAdapter] = None


def get_adapter(model_id: str) -> InternVLAdapter:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = InternVLAdapter(model_id)
    return _adapter_singleton


def generate_text(image_path: str, prompt: str, model_id: str) -> str:
    return get_adapter(model_id).generate_text(image_path, prompt)


def predict_patches(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).predict_patches(image_path, prompt)


def get_attention_cache(image_path: str, prompt: str, model_id: str):
    return get_adapter(model_id).get_attention_cache(image_path, prompt)

def supports(model_id: str) -> bool:
    mid = model_id.lower()
    return "internvl" in mid or "internv3" in mid or "opengvlab/internvl" in mid

