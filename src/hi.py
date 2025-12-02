import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig

# NEW: import the conversation template helper
from conversation import get_conv_template

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

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
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


MODEL_DIR = "/home/mmd/Desktop/Arshia/models/InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModel.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
).eval()

# ---- generation settings (will go into GenerationConfig) ----
max_new_tokens = 256
do_sample = True
temperature = 0.7
top_p = 0.9

pixel_values = load_image('data/images/img_00001.png', max_num=12).to(torch.bfloat16).to(model.device)

# -----------------------------
# Build prompt manually (like chat)
# -----------------------------
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

question = "Please describe the image shortly."  # no <image> yet

# 1) Ensure <image> placeholder
question_with_placeholder = "<image>\n" + question

# 2) Conversation template (same logic as chat())
template = get_conv_template(model.config.template)
template.system_message = model.system_message

template.append_message(template.roles[0], question_with_placeholder)
template.append_message(template.roles[1], None)
query = template.get_prompt()

# 3) Set img_context_token_id (required by InternVL)
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id

# 4) Insert the correct number of IMG_CONTEXT tokens
num_patches = pixel_values.shape[0]  # how many cropped images we fed in
image_tokens = (
    IMG_START_TOKEN
    + IMG_CONTEXT_TOKEN * (model.num_image_token * num_patches)
    + IMG_END_TOKEN
)
query = query.replace("<image>", image_tokens, 1)

# 5) Tokenize the full prompt
model_inputs = tokenizer(query, return_tensors="pt")
input_ids = model_inputs["input_ids"].to(model.device)
attention_mask = model_inputs["attention_mask"].to(model.device)

# 6) Build GenerationConfig (for the underlying language model)
eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
gen_config = GenerationConfig(
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature,
    top_p=top_p,
    eos_token_id=eos_token_id,
)

# -----------------------------
# Fuse image features + generate with attentions
# -----------------------------
with torch.no_grad():
    # a) Extract visual features (same as InternVLChatModel.generate)
    vit_embeds = model.extract_feature(pixel_values)          # (N_img, N_tokens_img, C)

    # b) Get text embeddings and inject vit_embeds at IMG_CONTEXT positions
    input_embeds = model.language_model.get_input_embeddings()(input_ids)  # (B, L, C)
    B, L, C = input_embeds.shape

    flat_ids = input_ids.reshape(-1)                          # (B*L,)
    flat_emb = input_embeds.reshape(-1, C)                    # (B*L, C)

    selected = (flat_ids == img_context_token_id)
    assert selected.sum() != 0, "No IMG_CONTEXT tokens found in the prompt."

    flat_emb[selected] = vit_embeds.reshape(-1, C).to(flat_emb.device)
    input_embeds = flat_emb.view(B, L, C)

    # c) Call the underlying LLM generate with attentions
    gen_out = model.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=gen_config,
        output_attentions=True,
        return_dict_in_generate=True,
    )

# -----------------------------
# Decode text + inspect attentions
# -----------------------------
sequences = gen_out.sequences           # (B, prompt+generated_len)
attentions = gen_out.attentions         # attentions during generation

# Decode full text then trim at template.sep (like chat() does)
full_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
response = full_text.split(template.sep.strip())[0].strip()

print(f"User: {question_with_placeholder}")
print(f"Assistant: {response}")

# `attentions` is a nested structure; you can inspect/print shapes
print(type(attentions))
if isinstance(attentions, tuple) or isinstance(attentions, list):
    # For greedy sampling, typically: attentions[step][layer]...
    first_step = attentions[0]
    print(f"num_layers: {len(first_step)}")
    print(f"shape of first layer attn at first step: {first_step[0].shape}")