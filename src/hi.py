import torch
from transformers import AutoModel, AutoTokenizer

# Change this to wherever you stored the model
# e.g. "/home/you/models/OpenGVLab/InternVL3_5-8B"
MODEL_DIR = "/home/mmd/Desktop/Arshia/models/InternVL3_5-8B"

# 1) Load tokenizer from local folder
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    use_fast=False,
)

# 2) Load model from local folder
model = AutoModel.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,   # or torch.float16 if needed
    low_cpu_mem_usage=True,
    use_flash_attn=True,          # set False if you don’t have flash-attn
    trust_remote_code=True,
    device_map="auto",            # put it on your GPU(s)
).eval()

# 3) Define generation settings
generation_config = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# 4) Ask it something (text-only)
question = "Give me a 3-sentence summary of the history of the internet."

response, history = model.chat(
    tokenizer,
    None,                 # no image -> text-only chat
    question,
    generation_config,
    history=None,
    return_history=True,
)

print("User:", question)
print("Assistant:", response)
