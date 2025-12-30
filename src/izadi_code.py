import os
import argparse
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

QUESTION = "Check the image: does it include a green circle in any patch? Reply strictly with 'yes' or 'no'."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
MAX_NEW_TOKENS = 1


def prepare_inputs(proc, text, img_path):
    img = Image.open(img_path).convert("RGB")
    msg = [{"role": "user",
             "content": [
                 {"type": "text", "text": text},
                 {"type": "image"}, 
                 
                 ]
                 
                }
            ]
    chat = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    batch = proc(text=[chat], images=[img], padding=True, return_tensors="pt")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
    return batch


def get_single_token_ids(tok, texts):
    ids = []
    for t in texts:
        enc = tok.encode(t, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return ids


@torch.no_grad()
def generate_text_with_yes_prob(model, tok, B):
    outputs = model.generate(
        **B,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.eos_token_id or None,
        return_dict_in_generate=True,
        output_scores=True,
    )

    seq = outputs.sequences[0]
    generated_text = tok.decode(seq.tolist()[-1], skip_special_tokens=True)

    logits = outputs.scores[0]
    probs = torch.softmax(logits, dim=-1)[0]

    yes_forms = ["YES", "Yes", "yes"]
    token_ids = get_single_token_ids(tok, yes_forms)
    yes_prob = float(probs[token_ids].sum()) if token_ids else None

    return generated_text, yes_prob


@torch.no_grad()
def run(proc, model, tok, img_path):
    inputs = prepare_inputs(proc, QUESTION, img_path)
    return generate_text_with_yes_prob(model, tok, inputs)


def pred_is_yes(generated_token: str, yes_prob: float | None, threshold: float = 0.5) -> bool:
    t = (generated_token or "").strip().lower()
    if t == "yes":
        return True
    if t == "no":
        return False
    # fallback if decode is weird:
    if t.startswith("y"):
        return True
    if t.startswith("n"):
        return False
    # last fallback: probability threshold
    if yes_prob is not None:
        return yes_prob >= threshold
    return False


def load_gt_from_annotations_jsonl(annotations_path: str) -> dict:
    """
    Returns mapping: basename(image_file) -> has_target (bool)
    Example key: 'image_000.png' -> True/False
    """
    gt = {}
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no} in {annotations_path}: {e}") from e

            img = obj.get("image", "")
            has_target = obj.get("has_target", None)
            if img and isinstance(has_target, bool):
                gt[os.path.basename(img)] = has_target

    if not gt:
        raise RuntimeError(f"No valid entries found in {annotations_path} (expected fields: image, has_target).")
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="/home/mmd/Desktop/Arshia/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output_path", type=str, default="izadi_result/Baseline_with_target.json")
    ap.add_argument("--input_folder_path", type=str, default="/home/mmd/Desktop/Arshia/visual-search/data/images")
    ap.add_argument("--annotations_path", type=str, default="data/annotations.jsonl")
    ap.add_argument("--yes_threshold", type=float, default=0.5)
    args = ap.parse_args()

    # Infer annotations.jsonl path from input folder (data_1000/images -> data_1000/annotations.jsonl)
    if args.annotations_path is None:
        data_root = os.path.dirname(args.input_folder_path.rstrip("/"))
        args.annotations_path = os.path.join(data_root, "annotations.jsonl")

    gt_by_basename = load_gt_from_annotations_jsonl(args.annotations_path)

    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    tok = proc.tokenizer

    output = {}

    TP = TN = FP = FN = 0
    missing_gt = 0

    files = sorted(os.listdir(args.input_folder_path))
    for file in files:
        file_path = os.path.join(args.input_folder_path, file)
        if not os.path.isfile(file_path):
            continue

        # Ground truth from annotations.jsonl using basename match
        true_yes = gt_by_basename.get(os.path.basename(file_path), None)

        generated_token, yes_prob = run(proc, model, tok, file_path)
        pred_yes = pred_is_yes(generated_token, yes_prob, threshold=args.yes_threshold)

        output[file_path] = {
            "generated_token": generated_token,
            "yes_prob": yes_prob,
            "predicted_label": "yes" if pred_yes else "no",
            "true_label": None if true_yes is None else ("yes" if true_yes else "no"),
        }

        if true_yes is None:
            missing_gt += 1
            continue

        if pred_yes and true_yes:
            TP += 1
        elif (not pred_yes) and (not true_yes):
            TN += 1
        elif pred_yes and (not true_yes):
            FP += 1
        else:
            FN += 1

        print(f"image: {file_path}")

    output["__summary__"] = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "missing_gt": missing_gt,
        "total_images": len([f for f in files if os.path.isfile(os.path.join(args.input_folder_path, f))]),
        "yes_threshold": args.yes_threshold,
        "annotations_path": args.annotations_path,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
