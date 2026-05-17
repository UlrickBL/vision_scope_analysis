"""
Extract SAE feature activations for all prompts × 2 modalities (text / image).

Steps:
  1. Load Qwen3.5-2B-Base with AutoModelForImageTextToText
  2. Register forward hooks on all 24 transformer layers
  3. For each prompt: run text pass and image pass, capture residuals at last token
  4. Apply SAE layer-by-layer (one SAE at a time to save GPU memory)
  5. Save results/features/layer{N:02d}_{text,image}.pt

Prerequisites:
  - pip install -r requirements.txt
  - Run scripts/01_create_images.py first
  - ~8 GB GPU VRAM recommended; ~15 GB free disk (model + SAE checkpoints)

Usage:
    python scripts/03_extract_features.py [--layers 0-23] [--device cuda]
"""

import argparse
import pathlib
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"
FEAT_DIR = ROOT / "results" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    "biology", "math", "physics", "chemistry", "history",
    "geography", "literature", "cs", "linguistics", "arts",
]
N_LAYERS = 24
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
SAE_D = 32768
TOP_K = 50


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else None,
    )
    if device not in ("auto", "cpu"):
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    assert hasattr(model, "model") and hasattr(model.model, "layers"), \
        "Cannot find model.model.layers — hook path invalid for this model version."
    assert len(model.model.layers) == N_LAYERS, \
        f"Expected {N_LAYERS} layers, got {len(model.model.layers)}"
    print(f"Model ready. {N_LAYERS} transformer layers confirmed.")
    return model, processor


# ── Forward hooks ─────────────────────────────────────────────────────────────

def register_hooks(model) -> tuple[dict, list]:
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Store as float32 on CPU to avoid GPU memory pressure
            captured[layer_idx] = hidden.detach().cpu().float()
        return hook

    hooks = [
        model.model.layers[i].register_forward_hook(make_hook(i))
        for i in range(N_LAYERS)
    ]
    return captured, hooks


def remove_hooks(hooks: list) -> None:
    for h in hooks:
        h.remove()


# ── Input preparation ─────────────────────────────────────────────────────────

def build_text_inputs(text: str, processor, device):
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def build_image_inputs(img_path: pathlib.Path, processor, device):
    messages = [{"role": "user", "content": [{"type": "image", "url": img_path.as_uri()}]}]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


# ── Token pooling ─────────────────────────────────────────────────────────────

def last_token_residuals(captured: dict, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
    last_idx = attention_mask[0].nonzero()[-1].item()
    return {layer: captured[layer][0, last_idx, :] for layer in range(N_LAYERS)}


def mean_token_residuals(captured: dict, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
    mask = attention_mask[0].bool().cpu()
    return {
        layer: captured[layer][0][mask].mean(dim=0)
        for layer in range(N_LAYERS)
    }


# ── SAE application ───────────────────────────────────────────────────────────

def apply_sae(residual: torch.Tensor, W_enc: torch.Tensor, b_enc: torch.Tensor) -> torch.Tensor:
    """residual: (2048,) float32 → sparse acts (32768,) float32, top-50 nonzero."""
    pre_acts = residual @ W_enc.T + b_enc
    topk_vals, topk_idx = pre_acts.topk(TOP_K)
    sparse = torch.zeros(SAE_D, dtype=torch.float32)
    sparse[topk_idx] = topk_vals
    return sparse


# ── Dataset iterator ──────────────────────────────────────────────────────────

def iter_prompts():
    for lang, fname in [("en", "questions_en.txt"), ("fr", "questions_fr.txt")]:
        lines = (DATA_DIR / fname).read_text(encoding="utf-8").strip().splitlines()
        for idx, text in enumerate(lines):
            yield {
                "id": f"{idx:03d}_{lang}",
                "language": lang,
                "topic": TOPICS[idx // 10],
                "text": text,
                "image_path": IMG_DIR / f"{idx:03d}_{lang}.png",
            }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_layer_range(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0-23",
                        help="Layer range to extract, e.g. '0-23' or '0-4'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    target_layers = parse_layer_range(args.layers)
    device = args.device
    sae_device = device if device != "cpu" else "cpu"
    print(f"Device: {device} | Extracting layers: {target_layers[0]}–{target_layers[-1]}")

    # Download SAE checkpoints
    print(f"Downloading SAE checkpoints from {SAE_REPO} ...")
    sae_dir = pathlib.Path(snapshot_download(SAE_REPO))
    print(f"SAE checkpoints at: {sae_dir}")

    model, processor = load_model(device)
    model_device = next(model.parameters()).device
    captured, hooks = register_hooks(model)

    prompts = list(iter_prompts())

    # ── Phase 1: forward passes — collect all residuals ───────────────────────
    print(f"\nRunning forward passes for {len(prompts)} prompts × 2 modalities ...")
    all_residuals: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "text": {},
        "image": {},
    }

    with torch.no_grad():
        for record in tqdm(prompts, desc="Forward passes"):
            pid = record["id"]

            # TEXT pass
            captured.clear()
            inputs = build_text_inputs(record["text"], processor, model_device)
            model(**inputs)
            all_residuals["text"][pid] = last_token_residuals(captured, inputs["attention_mask"])

            # IMAGE pass
            captured.clear()
            inputs = build_image_inputs(record["image_path"], processor, model_device)
            model(**inputs)
            all_residuals["image"][pid] = last_token_residuals(captured, inputs["attention_mask"])

            del inputs
            if device != "cpu":
                torch.cuda.empty_cache()

    remove_hooks(hooks)
    print("Forward passes complete.")

    # ── Phase 2: SAE application, one layer at a time ─────────────────────────
    all_ids = [r["id"] for r in prompts]

    print("\nApplying SAE layer by layer ...")
    for layer in tqdm(target_layers, desc="SAE layers"):
        sae_path = sae_dir / f"layer{layer}.sae.pt"
        ckpt = torch.load(sae_path, map_location=sae_device, weights_only=True)
        W_enc = ckpt["W_enc"].float().to(sae_device)  # (32768, 2048)
        b_enc = ckpt["b_enc"].float().to(sae_device)  # (32768,)

        results: dict[str, dict[str, torch.Tensor]] = {"text": {}, "image": {}}

        for pid in all_ids:
            for modality in ("text", "image"):
                res = all_residuals[modality][pid][layer].to(sae_device)
                sparse = apply_sae(res, W_enc, b_enc)
                results[modality][pid] = sparse.cpu()

        torch.save(results["text"],  FEAT_DIR / f"layer{layer:02d}_text.pt")
        torch.save(results["image"], FEAT_DIR / f"layer{layer:02d}_image.pt")

        del ckpt, W_enc, b_enc
        if sae_device != "cpu":
            torch.cuda.empty_cache()

    print(f"\nDone. Feature files saved to {FEAT_DIR}")
    print(f"Files: {sorted(FEAT_DIR.glob('*.pt'))[:4]} ...")


if __name__ == "__main__":
    main()
