import argparse
import pathlib
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

TOPICS = [
    "biology", "math", "physics", "chemistry", "history",
    "geography", "literature", "cs", "linguistics", "arts",
]

MODEL_CONFIGS = {
    "2b": {
        "model_id": "Qwen/Qwen3.5-2B-Base",
        "sae_repo": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
        "sae_d":    32768,
        "top_k":    50,
    },
    "27b": {
        "model_id": "Qwen/Qwen3.5-27B",
        "sae_repo": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100",
        "sae_d":    80000,
        "top_k":    100,
    },
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    print(f"Loading {model_id} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else None,
    )
    if device not in ("auto", "cpu"):
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    assert hasattr(model, "model") and hasattr(model.model, "language_model") \
        and hasattr(model.model.language_model, "layers"), \
        "Cannot find model.model.language_model.layers — hook path invalid for this model version."
    n_layers = len(model.model.language_model.layers)
    print(f"Model ready. {n_layers} transformer layers detected.")
    return model, processor, n_layers


# ── Forward hooks ─────────────────────────────────────────────────────────────

def register_hooks(model, n_layers: int) -> tuple[dict, list]:
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach().cpu().float()
        return hook

    hooks = [
        model.model.language_model.layers[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]
    return captured, hooks


def remove_hooks(hooks: list) -> None:
    for h in hooks:
        h.remove()


# ── Input preparation ─────────────────────────────────────────────────────────

def build_text_inputs(text: str, processor, device):
    return processor(text=text, return_tensors="pt").to(device)


def build_image_inputs(image: Image.Image, processor, device):
    text = f"<|vision_start|>{processor.image_token}<|vision_end|>"
    return processor(text=text, images=image, return_tensors="pt").to(device)


# ── Token pooling ─────────────────────────────────────────────────────────────

def last_token_residuals(captured: dict, attention_mask: torch.Tensor, n_layers: int) -> dict[int, torch.Tensor]:
    last_idx = attention_mask[0].nonzero()[-1].item()
    return {layer: captured[layer][0, last_idx, :] for layer in range(n_layers)}


# ── SAE application ───────────────────────────────────────────────────────────

def apply_sae(residual: torch.Tensor, W_enc: torch.Tensor, b_enc: torch.Tensor, top_k: int, sae_d: int) -> torch.Tensor:
    pre_acts = residual @ W_enc.T + b_enc
    topk_vals, topk_idx = pre_acts.topk(top_k)
    sparse = torch.zeros(sae_d, dtype=torch.float32, device=residual.device)
    sparse[topk_idx] = topk_vals
    return sparse


# ── Dataset iterator ──────────────────────────────────────────────────────────

def iter_prompts():
    ds = load_dataset("UlrickBL/vision_scope_prompts", split="train")
    yield from ds


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_layer_range(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="2b", choices=list(MODEL_CONFIGS),
                        help="Model config to use (default: 2b)")
    parser.add_argument("--layers", default=None,
                        help="Layer range to extract, e.g. '0-23'. Defaults to all layers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n", type=int, default=None,
                        help="Subsample N prompts for quick testing (default: all 200)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    device = args.device
    sae_device = device if device != "cpu" else "cpu"

    feat_dir = ROOT / "results" / "features" / args.model
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.model} | Device: {device}")

    print(f"Downloading SAE checkpoints from {cfg['sae_repo']} ...")
    sae_dir = pathlib.Path(snapshot_download(cfg["sae_repo"]))
    print(f"SAE checkpoints at: {sae_dir}")

    model, processor, n_layers = load_model(cfg["model_id"], device)
    model_device = next(model.parameters()).device

    target_layers = parse_layer_range(args.layers) if args.layers else list(range(n_layers))
    print(f"Extracting layers: {target_layers[0]}–{target_layers[-1]}")

    captured, hooks = register_hooks(model, n_layers)

    prompts = list(iter_prompts())
    if args.n is not None:
        prompts = prompts[: args.n]

    # ── Phase 1: forward passes ───────────────────────────────────────────────
    print(f"\nRunning forward passes for {len(prompts)} prompts × 2 modalities ...")
    all_residuals: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "text": {},
        "image": {},
    }

    with torch.no_grad():
        for record in tqdm(prompts, desc="Forward passes"):
            pid = record["id"]

            captured.clear()
            inputs = build_text_inputs(record["text"], processor, model_device)
            model(**inputs)
            all_residuals["text"][pid] = last_token_residuals(captured, inputs["attention_mask"], n_layers)

            captured.clear()
            inputs = build_image_inputs(record["image"], processor, model_device)
            model(**inputs)
            all_residuals["image"][pid] = last_token_residuals(captured, inputs["attention_mask"], n_layers)

            del inputs
            if device != "cpu":
                torch.cuda.empty_cache()

    remove_hooks(hooks)
    print("Forward passes complete.")

    # ── Phase 2: SAE application ──────────────────────────────────────────────
    all_ids = [r["id"] for r in prompts]

    print("\nApplying SAE layer by layer ...")
    for layer in tqdm(target_layers, desc="SAE layers"):
        sae_path = sae_dir / f"layer{layer}.sae.pt"
        ckpt = torch.load(sae_path, map_location=sae_device, weights_only=True)
        W_enc = ckpt["W_enc"].float().to(sae_device)
        b_enc = ckpt["b_enc"].float().to(sae_device)

        results: dict[str, dict[str, torch.Tensor]] = {"text": {}, "image": {}}

        for pid in all_ids:
            for modality in ("text", "image"):
                res = all_residuals[modality][pid][layer].to(sae_device)
                sparse = apply_sae(res, W_enc, b_enc, cfg["top_k"], cfg["sae_d"])
                results[modality][pid] = sparse.cpu()

        torch.save(results["text"],  feat_dir / f"layer{layer:02d}_text.pt")
        torch.save(results["image"], feat_dir / f"layer{layer:02d}_image.pt")

        del ckpt, W_enc, b_enc
        if sae_device != "cpu":
            torch.cuda.empty_cache()

    print(f"\nDone. Feature files saved to {feat_dir}")
    print(f"Files: {sorted(feat_dir.glob('*.pt'))[:4]} ...")


if __name__ == "__main__":
    main()
