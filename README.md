# Vision Scope Analysis

Mechanistic interpretability experiment measuring how truly multimodal Qwen3.5 models are, using SAE features extracted from every transformer layer.

**Research question:** send the same semantic content as text and as a rendered image - does the model activate the same internal features at each layer?

Dataset: [`UlrickBL/vision_scope_prompts`](https://huggingface.co/datasets/UlrickBL/vision_scope_prompts) - 200 prompts (100 EN + 100 FR) across 10 topics, each paired with a PNG rendering of the same text.

---

## Models

| Key   | Model                  | SAE                                       |
| ----- | ---------------------- | ----------------------------------------- |
| `2b`  | `Qwen/Qwen3.5-2B-Base` | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50` |
| `27b` | `Qwen/Qwen3.5-27B`     | `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`    |

---

## Usage

```bash
pip install -r requirements.txt
```

**Extract features:**

```bash
python scripts/03_extract_features.py --model 2b --device cuda
python scripts/03_extract_features.py --model 27b --device cuda

# Quick test (10 prompts, 5 layers)
python scripts/03_extract_features.py --model 2b --n 10 --layers 0-4 --device cuda
```

**Analyse and plot:**

```bash
python scripts/04_analyze.py --model 2b
python scripts/04_analyze.py --model 27b
```

Outputs go to `results/features/{model}/` and `results/plots/{model}/`.

---

## Metrics

Three metrics compare text vs. image SAE activations per (prompt, layer):

- **Jaccard** - overlap of top-k active features
- **Cosine** - directional similarity of the full sparse vector
- **Spearman** - rank correlation over the union of top-k features

Random baseline Jaccard ≈ 0.15% - anything above ~5% is signal.

---

## Limitations

- **Last-token pooling**: image inputs produce ~300 patch tokens vs ~15–30 for text; both are pooled to the final token position.
- **200 samples**: sufficient for layer-level trends, not for per-topic statistical claims.
- **Rendered text images**: a controlled setup, not natural image distribution.

---

## References

- Qwen Scope SAE: [Hugging Face](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50)
- Qwen Scope Technical Report: [arXiv 2605.11887](https://arxiv.org/abs/2605.11887)
