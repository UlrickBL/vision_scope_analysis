# Vision Scope Analysis

A mechanistic interpretability experiment to measure how truly multimodal **Qwen3.5-2B-Base** is, using Sparse Autoencoder (SAE) features extracted from every transformer layer.

## Research Question

Qwen3.5-2B-Base is a natively multimodal model: it processes both raw text and images through the same transformer backbone. But does it *really* understand them in the same way internally? We probe this by sending the identical semantic content in two forms — a text prompt and a rendered image of that same prompt — and measuring whether the model activates the same internal features at each layer.

If representations converge (high feature overlap at deeper layers), the model builds a modality-agnostic understanding. If they stay separate throughout, text and images are processed through independent circuits that never truly merge.

---

## Method

### Model

**Qwen3.5-2B-Base** (`Qwen/Qwen3.5-2B-Base`), a 2B-parameter natively multimodal model with 24 transformer layers and a hidden dimension of 2048. Loaded with `AutoModelForImageTextToText`.

### Sparse Autoencoders (Qwen Scope)

The [Qwen Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50) project provides a SAE trained on each of the 24 transformer layers of Qwen3.5-2B-Base. Each SAE expands the 2048-dimensional residual stream into 32,768 interpretable features, keeping the top-50 active per token.

SAE application:
```
pre_acts = residual @ W_enc.T + b_enc   # (32768,)
sparse_acts = top-50 of pre_acts        # (32768,) with 50 nonzero values
```

### Two-Modality Inputs

For every prompt in the dataset:

| Modality | Input |
|---|---|
| **Text** | The question/completion as a raw text message |
| **Image** | A clean PNG rendering of the same text (white bg, black font) |

Both inputs use the same chat template and `add_generation_prompt=True`, so the model is in the same "about to generate" state. The residual stream at the **last token** is extracted after each of the 24 layers.

### Metrics

For each (prompt, layer) pair, three similarity metrics compare text vs. image feature activations:

| Metric | Formula | Meaning |
|---|---|---|
| **Jaccard** | \|A ∩ B\| / \|A ∪ B\| | Overlap of the 50 most active features |
| **Cosine** | (a · b) / (‖a‖ ‖b‖) | Directional similarity of full 32K vectors |
| **Spearman** | ρ over union(top-50) | Rank correlation of activation magnitudes |

All metrics range from 0 (no similarity) to 1 (identical).

---

## Dataset

The dataset contains 200 prompts (100 French + 100 English), each as text and as a rendered image.

**HuggingFace dataset:** `YOUR_HF_USERNAME/vision_scope_prompts`

| Column | Type | Description |
|---|---|---|
| `id` | string | `{idx:03d}_{lang}`, e.g. `007_fr` |
| `language` | string | `en` or `fr` |
| `topic` | string | One of 10 knowledge topics |
| `text` | string | The prompt text |
| `image` | Image | PNG rendering of the text |

**Topics** (10 prompts each per language): `biology`, `math`, `physics`, `chemistry`, `history`, `geography`, `literature`, `cs`, `linguistics`, `arts`

**Prompt format:** mix of ~50% completion starters (e.g. "The speed of light in a vacuum is approximately") and ~50% direct questions (e.g. "What is the powerhouse of the cell?"). French prompts are semantic equivalents, not literal translations.

---

## How to Run

### Prerequisites

- Python 3.10+
- GPU with ≥ 8 GB VRAM (recommended)
- ~15 GB free disk space (model + SAE checkpoints)
- HuggingFace account with write access (for dataset push)

### Installation

```bash
pip install -r requirements.txt
```

### Step 1 — Render images

```bash
python scripts/01_create_images.py
```

Creates 200 PNG files in `data/images/`.

### Step 2 — Push dataset to HuggingFace

```bash
HF_TOKEN=hf_... python scripts/02_push_dataset.py --repo YOUR_HF_USERNAME/vision_scope_prompts
```

### Step 3 — Extract SAE features

```bash
python scripts/03_extract_features.py --layers 0-23 --device cuda
```

This will:
1. Download SAE checkpoints (~12 GB) via `huggingface_hub`
2. Run 200 × 2 = 400 forward passes through Qwen3.5-2B-Base
3. Apply each of the 24 SAEs to the captured residuals
4. Save `results/features/layer{N:02d}_{text,image}.pt`

To test on a subset first: `--layers 0-4`

### Step 4 — Analyse and plot

```bash
python scripts/04_analyze.py
```

Outputs:
- `results/analysis_results.parquet` — per (prompt, layer) metrics table
- `results/plots/convergence_summary.png`
- `results/plots/metrics_by_language.png`
- `results/plots/topic_heatmap.png`
- `results/plots/early_vs_late_boxplot.png`

---

## Expected Outputs

### `convergence_summary.png`

Three subplots (Jaccard / Cosine / Spearman) over all 24 layers. The shaded band is the interquartile range across 200 prompts. Dashed vertical lines mark layer boundaries (0–4 = early, 5–18 = middle, 19–23 = late).

### `metrics_by_language.png`

Same metrics overlaid for French vs English prompts. If the curves are close, both languages produce similarly convergent (or divergent) representations.

### `topic_heatmap.png`

Jaccard similarity as a heatmap: rows = topics, columns = layers. Topics with more concrete, unambiguous answers (math, physics) may converge faster than abstract ones (literature, arts).

### `early_vs_late_boxplot.png`

Distribution of Jaccard similarity for early layers (0–4) vs. late layers (19–23). A visible upward shift at late layers supports modality convergence.

---

## Interpreting Results

| Pattern | Interpretation |
|---|---|
| High Jaccard (> 0.3) in late layers | Text and image representations converge — model builds modality-invariant features |
| Low Jaccard (< 0.1) throughout | Text and image are processed in separate circuits, little internal sharing |
| Gradual increase across layers | Progressive abstraction — early layers stay modality-specific, later layers generalise |
| Sharp increase at a specific layer | A "fusion point" where visual and textual representations merge |
| FR and EN curves overlap | The convergence pattern is language-agnostic |

**Baseline expectation for random features:** with 32,768 features and top-50 active, two random vectors have an expected Jaccard of ≈ 50 / (100 − 50) × 1/(32768/100) ≈ 0.15% — effectively zero. Any measured overlap above ~5% is meaningful signal.

---

## Limitations

- **Sequence length asymmetry:** text prompts tokenise to ~15–30 tokens; images produce ~300 patch tokens. Both are pooled to the last-token residual, but that token has attended to a different amount of context in each modality.
- **Last-token pooling:** the last token summarises the full input, but for short text prompts it may not be the most information-rich position. Mean pooling (also captured in memory) is a robustness check.
- **200-sample dataset:** results are suggestive, not statistically conclusive. Extend the dataset for stronger claims.

---

## References

- Qwen3.5 model family: [Qwen Blog](https://qwenlm.github.io/)
- Qwen Scope SAE: [Hugging Face](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50)
- Qwen Scope Technical Report: [arXiv 2605.11887](https://arxiv.org/abs/2605.11887)
