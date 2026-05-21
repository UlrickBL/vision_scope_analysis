import pathlib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent
FEAT_DIR = ROOT / "results" / "features"
PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    "biology", "math", "physics", "chemistry", "history",
    "geography", "literature", "cs", "linguistics", "arts",
]
TOP_K = 50
N_LAYERS = 24

def jaccard_topk(a: torch.Tensor, b: torch.Tensor, k: int = TOP_K) -> float:
    ia = set(a.topk(k).indices.tolist())
    ib = set(b.topk(k).indices.tolist())
    return len(ia & ib) / len(ia | ib)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def spearman_topk(a: torch.Tensor, b: torch.Tensor, k: int = TOP_K) -> float:
    union_idx = torch.unique(torch.cat([a.topk(k).indices, b.topk(k).indices]))
    vals_a = a[union_idx].numpy()
    vals_b = b[union_idx].numpy()
    rho = spearmanr(vals_a, vals_b).statistic
    return float(rho) if not np.isnan(rho) else 0.0


def load_layer_pair(layer: int) -> tuple[dict, dict]:
    text_path  = FEAT_DIR / f"layer{layer:02d}_text.pt"
    image_path = FEAT_DIR / f"layer{layer:02d}_image.pt"
    if not text_path.exists():
        return None, None
    return (
        torch.load(text_path,  map_location="cpu", weights_only=True),
        torch.load(image_path, map_location="cpu", weights_only=True),
    )


def pid_to_meta(pid: str) -> tuple[int, str, str]:
    idx_s, lang = pid.split("_")
    idx = int(idx_s)
    topic = TOPICS[idx // 10]
    return idx, lang, topic

def compute_metrics() -> pd.DataFrame:
    available_layers = sorted(
        int(p.stem.replace("_text", "").replace("layer", ""))
        for p in FEAT_DIR.glob("layer*_text.pt")
    )
    if not available_layers:
        raise FileNotFoundError(
            f"No feature files found in {FEAT_DIR}. Run 03_extract_features.py first."
        )

    records = []
    for layer in tqdm(available_layers, desc="Computing metrics"):
        data_text, data_image = load_layer_pair(layer)
        if data_text is None:
            continue

        for pid in data_text:
            vec_t = data_text[pid]
            vec_i = data_image[pid]
            _, lang, topic = pid_to_meta(pid)

            records.append({
                "id":       pid,
                "layer":    layer,
                "language": lang,
                "topic":    topic,
                "jaccard":  jaccard_topk(vec_t, vec_i),
                "cosine":   cosine_sim(vec_t, vec_i),
                "spearman": spearman_topk(vec_t, vec_i),
            })

    return pd.DataFrame(records)



COLORS = {"jaccard": "#2196F3", "cosine": "#4CAF50", "spearman": "#FF9800"}
METRICS = ["jaccard", "cosine", "spearman"]
LABELS  = {"jaccard": "Jaccard (top-50)", "cosine": "Cosine", "spearman": "Spearman (top-50 union)"}


def plot_convergence(df: pd.DataFrame) -> None:
    layer_stats = df.groupby("layer")[METRICS].agg(["mean", lambda q: q.quantile(0.25), lambda q: q.quantile(0.75)])
    layer_stats.columns = [f"{m}_{s}" for m, s in layer_stats.columns]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    layers = layer_stats.index.tolist()

    for ax, metric in zip(axes, METRICS):
        color = COLORS[metric]
        mean = layer_stats[f"{metric}_mean"]
        q25  = layer_stats[f"{metric}_<lambda_0>"]
        q75  = layer_stats[f"{metric}_<lambda_1>"]

        ax.plot(layers, mean, marker="o", color=color, linewidth=2, markersize=4)
        ax.fill_between(layers, q25, q75, alpha=0.2, color=color)
        ax.axvline(x=4.5,  color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.axvline(x=18.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(LABELS[metric], fontsize=11)
        ax.set_title(f"{LABELS[metric]}\ntext vs. image", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, max(layers) + 0.5)
        ax.grid(True, alpha=0.3)
        ax.text(2.5,  0.96, "early", fontsize=8, color="gray", ha="center")
        ax.text(11.5, 0.96, "middle", fontsize=8, color="gray", ha="center")
        ax.text(21,   0.96, "late", fontsize=8, color="gray", ha="center")

    fig.suptitle(
        "SAE Feature Similarity: Text vs. Image — Qwen3.5-2B-Base\n"
        "(shaded band = IQR; dashed lines = layer-group boundaries)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "convergence_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved convergence_summary.png")


def plot_by_language(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    lang_colors = {"en": "#1565C0", "fr": "#C62828"}

    for ax, metric in zip(axes, METRICS):
        for lang, color in lang_colors.items():
            sub = df[df["language"] == lang].groupby("layer")[metric].mean()
            ax.plot(sub.index, sub.values, label=lang.upper(), color=color, linewidth=2)
        ax.set_xlabel("Layer"); ax.set_ylabel(LABELS[metric])
        ax.set_title(f"{LABELS[metric]} by language")
        ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Text vs. Image Feature Similarity by Language", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "metrics_by_language.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved metrics_by_language.png")


def plot_topic_heatmap(df: pd.DataFrame) -> None:
    pivot = df.groupby(["topic", "layer"])["jaccard"].mean().unstack()
    topics_ordered = [t for t in TOPICS if t in pivot.index]
    pivot = pivot.loc[topics_ordered]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=8)
    ax.set_yticks(range(len(topics_ordered)))
    ax.set_yticklabels(topics_ordered, fontsize=9)
    ax.set_xlabel("Layer"); ax.set_title("Jaccard Similarity by Topic and Layer")
    plt.colorbar(im, ax=ax, label="Jaccard similarity (top-50)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "topic_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved topic_heatmap.png")


def plot_early_vs_late(df: pd.DataFrame) -> None:
    available = df["layer"].unique()
    early_layers = [l for l in available if l <= 4]
    late_layers  = [l for l in available if l >= 19]

    if not early_layers or not late_layers:
        print("  Skipping early_vs_late_boxplot (insufficient layer range)")
        return

    early = df[df["layer"].isin(early_layers)]["jaccard"].values
    late  = df[df["layer"].isin(late_layers)]["jaccard"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [early, late],
        labels=[f"Early (L0–{max(early_layers)})", f"Late (L{min(late_layers)}–23)"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#BBDEFB")
    bp["boxes"][1].set_facecolor("#FFCCBC")
    ax.set_ylabel("Jaccard similarity (top-50)")
    ax.set_title("Feature Overlap: Early vs. Late Layers\nText vs. Image")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "early_vs_late_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved early_vs_late_boxplot.png")

def main() -> None:
    print("Computing metrics ...")
    df = compute_metrics()

    out = ROOT / "results" / "analysis_results.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")

    # Summary statistics
    print("\nMean metrics across all layers and prompts:")
    print(df[METRICS].mean().to_string())
    print("\nMean metrics — early layers (0–4):")
    print(df[df["layer"] <= 4][METRICS].mean().to_string())
    print("\nMean metrics — late layers (19–23):")
    print(df[df["layer"] >= 19][METRICS].mean().to_string())

    print("\nGenerating plots ...")
    plot_convergence(df)
    plot_by_language(df)
    plot_topic_heatmap(df)
    plot_early_vs_late(df)
    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
