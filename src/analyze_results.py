import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_runs(path: Path) -> pd.DataFrame:
    data = json.load(open(path))
    records = []
    for run in data["runs"]:
        stats = run["stats"]
        scores = run["scores"]
        records.append(
            {
                "dataset": run["dataset"],
                "condition": run["condition"],
                "model": run["model"],
                "grounding_count": scores["grounding_count"],
                "region_mentions": scores["region_mentions"],
                "gap_coverage": scores["gap_coverage"],
                "alignment_hits": scores["alignment_hits"],
                "oa_rate": stats["oa_rate"],
                "gs_share": stats["gs_share"],
                "mean_citations": stats["mean_citations"],
                "median_citations": stats["median_citations"],
            }
        )
    return pd.DataFrame(records)


def plot_scores(df: pd.DataFrame, out_dir: Path) -> None:
    melted = df.melt(
        id_vars=["dataset", "condition"],
        value_vars=["region_mentions", "gap_coverage", "alignment_hits"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(data=melted, x="metric", y="score", hue="condition", palette="viridis")
    plt.title("LLM gap-detection signals by prompt condition")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Prompt")
    plt.tight_layout()
    plt.savefig(out_dir / "llm_scores.png")
    plt.close()


def plot_attention(att_df: pd.DataFrame, out_dir: Path) -> None:
    melted = att_df.melt(id_vars=["dataset"], value_vars=["oa_rate", "gs_share"])
    plt.figure(figsize=(8, 4))
    sns.barplot(data=melted, x="variable", y="value", hue="dataset", palette="mako")
    plt.title("Attention proxies: open access and Global South authorship share")
    plt.ylabel("Rate")
    plt.xlabel("Metric")
    plt.ylim(0, 1)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_proxies.png")
    plt.close()


def main() -> None:
    results_path = Path("results/llm_outputs/llm_runs.json")
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(results_path)
    df.to_csv("results/metrics_summary.csv", index=False)

    att_df = (
        df[["dataset", "oa_rate", "gs_share", "mean_citations", "median_citations"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    plot_scores(df, out_dir)
    plot_attention(att_df, out_dir)

    print("Saved plots to results/plots and metrics_summary.csv")


if __name__ == "__main__":
    main()
