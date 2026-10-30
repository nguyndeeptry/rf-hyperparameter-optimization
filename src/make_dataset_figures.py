"""
Figures that explain the data and the behaviour of the optimisers (used in README.vi.md).

    python src/make_dataset_figures.py

Outputs (docs/):
  fig_dataset_pca.png          2-D PCA projection of both datasets, coloured by class
  fig_n_estimators_sampled.png histogram of the n_estimators values tried by Random Search vs TPE
                               (read from results/<dataset>/raw_results.json)
The PCA plot is only an illustration: 2 components keep 55-63% of the variance.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

OUT = config.ROOT / "docs"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "^", "s"]


def pca_figure():
    specs = (("breast_cancer", load_breast_cancer(), {0: "malignant", 1: "benign"}),
             ("wine", load_wine(), {0: "cultivar 0", 1: "cultivar 1", 2: "cultivar 2"}))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, data, labels) in zip(axes, specs):
        pca = PCA(2).fit(StandardScaler().fit_transform(data.data))
        z = pca.transform(StandardScaler().fit_transform(data.data))
        for c, lab in labels.items():
            m = data.target == c
            ax.scatter(z[m, 0], z[m, 1], s=22, alpha=0.7, c=COLORS[c], marker=MARKERS[c],
                       label=f"{lab} (n={m.sum()})")
        v = pca.explained_variance_ratio_
        ax.set_title(f"{name}: {data.data.shape[0]} samples, {data.data.shape[1]} features")
        ax.set_xlabel(f"PC1 ({v[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({v[1]:.0%} of variance)")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_dataset_pca.png", dpi=150)
    plt.close()


def n_estimators_figure():
    edges = list(range(50, 501, 50))
    edges[-1] = 501
    labels = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])]
    labels[-1] = "450-500"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, ds in zip(axes, config.SUPPORTED_DATASETS):
        data = json.loads((config.results_dir(ds, create=False) / "raw_results.json").read_text(encoding="utf-8"))
        x = np.arange(len(labels))
        for i, (m, color, hatch) in enumerate((("random_search", COLORS[0], ""), ("tpe", COLORS[1], "///"))):
            vals = [t["params"]["n_estimators"] for r in data if r["method"] == m for t in r["trial_history"]]
            counts, _ = np.histogram(vals, bins=edges)
            ax.bar(x + (i - 0.5) * 0.38, counts, 0.38, color=color, hatch=hatch, edgecolor="white",
                   label=f"{m} (mean {np.mean(vals):.0f}, {np.mean(np.array(vals) < 150):.0%} of trials < 150)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel("n_estimators tried")
        ax.set_title(f"{ds}: where each method spends its trials")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"number of trials (all seeds pooled)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_n_estimators_sampled.png", dpi=150)
    plt.close()


def main():
    OUT.mkdir(exist_ok=True)
    pca_figure()
    n_estimators_figure()
    print(f"Saved figures -> {OUT}")


if __name__ == "__main__":
    main()
