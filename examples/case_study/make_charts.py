"""Generate the SMS-spam case-study charts from the measured benchmark numbers.

These are the REAL results from the end-to-end run (UCI SMS Spam Collection,
held-out disjoint test set of 1,031 messages, 128 spam / 903 ham):

    Llama-3.1 FT (s1-ready) : acc 0.991  prec 0.961  rec 0.969  f1 0.965
    Llama-3.1 base (no FT)  : acc 0.425  prec 0.175  rec 0.977  f1 0.297
    TF-IDF + NaiveBayes     : acc 0.970  prec 0.839  rec 0.938  f1 0.886

    Fine-tuned confusion matrix (spam = positive):
        TP=124  FN=4
        FP=5    TN=898

Run:  python examples/case_study/make_charts.py
Writes: metrics_comparison.png, confusion_matrix.png  (next to this script)
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- measured results -------------------------------------------------------
MODELS = ["Llama-3.1 FT\n(s1-ready)", "Llama-3.1 base\n(no fine-tune)", "TF-IDF +\nNaive Bayes"]
METRICS = ["accuracy", "precision", "recall", "f1"]
SCORES = {
    "accuracy":  [0.991, 0.425, 0.970],
    "precision": [0.961, 0.175, 0.839],
    "recall":    [0.969, 0.977, 0.938],
    "f1":        [0.965, 0.297, 0.886],
}
CONFUSION = np.array([[124, 4],   # gold spam: pred spam, pred ham
                      [5, 898]])  # gold ham:  pred spam, pred ham

COLORS = ["#2563eb", "#9ca3af", "#f59e0b"]  # FT=blue, base=gray, NB=amber


def metrics_comparison(path: str) -> None:
    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(MODELS):
        vals = [SCORES[m][i] for m in METRICS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=model.replace("\n", " "),
                      color=COLORS[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("SMS spam detection — held-out test set (1,031 msgs, 12% spam)\n"
                 "spam = positive class", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def confusion_matrix(path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(CONFUSION, cmap="Blues")
    labels = ["spam", "ham"]
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"pred {l}" for l in labels])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"gold {l}" for l in labels])
    cells = [["TP", "FN"], ["FP", "TN"]]
    vmax = CONFUSION.max()
    for i in range(2):
        for j in range(2):
            val = CONFUSION[i, j]
            color = "white" if val > vmax * 0.5 else "black"
            ax.text(j, i, f"{cells[i][j]}\n{val}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")
    ax.set_title("Fine-tuned model confusion matrix", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    mc = os.path.join(HERE, "metrics_comparison.png")
    cm = os.path.join(HERE, "confusion_matrix.png")
    metrics_comparison(mc)
    confusion_matrix(cm)
    print(f"wrote {mc}")
    print(f"wrote {cm}")


if __name__ == "__main__":
    main()
