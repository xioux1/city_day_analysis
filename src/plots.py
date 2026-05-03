from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plots(metrics: dict, pred_df: pd.DataFrame, task: str, plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []

    if task == "classification" and "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        path = plots_dir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.6)
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    path_scatter = plots_dir / "prediction_scatter.png"
    fig.tight_layout()
    fig.savefig(path_scatter, dpi=150)
    plt.close(fig)
    output.append(path_scatter)

    if task == "classification" and "roc_auc" in metrics:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(["ROC AUC"], [metrics["roc_auc"]])
        ax.set_ylim(0, 1)
        ax.set_title("ROC AUC")
        path = plots_dir / "roc_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output.append(path)

    return output
