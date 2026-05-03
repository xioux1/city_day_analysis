from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil


from .config import load_config
from .data import dataset_summary, load_dataset
from .evaluate import evaluate_model
from .features import infer_task
from .plots import generate_plots
from .reporting import generate_html_report
from .train import train_model


def run_experiment(config_path: Path) -> Path:
    cfg = load_config(config_path)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{cfg['run_name']}"
    run_dir = Path(cfg["output"]["results_dir"]) / run_id
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    used_config_path = run_dir / "config.yaml"
    shutil.copy2(config_path, used_config_path)

    df = load_dataset(Path(cfg["data"]["path"]))
    target_column = cfg["data"]["target_column"] or df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    task = infer_task(y, cfg["model"]["task"])
    trained = train_model(X, y, task, cfg["split"], cfg["model"])
    metrics, pred_df = evaluate_model(trained["model"], trained["X_test"], trained["y_test"], task)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_path = run_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    warnings: list[str] = []
    plot_paths = []
    try:
        plot_paths = generate_plots(metrics, pred_df, task, plots_dir)
    except Exception as exc:
        warnings.append(f"Plot generation failed: {exc}")

    run_metadata = {"run_id": run_id, "timestamp": datetime.utcnow().isoformat(), "task": task}
    summary = dataset_summary(df, target_column)
    artifacts = [used_config_path, metrics_path, predictions_path, *plot_paths]

    generate_html_report(
        run_dir=run_dir,
        run_metadata=run_metadata,
        dataset_summary=summary,
        config=cfg,
        metrics=metrics,
        plot_paths=plot_paths,
        artifact_paths=artifacts,
        warnings=warnings,
    )
    return run_dir
