from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import json


def _render_obj(obj) -> str:
    return f"<pre>{escape(json.dumps(obj, indent=2, default=str))}</pre>"


def generate_html_report(
    run_dir: Path,
    run_metadata: dict,
    dataset_summary: dict,
    config: dict,
    metrics: dict,
    plot_paths: list[Path],
    artifact_paths: list[Path],
    warnings: list[str] | None = None,
) -> Path:
    warnings = warnings or []
    existing_plots = [p for p in plot_paths if p.exists()]
    missing_plots = [p for p in plot_paths if not p.exists()]

    plots_html = "".join(
        f'<div class="plot"><h4>{escape(p.name)}</h4><img src="{escape(p.relative_to(run_dir).as_posix())}" alt="{escape(p.name)}"></div>'
        for p in existing_plots
    )
    missing_html = "".join(f"<li>Missing plot: {escape(str(p))}</li>" for p in missing_plots)

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Experiment Report {escape(run_metadata['run_id'])}</title>
<style>body{{font-family:Arial,sans-serif;background:#fafafa;color:#222}}.container{{max-width:960px;margin:2rem auto;background:#fff;padding:1.5rem;border-radius:10px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f2f2f2}}.plot img{{max-width:100%;height:auto;border:1px solid #ddd}}.warn{{color:#9a6700;background:#fff8c5;padding:.5rem;border-radius:6px}}</style>
</head><body><div class='container'>
<h1>Experiment Report</h1>
<p><strong>Run ID:</strong> {escape(run_metadata['run_id'])}<br>
<strong>Date/Time:</strong> {escape(run_metadata.get('timestamp', datetime.utcnow().isoformat()))}</p>
<h2>Dataset Summary</h2>{_render_obj(dataset_summary)}
<h2>Model & Config Summary</h2>{_render_obj(config)}
<h2>Training Summary</h2>{_render_obj(run_metadata)}
<h2>Metrics</h2>{_render_obj(metrics)}
<h2>Plots</h2>{plots_html or '<p>No plots available.</p>'}
{f'<ul class="warn">{missing_html}</ul>' if missing_html else ''}
<h2>Warnings / Notes</h2>{''.join(f'<p class="warn">{escape(w)}</p>' for w in warnings) or '<p>None</p>'}
<h2>Artifacts</h2><ul>{''.join(f'<li>{escape(p.relative_to(run_dir).as_posix())}</li>' for p in artifact_paths if p.exists())}</ul>
</div></body></html>"""

    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
