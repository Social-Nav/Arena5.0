from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _read_rtf(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        'wall_time': float(row.get('wall_time') or 0.0),
                        'sim_time': float(row.get('sim_time') or 0.0),
                        'rtf': float(row.get('rtf') or 0.0),
                    }
                )
            except Exception:
                continue
    return rows


def _event_time(record: dict[str, Any]) -> float | None:
    value = record.get('sim_time')
    if value is None:
        value = record.get('time')
    try:
        return float(value)
    except Exception:
        return None


def generate_timing_plot(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = output_dir or (run_dir / 'timing')
    output_dir.mkdir(parents=True, exist_ok=True)

    timing_trace = _read_jsonl(run_dir / 'internnav_timing_trace.jsonl')
    internnav_trace = _read_jsonl(run_dir / 'internnav_trace.jsonl')
    rtf_rows = _read_rtf(run_dir / 'rtf.csv')

    lane_names = [
        'eval/reset',
        'model request',
        'model response',
        'raw cmd',
        'released cmd',
    ]
    lane_y = {name: idx for idx, name in enumerate(lane_names)}
    points: list[tuple[float, int, str, str]] = []

    for record in timing_trace:
        event = str(record.get('event') or '')
        t = _event_time(record)
        if t is None:
            continue
        if event in {'task_reset', 'eval_ready'}:
            points.append((t, lane_y['eval/reset'], event, 'tab:blue'))
        elif event == 'raw_cmd_received':
            points.append((t, lane_y['raw cmd'], event, 'tab:orange'))
        elif event == 'cmd_released':
            points.append((t, lane_y['released cmd'], event, 'tab:green'))

    for record in internnav_trace:
        event = str(record.get('event_type') or record.get('event') or '')
        t = _event_time(record)
        if t is None:
            continue
        if event == 'planning_request_started':
            points.append((t, lane_y['model request'], event, 'tab:purple'))
        elif event in {'planning_response_received', 'trajectory', 'discrete_action', 'stop'}:
            points.append((t, lane_y['model response'], event, 'tab:red'))
        elif event in {'resetting', 'episode_ready'}:
            points.append((t, lane_y['eval/reset'], event, 'tab:blue'))

    fig, (ax_events, ax_rtf) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=False,
        gridspec_kw={'height_ratios': [2.0, 1.0]},
    )

    if points:
        for t, y, label, color in points:
            ax_events.scatter([t], [y], c=color, s=22)
        ax_events.set_yticks(list(lane_y.values()), list(lane_y.keys()))
        ax_events.set_xlabel('sim time sec when available')
        ax_events.set_title('InternNav timing timeline')
        ax_events.grid(True, axis='x', alpha=0.25)
    else:
        ax_events.text(0.5, 0.5, 'No timing events found', ha='center', va='center')
        ax_events.set_axis_off()

    if rtf_rows:
        xs = [row['sim_time'] for row in rtf_rows]
        ys = [row['rtf'] for row in rtf_rows]
        ax_rtf.plot(xs, ys, linewidth=1.0)
        ax_rtf.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
        ax_rtf.set_ylabel('RTF')
        ax_rtf.set_xlabel('sim time sec')
        ax_rtf.grid(True, alpha=0.25)
    else:
        ax_rtf.text(0.5, 0.5, 'No RTF samples found', ha='center', va='center')
        ax_rtf.set_axis_off()

    fig.tight_layout()
    png_path = output_dir / 'timing_timeline.png'
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    summary = {
        'run_dir': str(run_dir),
        'timing_trace_records': len(timing_trace),
        'internnav_trace_records': len(internnav_trace),
        'rtf_samples': len(rtf_rows),
        'event_points': len(points),
        'plot_path': str(png_path),
    }
    summary_path = output_dir / 'timing_plot_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot InternNav timing manager artifacts for one run directory.')
    parser.add_argument('--dir', required=True, help='Arena eval run directory')
    parser.add_argument('--output-dir', default='', help='Optional output directory; defaults to <run>/timing')
    args = parser.parse_args()

    run_dir = Path(args.dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    summary = generate_timing_plot(run_dir, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
