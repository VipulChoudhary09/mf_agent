#!/usr/bin/env python3
"""Summarize privacy-minimized LiveKit per-turn latency records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path("livekit_metrics.jsonl")


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile without third-party dependencies."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def e2e_by_room(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("type") != "turn_metrics" or record.get("role") != "assistant":
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        latency = metrics.get("e2e_latency")
        if isinstance(latency, (int, float)):
            grouped[str(record.get("room", "unknown"))].append(float(latency))
    return dict(grouped)


def print_summary(grouped: dict[str, list[float]], room: str | None) -> int:
    if room is not None:
        grouped = {room: grouped.get(room, [])}

    populated = {name: values for name, values in grouped.items() if values}
    if not populated:
        print("No assistant e2e_latency records found.")
        return 1

    print(
        f"{'Room':38s} {'Turns':>5s} {'Mean':>8s} "
        f"{'P50':>8s} {'P95':>8s} {'Max':>8s}"
    )
    print("-" * 83)
    for name, values in sorted(populated.items()):
        print(
            f"{name[:38]:38s} {len(values):5d} "
            f"{statistics.fmean(values):8.3f} "
            f"{statistics.median(values):8.3f} "
            f"{percentile(values, 0.95):8.3f} "
            f"{max(values):8.3f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--room", help="Show one room only")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List rooms and assistant turn counts",
    )
    args = parser.parse_args()

    grouped = e2e_by_room(load_records(args.file))
    if args.list:
        if not grouped:
            print("No rooms found.")
            return 1
        for name, values in sorted(grouped.items()):
            print(f"{name}\t{len(values)} assistant turns")
        return 0

    return print_summary(grouped, args.room)


if __name__ == "__main__":
    raise SystemExit(main())
