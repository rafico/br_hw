from __future__ import annotations

import json
import time


def time_stage(timings: dict, stage_name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    timings[stage_name] = time.perf_counter() - t0
    return result


def write_timing_report(timings: dict, output_file: str = "timings.json"):
    if not timings:
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

    total = sum(timings.values())
    print("\n=== Pipeline Timings ===")
    for stage, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = (100.0 * elapsed / total) if total else 0.0
        print(f"{stage:22s}: {elapsed:8.2f}s ({pct:5.1f}%)")
    print(f"{'total':22s}: {total:8.2f}s")
    print(f"Timing report saved to: {output_file}")
