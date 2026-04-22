from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

VALID_SCENE_LABELS = {"normal", "crime"}


def _append_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_frame_ranges(frame_ranges: Any, path: str, errors: list[str]) -> None:
    _append_error(errors, isinstance(frame_ranges, list), f"{path} must be a list")
    if not isinstance(frame_ranges, list):
        return
    for index, frame_range in enumerate(frame_ranges):
        item_path = f"{path}[{index}]"
        _append_error(errors, isinstance(frame_range, list) and len(frame_range) == 2, f"{item_path} must be a [start, end] pair")
        if not isinstance(frame_range, list) or len(frame_range) != 2:
            continue
        start, end = frame_range
        _append_error(errors, _is_json_int(start) and _is_json_int(end), f"{item_path} values must be ints")
        if _is_json_int(start) and _is_json_int(end):
            _append_error(errors, start <= end, f"{item_path} must satisfy start <= end")


def validate_catalogue_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    _append_error(errors, isinstance(payload, dict), "catalogue payload must be a JSON object")
    if not isinstance(payload, dict):
        return errors

    catalogue = payload.get("catalogue")
    _append_error(errors, isinstance(catalogue, dict), "catalogue payload must contain a 'catalogue' object")
    if not isinstance(catalogue, dict):
        return errors

    for global_id, appearances in catalogue.items():
        try:
            int(global_id)
        except (TypeError, ValueError):
            errors.append(f"catalogue key {global_id!r} is not an int-like string")
        _append_error(errors, isinstance(appearances, list), f"catalogue[{global_id!r}] must be a list")
        if not isinstance(appearances, list):
            continue

        for index, appearance in enumerate(appearances):
            path = f"catalogue[{global_id!r}][{index}]"
            _append_error(errors, isinstance(appearance, dict), f"{path} must be an object")
            if not isinstance(appearance, dict):
                continue
            _append_error(errors, isinstance(appearance.get("clip_id"), str) and bool(appearance.get("clip_id")), f"{path}.clip_id must be a non-empty string")
            _append_error(errors, _is_json_int(appearance.get("local_track_id")), f"{path}.local_track_id must be an int")
            _append_error(errors, "frame_ranges" in appearance, f"{path}.frame_ranges is required")
            if "frame_ranges" in appearance:
                _validate_frame_ranges(appearance["frame_ranges"], f"{path}.frame_ranges", errors)
            if "cluster_probability" in appearance:
                value = appearance["cluster_probability"]
                _append_error(
                    errors,
                    _is_json_number(value) and 0.0 <= float(value) <= 1.0,
                    f"{path}.cluster_probability must be in [0, 1]",
                )

    return errors


def validate_scene_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    _append_error(errors, isinstance(payload, list), "scene payload must be a list")
    if not isinstance(payload, list):
        return errors

    for index, item in enumerate(payload):
        path = f"scene[{index}]"
        _append_error(errors, isinstance(item, dict), f"{path} must be an object")
        if not isinstance(item, dict):
            continue
        _append_error(errors, isinstance(item.get("clip_id"), str) and bool(item.get("clip_id")), f"{path}.clip_id must be a non-empty string")
        label = item.get("label")
        _append_error(errors, label in VALID_SCENE_LABELS, f"{path}.label must be one of {sorted(VALID_SCENE_LABELS)}")
        crime_segments = item.get("crime_segments", [])
        _append_error(errors, isinstance(crime_segments, list), f"{path}.crime_segments must be a list")
        if not isinstance(crime_segments, list):
            continue

        for seg_index, segment in enumerate(crime_segments):
            seg_path = f"{path}.crime_segments[{seg_index}]"
            _append_error(errors, isinstance(segment, dict), f"{seg_path} must be an object")
            if not isinstance(segment, dict):
                continue
            start = segment.get("timestamp_start")
            end = segment.get("timestamp_end")
            _append_error(errors, _is_json_number(start), f"{seg_path}.timestamp_start must be numeric")
            _append_error(errors, _is_json_number(end), f"{seg_path}.timestamp_end must be numeric")
            if _is_json_number(start) and _is_json_number(end):
                _append_error(errors, float(start) <= float(end), f"{seg_path} must satisfy timestamp_start <= timestamp_end")
            people = segment.get("involved_people_global", [])
            _append_error(errors, isinstance(people, list), f"{seg_path}.involved_people_global must be a list")
            if isinstance(people, list):
                for person_index, person_id in enumerate(people):
                    _append_error(errors, _is_json_int(person_id), f"{seg_path}.involved_people_global[{person_index}] must be an int")

    return errors


def validate_eval_report_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    _append_error(errors, isinstance(payload, dict), "eval report must be a JSON object")
    if not isinstance(payload, dict):
        return errors

    person_reid = payload.get("person_reid")
    scene = payload.get("scene")
    _append_error(errors, isinstance(person_reid, dict), "eval report must contain person_reid")
    _append_error(errors, isinstance(scene, dict), "eval report must contain scene")
    if not isinstance(person_reid, dict) or not isinstance(scene, dict):
        return errors

    metric_bounds = {
        "v_measure": (0.0, 1.0),
        "adjusted_rand_index": (-1.0, 1.0),
        "purity": (0.0, 1.0),
    }
    for key, (lower, upper) in metric_bounds.items():
        value = person_reid.get(key)
        _append_error(errors, _is_json_number(value), f"person_reid.{key} must be numeric")
        if _is_json_number(value):
            _append_error(errors, lower <= float(value) <= upper, f"person_reid.{key} must be in [{lower}, {upper}]")

    for key in ("accuracy", "macro_f1"):
        value = scene.get(key)
        _append_error(errors, _is_json_number(value), f"scene.{key} must be numeric")
        if _is_json_number(value):
            _append_error(errors, 0.0 <= float(value) <= 1.0, f"scene.{key} must be in [0, 1]")

    return errors


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_path(path: str, validator: Callable[[Any], list[str]]) -> list[str]:
    payload = _load_json(path)
    return validator(payload)


def _print_result(title: str, path: str, errors: Iterable[str]) -> int:
    errors = list(errors)
    if not errors:
        print(f"[ok] {title}: {path}")
        return 0
    print(f"[fail] {title}: {path}")
    for error in errors:
        print(f"  - {error}")
    return len(errors)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Validate pipeline output JSON files.")
    parser.add_argument("--catalogue", default="", help="Path to catalogue JSON to validate")
    parser.add_argument("--scene", default="", help="Path to scene labels JSON to validate")
    parser.add_argument("--eval-report", default="", help="Path to evaluation report JSON to validate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    total_failures = 0

    if args.catalogue:
        total_failures += _print_result("catalogue", args.catalogue, _validate_path(args.catalogue, validate_catalogue_payload))
    if args.scene:
        total_failures += _print_result("scene", args.scene, _validate_path(args.scene, validate_scene_payload))
    if args.eval_report:
        total_failures += _print_result("eval_report", args.eval_report, _validate_path(args.eval_report, validate_eval_report_payload))

    if not any((args.catalogue, args.scene, args.eval_report)):
        raise SystemExit("At least one of --catalogue, --scene, or --eval-report is required")

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
