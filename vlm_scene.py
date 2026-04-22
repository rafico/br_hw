from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)
CRIME_TAXONOMY = [
    "assault",
    "robbery",
    "theft",
    "pickpocketing",
    "vandalism",
    "weapon display",
    "public fighting",
    "kidnapping",
]


def _sample_fps(sample) -> float:
    metadata = getattr(sample, "metadata", None)
    frame_rate = getattr(metadata, "frame_rate", None)
    if frame_rate:
        return float(frame_rate)
    try:
        import cv2
    except ImportError:
        return 30.0

    cap = cv2.VideoCapture(sample.filepath)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 1e-6 else 30.0


def build_person_presence(catalogue: dict, fps_map: Dict[str, float]) -> dict:
    presence: Dict[str, List[dict]] = {}
    for global_id, appearances in catalogue.items():
        gid = int(global_id)
        for appearance in appearances:
            clip_id = str(appearance["clip_id"])
            fps = max(float(fps_map.get(clip_id, 30.0)), 1e-6)
            for start_frame, end_frame in appearance.get("frame_ranges", []):
                presence.setdefault(clip_id, []).append(
                    {
                        "global_id": gid,
                        "t_start_sec": max(float(start_frame - 1), 0.0) / fps,
                        "t_end_sec": max(float(end_frame - 1), 0.0) / fps,
                    }
                )

    for clip_id in presence:
        presence[clip_id].sort(key=lambda item: (item["t_start_sec"], item["global_id"]))
    return presence


def resolve_persons_for_event(event: dict, presence: dict, clip_id: str) -> List[int]:
    clip_presence = presence.get(str(clip_id), [])
    t_start = float(event.get("t_start_sec", 0.0))
    t_end = float(event.get("t_end_sec", t_start))
    overlapping = {
        int(item["global_id"])
        for item in clip_presence
        if item["t_start_sec"] <= t_end and t_start <= item["t_end_sec"]
    }
    proposed = {int(pid) for pid in event.get("global_person_ids", []) if int(pid) in overlapping}
    return sorted(proposed or overlapping)


def justify(events: List[dict]) -> str:
    if not events:
        return "No criminal activity detected in the clip."

    parts = []
    for event in events:
        ids = event.get("global_person_ids") or []
        people = f" involving person {', '.join(str(pid) for pid in ids)}" if ids else ""
        parts.append(
            f"{event.get('type', 'event')} at {event.get('t_start_sec', 0.0):.1f}s-{event.get('t_end_sec', 0.0):.1f}s{people}"
        )
    return "; ".join(parts)


def _validate_scene_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Scene payload must be a JSON object")

    label = str(payload.get("label", "")).strip()
    if label not in {"normal", "crime"}:
        raise ValueError("label must be 'normal' or 'crime'")

    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    events = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            raise ValueError("event entries must be objects")
        start = float(event.get("t_start_sec", 0.0))
        end = float(event.get("t_end_sec", start))
        if start < 0.0 or end < 0.0:
            raise ValueError("event timestamps must be non-negative")
        ids = [int(pid) for pid in event.get("global_person_ids", [])]
        events.append(
            {
                "type": str(event.get("type", "event")),
                "t_start_sec": start,
                "t_end_sec": end,
                "global_person_ids": ids,
                "evidence": str(event.get("evidence", "")),
            }
        )

    return {
        "label": label,
        "confidence": confidence,
        "events": events,
        "rationale": str(payload.get("rationale", "")),
    }


def _call_gemini(
    video_path: str,
    presence_hint: dict,
    crime_taxonomy: List[str],
    model: str = "gemini-2.5-flash",
) -> dict:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("google-generativeai is not installed") from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    uploaded = genai.upload_file(path=video_path)
    while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
        time.sleep(2)
        uploaded = genai.get_file(uploaded.name)

    state_name = getattr(uploaded.state, "name", str(uploaded.state))
    if state_name != "ACTIVE":
        raise RuntimeError(f"Gemini upload failed for {video_path}: state={state_name}")

    system_instruction = (
        "You are a surveillance-clip classifier. Output strict JSON. "
        f"Crime taxonomy: {', '.join(crime_taxonomy)}."
    )
    user_prompt = (
        "Classify the attached clip. Known persons present (by tracked global ID with time spans): "
        f"{json.dumps(presence_hint)}. If crime, list events tying types to global_person_ids and time spans."
    )
    strict_prompt = user_prompt + " Return only valid JSON matching the requested schema."

    generator = genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }

    last_error = None
    for prompt in (user_prompt, strict_prompt):
        response = generator.generate_content(
            [prompt, uploaded],
            generation_config=generation_config,
        )
        try:
            payload = json.loads(response.text)
            return _validate_scene_payload(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"Gemini returned invalid schema for {video_path}: {last_error}")


def _call_internvideo2(video_path: str, presence_hint: dict, crime_taxonomy: List[str]) -> dict:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is not installed") from exc

    try:
        AutoModel.from_pretrained(
            "OpenGVLab/InternVideo2-stage2_1b-224p-f4",
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(f"InternVideo2 backend is unavailable: {exc}") from exc

    raise RuntimeError("InternVideo2 inference wiring is not available in this environment.")


def classify_scenes_vlm(
    dataset,
    catalogue_path: str = "catalogue_v2.json",
    output_file: str = "scene_labels_v2.json",
    backend: str = "gemini",
    model: str = "gemini-2.5-flash",
) -> list:
    try:
        with open(catalogue_path, "r", encoding="utf-8") as f:
            catalogue_payload = json.load(f)
    except FileNotFoundError:
        catalogue_payload = {"catalogue": {}}

    fps_map = {
        Path(sample.filepath).stem: _sample_fps(sample)
        for sample in dataset
    }
    presence = build_person_presence(catalogue_payload.get("catalogue", {}), fps_map)

    results = []
    for sample in dataset:
        clip_id = Path(sample.filepath).stem
        if backend == "gemini":
            raw = _call_gemini(
                sample.filepath,
                presence_hint=presence.get(clip_id, []),
                crime_taxonomy=CRIME_TAXONOMY,
                model=model,
            )
        elif backend == "internvideo":
            raw = _call_internvideo2(
                sample.filepath,
                presence_hint=presence.get(clip_id, []),
                crime_taxonomy=CRIME_TAXONOMY,
            )
        else:
            raise ValueError(f"Unsupported scene backend: {backend}")

        validated = _validate_scene_payload(raw)
        crime_segments = []
        for event in validated.get("events", []):
            ids = resolve_persons_for_event(event, presence, clip_id)
            crime_segments.append(
                {
                    "type": event["type"],
                    "timestamp_start": round(float(event["t_start_sec"]), 2),
                    "timestamp_end": round(float(event["t_end_sec"]), 2),
                    "involved_people_global": ids,
                    "evidence": event.get("evidence", ""),
                }
            )

        final_label = "crime" if crime_segments else "normal"
        if final_label != validated["label"]:
            LOGGER.warning(
                "Normalized scene label for clip %s from %s to %s based on %d event(s)",
                clip_id,
                validated["label"],
                final_label,
                len(crime_segments),
            )

        justification_events = [
            {
                "type": segment["type"],
                "t_start_sec": segment["timestamp_start"],
                "t_end_sec": segment["timestamp_end"],
                "global_person_ids": segment["involved_people_global"],
            }
            for segment in crime_segments
        ]

        results.append(
            {
                "clip_id": clip_id,
                "label": final_label,
                "justification": justify(justification_events),
                "rationale": validated.get("rationale", ""),
                "max_confidence": round(float(validated["confidence"]), 4),
                "crime_segments": crime_segments if final_label == "crime" else [],
            }
        )

    results.sort(key=lambda item: item["clip_id"])
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Scene labels saved to %s", Path(output_file).resolve())
    return results
