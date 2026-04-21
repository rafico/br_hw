from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    {
        "name": "baseline",
        "args": ["--legacy-clustering", "--tracker-type", "bytetrack", "--reid-backbone", "osnet_ain", "--scene-backend", "videomae"],
    },
    {
        "name": "stage1",
        "args": ["--legacy-clustering", "--tracker-type", "botsort", "--reid-backbone", "ensemble", "--use-rerank", "--cooccurrence-constraint", "--scene-backend", "videomae"],
    },
    {
        "name": "stage2",
        "args": ["--use-new-clustering", "--tracker-type", "botsort", "--reid-backbone", "ensemble", "--scene-backend", "gemini", "--evaluate"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablations over configured pipeline stages")
    parser.add_argument("--dataset-dir", required=True)
    return parser.parse_args()


def load_eval_report(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_config(name: str, dataset_dir: str, extra_args: list[str]) -> dict:
    cmd = [sys.executable, "run.py", "--dataset-dir", dataset_dir, "--overwrite-algo", *extra_args]
    completed = subprocess.run(cmd, check=False)
    report = load_eval_report(Path("eval_report.json"))
    return {
        "name": name,
        "returncode": completed.returncode,
        "report": report,
    }


def write_report(results: list[dict], output_path: Path = Path("ablation_report.md")) -> None:
    lines = [
        "| Stage | Return Code | V-measure | ARI | Purity | Scene Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        report = result.get("report", {})
        person = report.get("person_reid", {})
        scene = report.get("scene", {})
        lines.append(
            "| {name} | {returncode} | {v:.4f} | {ari:.4f} | {purity:.4f} | {scene_acc:.4f} |".format(
                name=result["name"],
                returncode=result["returncode"],
                v=float(person.get("v_measure", 0.0)),
                ari=float(person.get("adjusted_rand_index", 0.0)),
                purity=float(person.get("purity", 0.0)),
                scene_acc=float(scene.get("accuracy", 0.0)),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    results = [
        run_config(item["name"], args.dataset_dir, item["args"])
        for item in ABLATIONS
    ]
    write_report(results)


if __name__ == "__main__":
    main()
