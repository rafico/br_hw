from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SUITE_NAMES = ("unit", "offline", "dataset-smoke", "release-local", "manual-visual-prep")
OFFLINE_TEST_MODULES = (
    "tests.test_qa_tools",
    "tests.test_rerank_and_clustering",
    "tests.test_vlm_scene_and_evaluate",
    "tests.test_rerun_visualizer",
)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]


def _python_cmd(python_bin: str, *args: str) -> tuple[str, ...]:
    return (python_bin, *args)


def build_suite_commands(
        suite: str,
        *,
        python_bin: str | None = None,
        dataset_dir: str = "",
        rerun_save: str = "recording.rrd",
        scene_backend: str = "videomae",
) -> list[CommandSpec]:
    python_bin = python_bin or sys.executable
    root = Path.cwd()
    dataset_dir = str(dataset_dir)

    unit_commands = [
        CommandSpec("unit_tests", _python_cmd(python_bin, "-m", "unittest", "discover", "-s", "tests", "-v")),
    ]

    offline_commands = [
        CommandSpec("offline_unit_tests", _python_cmd(python_bin, "-m", "unittest", "-v", *OFFLINE_TEST_MODULES)),
        CommandSpec("run_help", _python_cmd(python_bin, "run.py", "--help")),
        CommandSpec("finetune_help", _python_cmd(python_bin, "finetune_reid.py", "--help")),
        CommandSpec("seed_ground_truth_help", _python_cmd(python_bin, "scripts/seed_ground_truth.py", "--help")),
        CommandSpec("qa_validate_help", _python_cmd(python_bin, "scripts/qa_validate_outputs.py", "--help")),
        CommandSpec("qa_runner_help", _python_cmd(python_bin, "scripts/run_qa.py", "--help")),
    ]

    if suite == "unit":
        return unit_commands
    if suite == "offline":
        return offline_commands

    if not dataset_dir:
        raise ValueError(f"--dataset-dir is required for suite {suite!r}")

    dataset_smoke_commands = offline_commands + [
        CommandSpec(
            "legacy_pipeline_smoke",
            _python_cmd(
                python_bin,
                "run.py",
                "--dataset-dir",
                dataset_dir,
                "--overwrite-loading",
                "--overwrite-algo",
                "--visualizer",
                "none",
                "--scene-backend",
                scene_backend,
                "--legacy-clustering",
                "--tracker-type",
                "bytetrack",
                "--reid-backbone",
                "osnet_ain",
                "--yolo-model",
                "yolov11",
            ),
        ),
        CommandSpec(
            "legacy_output_validation",
            _python_cmd(
                python_bin,
                "scripts/qa_validate_outputs.py",
                "--catalogue",
                str(root / "catalogue_simple.json"),
                "--scene",
                str(root / "scene_labels.json"),
            ),
        ),
    ]
    if suite == "dataset-smoke":
        return dataset_smoke_commands

    if suite == "manual-visual-prep":
        return [
            CommandSpec(
                "manual_visual_rerun_export",
                _python_cmd(
                    python_bin,
                    "run.py",
                    "--dataset-dir",
                    dataset_dir,
                    "--overwrite-loading",
                    "--overwrite-algo",
                    "--visualizer",
                    "rerun",
                    "--rerun-save",
                    rerun_save,
                    "--scene-backend",
                    scene_backend,
                    "--tracker-type",
                    "botsort",
                    "--reid-backbone",
                    "ensemble",
                    "--use-new-clustering",
                ),
            ),
            CommandSpec(
                "manual_visual_output_validation",
                _python_cmd(
                    python_bin,
                    "scripts/qa_validate_outputs.py",
                    "--catalogue",
                    str(root / "catalogue_v2.json"),
                    "--scene",
                    str(root / "scene_labels_v2.json"),
                ),
            ),
            CommandSpec(
                "manual_visual_review_bundle",
                _python_cmd(
                    python_bin,
                    "scripts/prepare_manual_visual_review.py",
                    "--dataset-dir",
                    dataset_dir,
                    "--catalogue",
                    str(root / "catalogue_v2.json"),
                    "--scene",
                    str(root / "scene_labels_v2.json"),
                    "--rerun-recording",
                    rerun_save,
                    "--manifest-out",
                    str(root / "qa_artifacts" / "manual_visual_review.json"),
                    "--notes-out",
                    str(root / "qa_artifacts" / "manual_visual_review.md"),
                ),
            ),
        ]

    if suite != "release-local":
        raise ValueError(f"Unsupported suite: {suite!r}")

    return dataset_smoke_commands + [
        CommandSpec(
            "stage2_pipeline",
            _python_cmd(
                python_bin,
                "run.py",
                "--dataset-dir",
                dataset_dir,
                "--overwrite-loading",
                "--overwrite-algo",
                "--visualizer",
                "none",
                "--scene-backend",
                scene_backend,
                "--tracker-type",
                "botsort",
                "--reid-backbone",
                "ensemble",
                "--use-new-clustering",
            ),
        ),
        CommandSpec(
            "stage2_output_validation",
            _python_cmd(
                python_bin,
                "scripts/qa_validate_outputs.py",
                "--catalogue",
                str(root / "catalogue_v2.json"),
                "--scene",
                str(root / "scene_labels_v2.json"),
            ),
        ),
        CommandSpec(
            "finetune_release_path",
            _python_cmd(
                python_bin,
                "run.py",
                "--dataset-dir",
                dataset_dir,
                "--overwrite-loading",
                "--overwrite-algo",
                "--visualizer",
                "none",
                "--scene-backend",
                scene_backend,
                "--tracker-type",
                "botsort",
                "--reid-backbone",
                "ensemble",
                "--use-new-clustering",
                "--finetune-reid",
            ),
        ),
        CommandSpec(
            "rerun_release_path",
            _python_cmd(
                python_bin,
                "run.py",
                "--dataset-dir",
                dataset_dir,
                "--overwrite-loading",
                "--overwrite-algo",
                "--visualizer",
                "rerun",
                "--rerun-save",
                rerun_save,
                "--scene-backend",
                scene_backend,
                "--tracker-type",
                "botsort",
                "--reid-backbone",
                "ensemble",
                "--use-new-clustering",
            ),
        ),
        CommandSpec(
            "rerun_output_validation",
            _python_cmd(
                python_bin,
                "scripts/qa_validate_outputs.py",
                "--catalogue",
                str(root / "catalogue_v2.json"),
                "--scene",
                str(root / "scene_labels_v2.json"),
            ),
        ),
    ]


def run_suite(
        suite: str,
        *,
        python_bin: str | None = None,
        dataset_dir: str = "",
        rerun_save: str = "recording.rrd",
        scene_backend: str = "videomae",
        dry_run: bool = False,
) -> int:
    commands = build_suite_commands(
        suite,
        python_bin=python_bin,
        dataset_dir=dataset_dir,
        rerun_save=rerun_save,
        scene_backend=scene_backend,
    )
    for command in commands:
        print(f"[qa] {command.name}")
        print(f"[qa] $ {' '.join(shlex.quote(part) for part in command.argv)}")
        if dry_run:
            continue
        completed = subprocess.run(command.argv, check=False)
        if completed.returncode != 0:
            print(f"[qa] failed: {command.name} (exit={completed.returncode})")
            return int(completed.returncode)
    print(f"[qa] suite passed: {suite}")
    return 0


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run named QA suites for this repository.")
    parser.add_argument("--suite", required=True, choices=SUITE_NAMES, help="QA suite to execute")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for child commands")
    parser.add_argument("--dataset-dir", default="", help="Dataset directory for dataset-backed suites")
    parser.add_argument("--rerun-save", default="recording.rrd", help="Output path used for the Rerun QA run")
    parser.add_argument("--scene-backend", default="videomae", choices=["videomae", "gemini", "internvideo"], help="Scene backend used by dataset-backed QA suites")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_suite(
        args.suite,
        python_bin=args.python,
        dataset_dir=args.dataset_dir,
        rerun_save=args.rerun_save,
        scene_backend=args.scene_backend,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
