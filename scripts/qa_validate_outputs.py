from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_root_on_path()
    from qa.output_validation import main as output_validation_main

    return output_validation_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
