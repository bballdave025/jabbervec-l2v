"""Tiny scaffolder reminder.

This repo is already scaffolded. Keep this script as a placeholder for future
project-template experiments rather than as required machinery.
"""

from pathlib import Path


def main() -> None:
    for path in ["data/raw", "data/processed", "runs", "notebooks"]:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"ensured {path}")


if __name__ == "__main__":
    main()
