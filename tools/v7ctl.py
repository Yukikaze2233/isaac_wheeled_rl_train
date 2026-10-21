#!/usr/bin/env python3
"""Command entry point; implementation lives in the application package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wheeled_tasks.chassis.application.console import main


if __name__ == "__main__":
    main()
