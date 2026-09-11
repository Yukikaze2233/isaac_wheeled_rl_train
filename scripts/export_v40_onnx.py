#!/usr/bin/env python3
"""Export a metadata-bound V4.0 FrameStack ActorCritic; never train or run physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Stock RSL-RL checkpoint with primitive infos")
    parser.add_argument("--run-manifest", required=True, type=Path, help="Version-1 run manifest JSON")
    parser.add_argument("--output", required=True, type=Path, help="New .onnx file (also creates .onnx.json; no overwrite)")
    args = parser.parse_args(argv)
    try:
        from wheeled_algo.v40_export import export_checkpoint, sidecar_path

        report = export_checkpoint(args.checkpoint, args.run_manifest, args.output)
    except Exception as exc:
        # No fallback to unsafe unpickling, unverified output, or the legacy exporter.
        parser.exit(1, f"V4.0 export rejected: {exc}\n")
    print(json.dumps({
        "onnx": str(args.output), "sidecar": str(sidecar_path(args.output)),
        "contract_id": report["contract_id"], "onnx_sha256": report["onnx_sha256"],
        "validation_samples": report["validation"]["sample_count"],
        "max_abs_error": report["validation"]["max_abs_error"],
        "note": "numerical export validation only; not evidence of policy quality",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
