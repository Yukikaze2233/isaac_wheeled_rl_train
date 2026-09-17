#!/usr/bin/env python3
"""Verify completed-update lineage and optimizer step advancement in trusted run artifacts."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = torch.load(args.before, map_location="cpu", weights_only=True)
    after = torch.load(args.after, map_location="cpu", weights_only=True)
    for key in ("contract_id", "contract_sha256", "asset_manifest_sha256", "control_math_sha256"):
        if before["infos"][key] != after["infos"][key]:
            raise ValueError(f"Resume identity mismatch: {key}")
    optimizer_keys = [k for k in before if "optimizer" in k and isinstance(before[k], dict) and "state" in before[k]]
    if not optimizer_keys:
        raise ValueError("Missing optimizer states")
    steps = []
    for key in optimizer_keys:
        a, b = before[key]["state"], after[key]["state"]
        if set(a) != set(b):
            raise ValueError("Optimizer parameter mapping changed")
        for name, state in a.items():
            if "step" in state:
                old, new = int(state["step"]), int(b[name]["step"])
                if new <= old:
                    raise ValueError("Optimizer did not continue")
                steps.append({"optimizer": key, "parameter": name, "before": old, "after": new})
    if not steps or after["infos"]["successful_updates_total"] <= before["infos"]["successful_updates_total"]:
        raise ValueError("No successful resumed updates")
    report = {"passed": True, "before_infos": before["infos"], "after_infos": after["infos"], "optimizer_steps": steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
