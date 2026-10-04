"""Write diagnostic scalars from the immutable physical-prefix report; no server or PPO."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
args = parser.parse_args()
report = json.loads(args.report.read_text())
assert report["execution_status"] == "completed"
assert report["formal_updates_consumed"] == report["optimizer_steps"] == 0
assert report["actor_tensors_unchanged"] and report["checkpoint_unchanged"]
assert report["qualification"] is False
summary = report["summary"]
out = args.report.parent / "tensorboard"
out.mkdir(exist_ok=False)
writer = SummaryWriter(str(out))
writer.add_text("RecoveryPrefix/Scope", "Fixed four first-birth cohorts, seed 290929, old flat parent actor, nominal delay; diagnostic only. Step 0 is one completed probe, not PPO iteration.", 0)
writer.add_scalar("RecoveryPrefix/first_birth_denominator", summary["fixed_birth_denominator"], 0)
for name, count in summary["partitions"].items():
    writer.add_scalar(f"RecoveryPrefix/partition/{name}_count", count, 0)
for name, item in summary["events_per_seeded_cohort"].items():
    writer.add_scalar(f"RecoveryPrefix/event/{name}_fraction", item["fraction"], 0)
writer.add_scalar("RecoveryPrefix/actual_physics_exposure_s", report["elapsed_physics_s"], 0)
writer.add_scalar("RecoveryPrefix/probe_wall_seconds", report["wall_seconds"], 0)
for row in summary["rows"]:
    base = f"RecoveryPrefix/case/{row['case']}"
    writer.add_scalar(base + "/success", int(row["partition"] == "success"), 0)
    writer.add_scalar(base + "/terminal_exposure_s", row["events"]["terminal_pre_autoreset"], 0)
    writer.add_scalar(base + "/reached_actual_RL", int("first_pure_rl_post_physics" in row["events"]), 0)
    writer.add_scalar(base + "/actual_blend_ticks", sum(item["ticks"] for item in summary["native_control_ownership_ticks"] if item["lane"] == row["lane"] and item["ownership"] == "script_actor_blend"), 0)
writer.flush()
writer.close()
files = {str(path.relative_to(args.report.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir()) if path.is_file()}
receipt = {"source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(), "event_files_sha256": files, "global_step": 0, "scope": "fixed_four_case_diagnostic_not_training_curve_or_qualification", "local_tensorboard_server_started": False}
(out / "writer_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
