#!/usr/bin/env python3
"""Translate measured PPO throughput into per-stage budget durations, not convergence promises."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_curriculum import resolve_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    plan = resolve_plan(json.loads(args.plan.read_text()), lambda name: json.loads((ROOT / name).read_text()))
    capacity = json.loads(args.capacity_report.read_text())
    measured = next(p for p in capacity["probes"] if p["num_envs"] == args.num_envs and p["status"] == "completed")
    update_seconds = measured["update_seconds"]
    startup = max(0., measured["wall_seconds"] - measured["successful_updates"] * update_seconds)
    rows = []
    for recipe in plan["stages"]:
        updates = math.ceil(recipe["updates"] * plan["target_num_envs"] / args.num_envs)
        blocks = math.ceil(updates / plan["block_updates"])
        complex_scene = recipe["kind"] in ("terrain", "jump", "mixed")
        # The actual first flat evaluations took about 49-69 seconds.
        evaluation_low, evaluation_high = (60., 150.) if complex_scene else (50., 90.)
        factor = 2. if complex_scene else 1.25
        low = updates * update_seconds + blocks * (startup + evaluation_low) + 2 * evaluation_low
        high = updates * update_seconds * factor + blocks * (startup * 1.25 + evaluation_high) + 2 * evaluation_high
        rows.append({"stage": recipe["name"], "kind": recipe["kind"], "updates_budget": updates,
                     "blocks_max": blocks, "ppo_only_minutes": updates * update_seconds / 60,
                     "budget_minutes_low": low / 60, "budget_minutes_high": high / 60})
    report = {"scope": "time_to_exhaust_each_stage_budget_not_time_to_converge", "num_envs": args.num_envs,
              "measured_update_seconds": update_seconds, "estimated_process_overhead_seconds": startup,
              "runtime_limit_hours": plan["max_runtime_seconds"] / 3600,
              "total_budget_hours_low": sum(r["budget_minutes_low"] for r in rows) / 60,
              "total_budget_hours_high": sum(r["budget_minutes_high"] for r in rows) / 60,
              "stages": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        lines = ["# SCUT35 各阶段预算耗时估算", "",
                 "这是按更新上限跑满的估算，不是收敛时间预测。通过门槛可提前结束；回归保护也可能提前暂停。", "",
                 f"实测 {args.num_envs} 环境，约 {update_seconds:.2f}s/update；每次进程初始化/导出约 {startup:.0f}s。",
                 "已计入每批重新初始化、独立评测和最终确认；复杂场景以更宽范围估算。", "",
                 "| 阶段 | 更新上限 | 含评测预算耗时 |", "|---|---:|---:|"]
        lines += [f"| {r['stage']} | {r['updates_budget']} | {r['budget_minutes_low']:.0f}–{r['budget_minutes_high']:.0f} min |" for r in rows]
        lines += ["", f"所有阶段均跑满：约 **{report['total_budget_hours_low']:.1f}–{report['total_budget_hours_high']:.1f}h**。",
                  "整体墙钟预算为 72h；未完成阶段不会被标为通过。首次模型复用和提前验收会显著缩短实际时间。"]
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "stages"}, indent=2))


if __name__ == "__main__":
    main()
