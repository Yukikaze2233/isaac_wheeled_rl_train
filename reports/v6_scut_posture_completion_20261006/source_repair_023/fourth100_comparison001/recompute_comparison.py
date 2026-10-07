"""Recompute four disjoint saved windows using the reviewed native arithmetic."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
REUSED = SOURCE / "first100_independent_comparison001/recompute_comparison.py"
spec = importlib.util.spec_from_file_location("native_window_arithmetic", REUSED)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

INPUTS = {
    "first100": ("first100_window_023.read003.json", [6436, 6535], [33640, 33739],
                 "f45e3bb5995f165c5655feb13d4152fe769a24b58e9e452e7d8aa4d3d9451383"),
    "second100": ("second100_window_023.read001.json", [6536, 6635], [33740, 33839],
                  "beb19c1e9f0b37b92e17b7c5a8d74900179ce0023e2ef140bf03bd6ee9dbdc12"),
    "third100": ("third100_window_023.read001.json", [6636, 6735], [33840, 33939],
                 "7efc654d0861d4dbabefef8a817df3950a93c272088440f78b315c1557254fde"),
    "fourth100": ("fourth100_window_023.read001.json", [6736, 6835], [33940, 34039],
                  "42ea53450693bdfbf6caa7b150713b8a26bdee7df022323ce3462d133ddbb619"),
}


def main():
    snapshots = {}
    for name, (filename, rec, tb, expected) in INPUTS.items():
        path = SOURCE / filename
        assert reader.sha(path) == expected
        snapshots[name] = reader.audit(name, path, "100", rec, tb)
        assert snapshots[name]["source_commit"] == "6f07ec298e01d2686216787cfba34b394151f1aa"
        assert "/attempt_009/train/" in snapshots[name]["reported_crc_evidence"]["file"]
    rows = list(snapshots.values())
    for before, after in zip(rows, rows[1:]):
        assert before["clock"]["end_update"] == after["clock"]["baseline_update"]
        assert before["clock"]["end_step"] + 1 == after["clock"]["start_step"]
        assert before["clock"]["task_actor_updates_inclusive"][1] + 1 == after["clock"]["task_actor_updates_inclusive"][0]
    comparison = {
        "scope": "four_disjoint_sampled_training_windows_not_paired_policy_qualification",
        "snapshots": snapshots,
        "fourth_minus_third": reader.paired(rows[3], rows[2]),
        "fourth_minus_first": reader.paired(rows[3], rows[0]),
    }
    review = {
        "status": "saved_native_report_arithmetic_consistent",
        "review_scope": "Root recomputation reusing independently reviewed arithmetic; no fresh independent review claimed",
        "complete": True,
        "input_sha256": {name: row[3] for name, row in INPUTS.items()},
        "reused_arithmetic_sha256": reader.sha(REUSED),
        "group_count": 95, "pool_count": 7,
        "physical_frames_per_window": 9830400, "actor_updates_per_window": 100,
        "overlap_actor_updates": 0,
        "fourth_REC_window": [6736, 6835], "fourth_TB_window": [33940, 34039],
        "fourth_recovery_counts": rows[3]["arithmetic"]["pool_counts"]["scripted_recovery"],
        "L2_scope": "instantaneous_population_success_not_ended_episode_rate",
        "recovery_pool_scope": "54_static_groups_mix_L0_L1_L2",
        "comparison_scope": "stochastic_curriculum_DR_exposure_not_causal_entropy_AB",
        "SSH_calls": 0, "GPU_calls": 0, "PPO_updates": 0, "qualification": None,
    }
    for name, row in INPUTS.items():
        assert reader.sha(SOURCE / row[0]) == row[3]
    for filename, value in (("comparison.json", comparison), ("review.json", review)):
        (HERE / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    latest = rows[3]
    print(json.dumps({"review": review, "stand": latest["ordinary_stand"],
                     "motion": latest["motion"], "yaw": latest["yaw_mae_rad_s"],
                     "stop": latest["conditional_stop_phases"],
                     "hold": latest["hold_gate_means"], "std": latest["policy_mean_std"],
                     "wall": latest["iteration_timing"]["iteration_wall_seconds"],
                     "valid_fraction": latest["learning_ownership"]["valid_fraction"],
                     "L2_success": latest["full_chain_success"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
