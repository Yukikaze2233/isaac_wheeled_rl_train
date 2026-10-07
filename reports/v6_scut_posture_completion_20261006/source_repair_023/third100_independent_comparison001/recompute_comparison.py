"""Recompute three saved, disjoint native windows; no simulator or SSH."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
REUSED = SOURCE / "first100_independent_comparison001/recompute_comparison.py"
spec = importlib.util.spec_from_file_location("native_window_arithmetic", REUSED)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def main():
    inputs = {
        "first100": ("first100_window_023.read003.json", [6436, 6535], [33640, 33739],
                     "f45e3bb5995f165c5655feb13d4152fe769a24b58e9e452e7d8aa4d3d9451383"),
        "second100": ("second100_window_023.read001.json", [6536, 6635], [33740, 33839],
                      "beb19c1e9f0b37b92e17b7c5a8d74900179ce0023e2ef140bf03bd6ee9dbdc12"),
        "third100": ("third100_window_023.read001.json", [6636, 6735], [33840, 33939],
                     "7efc654d0861d4dbabefef8a817df3950a93c272088440f78b315c1557254fde"),
    }
    snapshots = {}
    for label, (name, rec, tb, sha) in inputs.items():
        path = SOURCE / name
        assert reader.sha(path) == sha
        snapshots[label] = reader.audit(label, path, "100", rec, tb)
        value = snapshots[label]
        assert value["source_commit"] == "6f07ec298e01d2686216787cfba34b394151f1aa"
        assert "/attempt_009/train/" in value["reported_crc_evidence"]["file"]
    values = list(snapshots.values())
    for previous, current in zip(values, values[1:]):
        assert previous["clock"]["end_update"] == current["clock"]["baseline_update"]
        assert previous["clock"]["end_step"] + 1 == current["clock"]["start_step"]
        assert previous["clock"]["task_actor_updates_inclusive"][1] + 1 == current["clock"]["task_actor_updates_inclusive"][0]
    comparison = dict(scope="three_nonoverlapping_sampled_training_windows_not_paired_policy_qualification",
                      snapshots=snapshots,
                      third_minus_second=reader.paired(values[2], values[1]),
                      third_minus_first=reader.paired(values[2], values[0]))
    review = dict(status="saved_native_report_arithmetic_consistent", complete=True,
                  input_sha256={label: row[3] for label, row in inputs.items()},
                  reused_arithmetic_sha256=reader.sha(REUSED),
                  group_count=95, pool_count=7, physical_frames_per_window=9830400,
                  actor_updates_per_window=100, overlap_actor_updates=0,
                  third_REC_window=[6636, 6735], third_TB_window=[33840, 33939],
                  third_recovery_counts=values[2]["arithmetic"]["pool_counts"]["scripted_recovery"],
                  L2_scope="instantaneous_population_success_not_ended_episode_rate",
                  recovery_pool_scope="54_static_groups_mix_L0_L1_L2",
                  comparison_scope="stochastic_curriculum_DR_exposure_not_causal_entropy_AB",
                  SSH_calls=0, GPU_calls=0, PPO_updates=0, qualification=None)
    for label, row in inputs.items():
        assert reader.sha(SOURCE / row[0]) == row[3]
    for name, value in (("comparison.json", comparison), ("review.json", review)):
        (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(review, ensure_ascii=False))
    print(json.dumps({name: snapshots[name] for name in ()}))


if __name__ == "__main__":
    main()
