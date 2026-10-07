"""Completed episodes keep an exact, lesson-specific TensorBoard denominator."""
from types import SimpleNamespace

import torch

from train_chassis import log_recovery_outcomes
from wheeled_tasks.chassis.recovery_prefix_curriculum import RecoveryPrefixCurriculum


def sampler(counts):
    result = RecoveryPrefixCurriculum.__new__(RecoveryPrefixCurriculum)
    result.outcome_counts = torch.tensor(counts, dtype=torch.long)
    return result


def test_rollout_endpoint_is_not_a_frame_mean_or_mixed_lesson_rate():
    curriculum = sampler([[100, 90, 2, 8], [20, 8, 3, 9], [7, 1, 2, 4]])
    before = curriculum.outcome_counts.clone()
    values = curriculum.rollout_outcome_metrics()
    assert values["RecoveryOutcomes/original_chain/ended"] == 7
    assert values["RecoveryOutcomes/original_chain/successes"] == 1
    assert values["RecoveryOutcomes/original_chain/cumulative_success_fraction"] == 1 / 7
    assert values["RecoveryOutcomes/prefix_stand/cumulative_success_fraction"] == .9
    assert torch.equal(curriculum.outcome_counts, before)
    curriculum.outcome_counts[2] += torch.tensor([2, 1, 0, 1])
    next_values = curriculum.rollout_outcome_metrics()
    assert next_values["RecoveryOutcomes/original_chain/ended"] - values["RecoveryOutcomes/original_chain/ended"] == 2
    assert next_values["RecoveryOutcomes/original_chain/successes"] - values["RecoveryOutcomes/original_chain/successes"] == 1


def test_zero_completed_episodes_cannot_report_zero_success_rate():
    values = sampler([[0, 0, 0, 0]] * 3).rollout_outcome_metrics()
    assert values["RecoveryOutcomes/original_chain/has_completed_episode"] == 0
    assert values["RecoveryOutcomes/original_chain/ended"] == 0
    assert "RecoveryOutcomes/original_chain/cumulative_success_fraction" not in values


def test_writer_uses_one_post_rollout_snapshot_and_the_real_step():
    curriculum = sampler([[2, 2, 0, 0], [4, 1, 1, 2], [3, 1, 1, 1]])
    written = []
    writer = SimpleNamespace(add_scalar=lambda *args: written.append(args))
    log_recovery_outcomes(writer, SimpleNamespace(pose_curriculum=curriculum), 38200)
    assert ("RecoveryOutcomes/original_chain/ended", 3, 38200) in written
    assert len({tag for tag, _, _ in written}) == len(written) == 18
    assert all(step == 38200 for _, _, step in written)


def test_disabled_writer_and_original_fixed_curriculum_do_not_read_or_mutate():
    def reject():
        raise AssertionError("disabled logging must not read a sampler")
    log_recovery_outcomes(None, SimpleNamespace(pose_curriculum=SimpleNamespace(rollout_outcome_metrics=reject)), 1)
    log_recovery_outcomes(SimpleNamespace(add_scalar=reject), None, 1)
    log_recovery_outcomes(SimpleNamespace(add_scalar=reject), SimpleNamespace(pose_curriculum=object()), 1)
