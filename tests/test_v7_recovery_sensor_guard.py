"""Focused timestamp admission checks; no simulator or hardware qualification."""
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from wheeled_tasks.chassis.recovery_sensor_guard import RecoverySensorGuard, RecoverySensorIssue


def guard(count=2, **overrides):
    settings = dict(feedback_max_age_s=.02, acceleration_max_age_s=.03, sensor_max_skew_s=.01)
    return RecoverySensorGuard(count, "cpu", **(settings | overrides))


def samples(count=2, sequence=1, stamp=.09):
    return (torch.full((count, 8), sequence, dtype=torch.long),
            torch.full((count, 8), stamp, dtype=torch.float64))


def test_zero_tick_is_missing_and_repeat_expires_without_committing():
    owner = guard()
    seq, stamp = samples(sequence=0, stamp=0.)
    status = owner.update(0., seq, stamp)
    assert status.valid.tolist() == [False, False]
    assert status.issue.tolist() == [RecoverySensorIssue.MISSING] * 2
    assert status.invalid_mask.tolist() == [0xff] * 2
    assert status.maximum_motor_age_ms.tolist() == [-1.] * 2
    seq, stamp = samples()
    assert owner.update(.1, seq, stamp).valid.all()
    assert owner.update(.105, seq, stamp).valid.all()
    expired = owner.update(.115, seq, stamp)
    assert expired.issue.tolist() == [RecoverySensorIssue.EXPIRED] * 2
    assert expired.invalid_mask.tolist() == [0x7f] * 2
    torch.testing.assert_close(owner._previous_sequences, seq)
    torch.testing.assert_close(owner._previous_stamps_ns, torch.round(stamp * 1e9).long())


def test_first_failed_channel_issue_and_all_invalid_bits():
    owner = guard(3)
    seq, stamp = samples(3, sequence=10)
    owner.update(.1, seq, stamp)
    seq[0, 0], stamp[0, 0] = 1, .075
    seq[0, 7], stamp[0, 7] = 0, 0.
    stamp[1, 2], seq[1, 3] = .11, 0
    seq[2, 0], stamp[2, 0] = 9, .085
    stamp[2, 1], seq[2, 2] = .095, 11
    status = owner.update(.1, seq, stamp)
    assert status.issue.tolist() == [RecoverySensorIssue.EXPIRED,
                                     RecoverySensorIssue.FUTURE,
                                     RecoverySensorIssue.REVERSED]
    assert status.invalid_mask.tolist() == [0x81, 0x0c, 0x07]
    assert status.channel_age_s[0, 7].item() == -.001
    assert status.channel_age_s[1, 2].item() == -.001


def test_invalid_row_does_not_commit_any_channel_and_other_row_commits():
    owner = guard()
    seq, stamp = samples()
    owner.update(.1, seq, stamp)
    new_seq, new_stamp = samples(sequence=2, stamp=.095)
    new_seq[0, 7] = 1
    status = owner.update(.105, new_seq, new_stamp)
    assert status.issue.tolist() == [RecoverySensorIssue.INCONSISTENT_SEQUENCE, 0]
    torch.testing.assert_close(owner._previous_sequences[0], seq[0])
    torch.testing.assert_close(owner._previous_stamps_ns[0], torch.round(stamp[0] * 1e9).long())
    torch.testing.assert_close(owner._previous_sequences[1], new_seq[1])
    torch.testing.assert_close(owner._previous_stamps_ns[1], torch.round(new_stamp[1] * 1e9).long())
    new_seq[0], new_stamp[0] = seq[0], stamp[0]
    assert owner.update(.106, new_seq, new_stamp).valid.all()


def test_skew_marks_full_row_and_exact_binary_limits_pass():
    owner = guard(1)
    seq, stamp = samples(1)
    stamp[0, 7] = .079
    status = owner.update(.1, seq, stamp)
    assert status.issue.item() == RecoverySensorIssue.SKEW
    assert status.invalid_mask.item() == 0xff
    assert not owner._previous_sequences.any()
    owner = guard(1, feedback_max_age_s=.03125,
                  acceleration_max_age_s=.046875, sensor_max_skew_s=.015625)
    seq, stamp = samples(1, stamp=.03125)
    stamp[0, 7] = .015625
    status = owner.update(.0625, seq, stamp)
    assert status.valid.item()
    assert status.maximum_motor_age_ms.item() == 31.25
    assert status.acceleration_age_ms.item() == 46.875


def test_reset_only_selected_lane_and_guard_does_not_own_input_tensors():
    owner = guard()
    seq, stamp = samples(sequence=5, stamp=.1)
    owner.update(.11, seq, stamp)
    seq.zero_()
    stamp.zero_()
    assert owner._previous_sequences.eq(5).all()
    assert owner._previous_stamps_ns.eq(100_000_000).all()
    owner.reset(torch.tensor([1], dtype=torch.long))
    owner.reset(torch.empty(0, dtype=torch.long))
    seq, stamp = samples(sequence=1, stamp=.09)
    status = owner.update(torch.tensor([.1, .1], dtype=torch.float64), seq, stamp)
    assert status.issue.tolist() == [RecoverySensorIssue.REVERSED, 0]
    assert owner._previous_sequences[0].eq(5).all()
    assert owner._previous_sequences[1].eq(1).all()


@pytest.mark.parametrize("settings", [dict(feedback_max_age_s=0.),
    dict(acceleration_max_age_s=float("nan")), dict(sensor_max_skew_s=.051),
    dict(feedback_max_age_s=True)])
def test_limits_match_deployment_guard(settings):
    with pytest.raises(ValueError, match="limits"):
        guard(**settings)


def test_metadata_uses_integer_sequence_and_float64_second_clock():
    owner = guard()
    seq, stamp = samples()
    with pytest.raises(ValueError, match="sequences"):
        owner.update(.1, seq.float(), stamp)
    with pytest.raises(ValueError, match="stamps"):
        owner.update(.1, seq, stamp.float())
    with pytest.raises(ValueError, match="clock"):
        owner.update(torch.tensor(.1), seq, stamp)
    with pytest.raises(ValueError, match="reset"):
        owner.reset(torch.tensor([0.]))


def test_subnanosecond_adapter_uses_rounded_stamp_for_repeat_identity():
    owner = guard(1)
    seq, stamp = samples(1)
    assert owner.update(.1, seq, stamp).valid.item()
    stamp += .1e-9
    assert owner.update(.101, seq, stamp).valid.item()
    stamp += .6e-9
    status = owner.update(.102, seq, stamp)
    assert status.issue.item() == RecoverySensorIssue.INCONSISTENT_SEQUENCE
    assert status.invalid_mask.item() == 0xff
    assert owner._previous_stamps_ns.eq(90_000_000).all()


def test_vectors_match_actual_cpp_header(tmp_path):
    """Compare against the frozen deployment header, independently of its checkout."""
    header = (Path(__file__).resolve().parents[1] / "references"
              / "rmcs_recovery_sensor_guard_6104e765" / "recovery_sensor_guard.hpp")
    assert header.is_file(), "The frozen deployment sensor guard must be present"
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("The C++ compiler is unavailable")
    source = tmp_path / "sensor_guard.cpp"
    source.write_text('''#include "recovery_sensor_guard.hpp"
#include <iomanip>
#include <iostream>
int main() {
    std::array<rmcs::rl::RecoverySensorGuard, 2> guards{};
    char operation; std::size_t lane;
    std::cout << std::setprecision(17);
    while (std::cin >> operation >> lane) {
        if (operation == 'R') { guards.at(lane).reset(); continue; }
        std::uint64_t now;
        std::array<rmcs::rl::RecoverySampleStamp, 8> samples;
        std::cin >> now;
        for (auto& sample : samples) std::cin >> sample.sequence >> sample.steady_ns;
        const auto status = guards.at(lane).update(now, samples);
        std::cout << status.valid << ' ' << int(status.issue) << ' ' << status.invalid_mask
                  << ' ' << status.maximum_motor_age_ms << ' ' << status.imu_age_ms
                  << ' ' << status.acceleration_age_ms << '\\n';
    }
}
''')
    executable = tmp_path / "sensor_guard"
    subprocess.run([compiler, "-std=c++17", "-I", str(header.parent), str(source),
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    owner = guard()
    commands, expected = [], []

    def step(now_ns, seq, stamps_ns):
        status = owner.update(now_ns * 1e-9, seq, stamps_ns.double() * 1e-9)
        for lane in range(2):
            metadata = " ".join(f"{s} {t}" for s, t in zip(seq[lane].tolist(), stamps_ns[lane].tolist()))
            commands.append(f"U {lane} {now_ns} {metadata}")
            expected.append((int(status.valid[lane]), int(status.issue[lane]),
                             int(status.invalid_mask[lane]), float(status.maximum_motor_age_ms[lane]),
                             float(status.imu_age_ms[lane]), float(status.acceleration_age_ms[lane])))

    def metadata(sequence, stamp_ns):
        return (torch.full((2, 8), sequence, dtype=torch.long),
                torch.full((2, 8), stamp_ns, dtype=torch.long))

    step(0, *metadata(0, 0))
    step(100_000_000, *metadata(1, 90_000_000))
    step(105_000_000, *metadata(1, 90_000_000))
    seq, stamps = metadata(2, 100_000_000)
    stamps[0] = 95_000_000
    seq[0, 7] = 1
    step(110_000_000, seq, stamps)
    seq, stamps = metadata(2, 100_000_000)
    stamps[0] = 96_000_000
    step(111_000_000, seq, stamps)
    seq, stamps = metadata(2, 100_000_000)
    seq[0] = 3
    stamps[0] = 106_000_000
    stamps[0, 0] = 75_000_000
    seq[0, 7], stamps[0, 7] = 0, 0
    stamps[1, 2] = 113_000_000
    step(112_000_000, seq, stamps)
    owner.reset(torch.tensor([1], dtype=torch.long))
    commands.append("R 1")
    seq, stamps = metadata(2, 96_000_000)
    seq[1], stamps[1] = 1, 109_000_000
    step(113_000_000, seq, stamps)
    seq, stamps = metadata(3, 110_000_000)
    stamps[0, 7] = 99_000_000
    seq[1] = 2
    step(114_000_000, seq, stamps)
    # Exercise the real 20/30/10 ms limits at short and multi-day global times.
    # Integer nanosecond admission avoids subtracting large float64 seconds.
    for now_ns in (100_000_000, 250_000_000_000_000):
        for motor_age_ns, acceleration_age_ns in (
            (20_000_000, 30_000_000),
            (20_000_001, 30_000_000),
            (20_000_000, 30_000_001),
            (5_000_000, 15_000_000),
            (5_000_000, 15_000_001),
        ):
            owner.reset(torch.arange(2, dtype=torch.long))
            commands.extend(("R 0", "R 1"))
            seq, stamps = metadata(1, now_ns - motor_age_ns)
            stamps[:, 7] = now_ns - acceleration_age_ns
            step(now_ns, seq, stamps)
    result = subprocess.run([str(executable)], input="\n".join(commands) + "\n",
                            check=True, capture_output=True, text=True)
    actual = [tuple(float(value) for value in line.split()) for line in result.stdout.splitlines()]
    assert len(actual) == len(expected)
    for reference, received in zip(actual, expected):
        assert reference[:3] == received[:3]
        assert reference[3:] == pytest.approx(received[3:], abs=1e-10)
