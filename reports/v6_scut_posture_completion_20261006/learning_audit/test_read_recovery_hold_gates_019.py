"""Native TFRecord framing and coverage/denominator checks for gate wrapper."""
import importlib.util
from pathlib import Path
import struct

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hold_gate_wrapper", HERE / "read_recovery_hold_gates_019.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
reader = wrapper.load_reader()


def record(step, values):
    event = reader.Event(step=step, wall_time=1000. + step)
    for key, value in values.items():
        event.summary.value.add(tag=key, simple_value=value)
    payload = event.SerializeToString()
    size = struct.pack("<Q", len(payload))
    return size + struct.pack("<I", reader.masked_crc32c(size)) + payload + struct.pack("<I", reader.masked_crc32c(payload))


def fixture(tmp_path, *, empty=False, missing=False, wrong_samples=False):
    authority = dict(groups=["forward"], population={"forward": 2})
    window = reader.window_spec(50, 60)
    tags = wrapper.wanted_tags(authority)
    data = bytearray()
    for step in window["dense_steps"]:
        values = dict.fromkeys(tags, 0.)
        values["/recovery/hold_population"] = 0. if empty else 10.
        for name in wrapper.GATES:
            values["/recovery/hold_gate/" + name + "_pass_fraction"] = 0. if empty else .5
        values["LearningAudit/forward/rollout_samples"] = 47. if wrong_samples else 48.
        values["LearningAudit/forward/command_vx_m_s"] = .3
        values["LearningAudit/forward/command_yaw_rad_s"] = .2
        values["LearningAudit/forward/held_low_vx_fraction"] = .25
        if missing and step == window["dense_steps"][-1]:
            del values["/recovery/hold_gate/height_pass_fraction"]
        data.extend(record(step, values))
    path = tmp_path / "events.out.tfevents.synthetic"
    path.write_bytes(data)
    return authority, window, path


def parse(authority, window, path):
    return reader.parse_events(path, wrapper.wanted_tags(authority), window["end_step"], min_step=window["start_step"])


def test_complete_crc_window_and_physical_command_denominator(tmp_path):
    authority, window, path = fixture(tmp_path)
    series, audit = parse(authority, window, path)
    result = wrapper.summarize(reader, series, authority, 50, 60)
    assert result["complete"] and audit["crc_valid_records"] == 50
    assert result["hold_gates"]["height"]["mean"] == .5
    assert result["commands"]["forward"]["physical_observation_samples"] == 2400
    assert result["commands"]["forward"]["command_vx_m_s"] == pytest.approx(.3)
    assert result["commands"]["forward"]["command_yaw_rad_s"] == pytest.approx(.2)
    assert any("Multiplying their averages" in note for note in result["limitations"])


def test_zero_holding_population_is_not_all_gate_failure_evidence(tmp_path):
    authority, window, path = fixture(tmp_path, empty=True)
    series, _ = parse(authority, window, path)
    result = wrapper.summarize(reader, series, authority, 50, 60)
    assert result["complete"] and result["rollouts_with_positive_mean_hold_population"] == 0
    assert result["hold_gates"]["valid"]["mean"] == 0.
    assert any("not evidence that every lane failed" in note for note in result["limitations"])


def test_missing_end_gate_keeps_window_incomplete(tmp_path):
    authority, window, path = fixture(tmp_path, missing=True)
    series, _ = parse(authority, window, path)
    result = wrapper.summarize(reader, series, authority, 50, 60)
    assert not result["complete"]
    assert result["missing"] == {"/recovery/hold_gate/height_pass_fraction": [window["end_step"]]}
    assert "commands" not in result


@pytest.mark.parametrize("byte", [8, -1])
def test_corrupt_header_or_payload_crc_rejected(tmp_path, byte):
    authority, window, path = fixture(tmp_path)
    damaged = bytearray(path.read_bytes())
    damaged[byte] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="CRC mismatch"):
        parse(authority, window, path)


def test_wrong_command_sample_denominator_is_rejected(tmp_path):
    authority, window, path = fixture(tmp_path, wrong_samples=True)
    series, _ = parse(authority, window, path)
    with pytest.raises(ValueError, match="sample denominator differs"):
        wrapper.summarize(reader, series, authority, 50, 60)
