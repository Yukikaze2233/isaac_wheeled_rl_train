"""Audit batching must retain native TensorBoard values, tags and clocks."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from torch.utils.tensorboard import SummaryWriter


SPEC = spec_from_file_location("train_chassis_scalar_batch",
    Path(__file__).resolve().parents[1] / "scripts/train_chassis.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_summaries(directory):
    paths = list(directory.glob("events.out.tfevents.*"))
    assert len(paths) == 1
    return [event for raw in RawEventFileLoader(str(paths[0])).Load()
            if (event := Event.FromString(raw)).HasField("summary")]


@pytest.mark.parametrize("count", [0, 1, 1000])
def test_native_batch_preserves_order_bits_and_repeated_steps(tmp_path, count):
    values = {f"LearningAudit/group_{i % 7}/metric_{i}": (i - 500) / 7.
              for i in range(count)}
    reference, candidate = tmp_path / "scalar", tmp_path / "batch"
    with SummaryWriter(reference) as writer:
        for step in (45566, 45567):
            for tag, value in values.items():
                writer.add_scalar(tag, value, step)
    with SummaryWriter(candidate) as writer:
        for step in (45566, 45567):
            MODULE.log_scalar_batch(writer, values, step)
    old, new = read_summaries(reference), read_summaries(candidate)

    def records(events):
        return [(event.step, value.tag, struct.pack("<f", value.simple_value),
                 value.metadata.SerializeToString())
                for event in events for value in event.summary.value]

    assert records(new) == records(old)
    assert len(old) == 2 * count
    assert len(new) == 2 * ((count + 511) // 512)
    if count:
        old_acc = EventAccumulator(str(reference), size_guidance={"scalars": 0}).Reload()
        new_acc = EventAccumulator(str(candidate), size_guidance={"scalars": 0}).Reload()
        assert new_acc.Tags() == old_acc.Tags()
        for tag in values:
            assert [(x.step, x.value) for x in new_acc.Scalars(tag)] == [
                (x.step, x.value) for x in old_acc.Scalars(tag)]


def test_native_batch_retains_signed_zero_nonfinite_and_large_values(tmp_path):
    values = dict(zip(("negative_zero", "inf", "minus_inf", "nan", "large"),
                      (-0., float("inf"), -float("inf"), float("nan"), 1e30)))
    with SummaryWriter(tmp_path / "scalar") as writer:
        for tag, value in values.items():
            writer.add_scalar(tag, value, 99)
    with SummaryWriter(tmp_path / "batch") as writer:
        MODULE.log_scalar_batch(writer, values, 99)
    old = b"".join(value.SerializeToString()
        for event in read_summaries(tmp_path / "scalar") for value in event.summary.value)
    new = b"".join(value.SerializeToString()
        for event in read_summaries(tmp_path / "batch") for value in event.summary.value)
    assert new == old


def test_custom_writer_and_disabled_logging_keep_existing_semantics():
    values = {"a": 1., "b": 2.}
    written = []
    MODULE.log_scalar_batch(SimpleNamespace(add_scalar=lambda *a: written.append(a)), values, 7)
    assert written == [("a", 1., 7), ("b", 2., 7)]
    MODULE.log_scalar_batch(None, values, 7)

    def reject(*args):
        raise OSError("writer failure")

    with pytest.raises(OSError, match="writer failure"):
        MODULE.log_scalar_batch(SimpleNamespace(add_scalar=reject), values, 7)
