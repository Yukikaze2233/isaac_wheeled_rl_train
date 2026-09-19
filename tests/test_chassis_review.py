"""A live TensorBoard tail must not invalidate or invent complete scalar records."""
import importlib.util
from pathlib import Path
import struct

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c


def test_crc_checked_review_discards_only_incomplete_tail(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("chassis_review", root / "tools/review_chassis_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    event = Event(step=7, wall_time=123.)
    event.summary.value.add(tag="/task/height_error_m", simple_value=.003)
    payload = event.SerializeToString()
    header = struct.pack("<Q", len(payload))
    record = header + struct.pack("<I", masked_crc32c(header)) + payload + struct.pack("<I", masked_crc32c(payload))
    path = tmp_path / "events.test"
    path.write_bytes(record + record[:14])
    scalars, audit = module.events(path)
    assert scalars["/task/height_error_m"][0][0] == 7
    assert scalars["/task/height_error_m"][0][1] == pytest.approx(.003)
    assert audit == {"complete_record_bytes": len(record), "incomplete_live_tail_bytes": 14}
    corrupt = bytearray(record)
    corrupt[13] ^= 1
    path.write_bytes(corrupt)
    with pytest.raises(ValueError, match="data CRC"):
        module.events(path)
