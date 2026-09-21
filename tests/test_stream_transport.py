"""UDP datagrams retain exact bytes across fragmented SSH pipe transport."""
from pathlib import Path
import os
import queue
import struct
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from windows_native.media_tunnel import reader, send_frame


def test_binary_udp_round_trip_including_large_datagrams():
    source, target = os.pipe()
    incoming = queue.Queue()
    thread = threading.Thread(target=reader, args=(source, incoming))
    thread.start()
    try:
        payload = (b"\x00\xff\r\n\x80" * 12000)
        send_frame(target, 1, payload)
        assert incoming.get(timeout=3) == (1, payload)
        os.close(target)
        target = None
        assert incoming.get(timeout=3) is None
    finally:
        if target is not None:
            os.close(target)
        thread.join(timeout=3)
        os.close(source)


def test_invalid_peer_is_rejected_before_payload_read():
    source, target = os.pipe()
    incoming = queue.Queue()
    os.write(target, struct.pack("!II", 0, 96))
    os.close(target)
    reader(source, incoming)
    os.close(source)
    assert incoming.get_nowait() is None
