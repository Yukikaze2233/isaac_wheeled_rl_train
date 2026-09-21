#!/usr/bin/env python3
"""Reuse the bounded native UDP-over-SSH transport, tolerating transient ICMP.

Derived from this repository's isaac60 Windows-native streaming transport.
Windows can report UDP port-unreachable while Kit opens a new WebRTC session;
that datagram loss must not terminate the entire media relay.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import select
import shlex
import socket
import struct
import subprocess
import threading
import time


def read_exact(fd, size):
    data = bytearray()
    while len(data) < size:
        block = os.read(fd, size - len(data))
        if not block:
            raise EOFError
        data.extend(block)
    return bytes(data)


def reader(fd, incoming):
    try:
        while True:
            peer, size = struct.unpack("!II", read_exact(fd, 8))
            if not 0 < size <= 65507 or not 0 < peer <= 64:
                raise ValueError("Invalid media frame")
            incoming.put((peer, read_exact(fd, size)))
    except (EOFError, OSError, ValueError):
        incoming.put(None)


def send_frame(fd, peer, payload):
    data = struct.pack("!II", peer, len(payload)) + payload
    while data:
        written = os.write(fd, data)
        data = data[written:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("local", "remote"))
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--control-path")
    parser.add_argument("--host")
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--remote-python", default="/mnt/d/Python/Python3.11.4/python.exe")
    parser.add_argument("--remote-script", required=True)
    parser.add_argument("--remote-audit", default="D:/isaac60-native/audits/v5-media.json")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 7200:
        parser.error("Lifetime must be bounded to 1-7200 seconds")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(0, os.O_BINARY)
        msvcrt.setmode(1, os.O_BINARY)
    process = listener = error_log = None
    incoming = queue.Queue()
    sockets, addresses, identifiers = {}, {}, {}
    stats = {"pid": os.getpid(), "mode": args.mode, "to_udp": 0, "from_udp": 0,
             "bytes_to_udp": 0, "bytes_from_udp": 0, "transient_udp_resets": 0}
    start, last_write = time.monotonic(), 0.
    try:
        if args.mode == "local":
            if not args.host or not args.control_path:
                parser.error("Local relay requires host and ControlMaster")
            command = shlex.join([args.remote_python, "-I", "-u", "-B", args.remote_script, "remote",
                "--seconds", str(args.seconds), "--audit", args.remote_audit, "--remote-script", args.remote_script])
            error_log = args.audit.with_suffix(".ssh.log").open("wb")
            process = subprocess.Popen(["ssh", "-S", args.control_path, "-o", "BatchMode=yes", "-T",
                "-p", str(args.ssh_port), args.host, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log)
            input_fd, output_fd = process.stdout.fileno(), process.stdin.fileno()
            listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listener.bind(("127.0.0.1", 47998))
            listener.setblocking(False)
        else:
            input_fd, output_fd = 0, 1
        threading.Thread(target=reader, args=(input_fd, incoming), daemon=True).start()
        running = True
        while running and time.monotonic() - start < args.seconds:
            while True:
                try:
                    item = incoming.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    running = False
                    break
                peer, payload = item
                if args.mode == "local":
                    if peer not in addresses:
                        continue
                    listener.sendto(payload, addresses[peer])
                else:
                    if peer not in sockets:
                        channel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        channel.bind(("127.0.0.1", 0))
                        channel.setblocking(False)
                        channel.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
                        sockets[peer] = channel
                    try:
                        sockets[peer].sendto(payload, ("127.0.0.1", 47998))
                    except ConnectionResetError:
                        stats["transient_udp_resets"] += 1
                        continue
                stats["to_udp"] += 1
                stats["bytes_to_udp"] += len(payload)
            readable = [listener] if listener else list(sockets.values())
            ready = select.select(readable, [], [], .01)[0] if readable else []
            for channel in ready:
                try:
                    payload, address = channel.recvfrom(65535)
                except ConnectionResetError:
                    stats["transient_udp_resets"] += 1
                    continue
                if args.mode == "local":
                    if address not in identifiers:
                        if len(identifiers) >= 64:
                            continue
                        peer = len(identifiers) + 1
                        identifiers[address], addresses[peer] = peer, address
                    peer = identifiers[address]
                else:
                    peer = next(k for k, value in sockets.items() if value is channel)
                send_frame(output_fd, peer, payload)
                stats["from_udp"] += 1
                stats["bytes_from_udp"] += len(payload)
            now = time.monotonic()
            if now - last_write > 1.:
                args.audit.write_text(json.dumps({**stats, "elapsed": now - start}, indent=2))
                last_write = now
            if not readable:
                time.sleep(.01)
    finally:
        for channel in sockets.values():
            channel.close()
        if listener:
            listener.close()
        if process:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            stats["ssh_exit"] = process.returncode
        if error_log:
            error_log.close()
        args.audit.write_text(json.dumps({**stats, "closed": True, "elapsed": time.monotonic() - start}, indent=2))


if __name__ == "__main__":
    main()
