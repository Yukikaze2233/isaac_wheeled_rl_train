#!/usr/bin/env python3
"""Bounded localhost UDP over an existing SSH channel; no firewall/system changes.

The remote half runs as native Windows Python so media never traverses WSL's UDP
loopback path. Video is still encoded by Kit and decoded by the official client.
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
import sys
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
                raise ValueError("invalid media tunnel frame")
            incoming.put((peer, read_exact(fd, size)))
    except (EOFError, OSError, ValueError):
        incoming.put(None)


def send_frame(fd, peer, payload):
    data = struct.pack("!II", peer, len(payload)) + payload
    while data:
        count = os.write(fd, data)
        data = data[count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "remote"))
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--control-path")
    parser.add_argument("--host")
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--remote-python", default="/mnt/d/Python/Python3.11.4/python.exe")
    parser.add_argument("--remote-script", default="D:/isaac60-native/tools/media_tunnel.py")
    parser.add_argument("--remote-audit", default="D:/isaac60-native/audits/media-tunnel-remote.json")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 7200:
        parser.error("lifetime must be 1..7200 seconds")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    process = listener = error_log = None
    incoming = queue.Queue()
    sockets, addresses, identifiers = {}, {}, {}
    counters = {"mode": args.mode, "pid": os.getpid(), "to_udp": 0, "from_udp": 0,
                "bytes_to_udp": 0, "bytes_from_udp": 0, "transport": "SSH-framed UDP, localhost endpoints"}
    started = time.monotonic()
    try:
        if args.mode == "local":
            if not args.host or not args.control_path:
                parser.error("local mode requires host and existing ControlMaster")
            command = " ".join(shlex.quote(x) for x in [args.remote_python, "-I", "-u", "-B", args.remote_script,
                "remote", "--seconds", str(args.seconds), "--audit", args.remote_audit])
            error_log = args.audit.with_suffix(".ssh.log").open("wb")
            process = subprocess.Popen(["ssh", "-S", args.control_path, "-o", "BatchMode=yes", "-T",
                                        "-p", str(args.ssh_port), args.host, command],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log)
            input_fd, output_fd = process.stdout.fileno(), process.stdin.fileno()
            listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listener.bind(("127.0.0.1", 47998))
            listener.setblocking(False)
        else:
            input_fd, output_fd = sys.stdin.fileno(), sys.stdout.fileno()
        thread = threading.Thread(target=reader, args=(input_fd, incoming), daemon=True)
        thread.start()
        running = True
        while running and time.monotonic() - started < args.seconds:
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
                    sockets[peer].sendto(payload, ("127.0.0.1", 47998))
                counters["to_udp"] += 1
                counters["bytes_to_udp"] += len(payload)
            readable = [listener] if listener else list(sockets.values())
            if not readable:
                time.sleep(.01)
                continue
            ready, _, _ = select.select(readable, [], [], .01)
            for channel in ready:
                payload, address = channel.recvfrom(65535)
                if args.mode == "local":
                    if address not in identifiers:
                        if len(identifiers) >= 64:
                            continue
                        peer = len(identifiers) + 1
                        identifiers[address] = peer
                        addresses[peer] = address
                    peer = identifiers[address]
                else:
                    peer = next(key for key, value in sockets.items() if value is channel)
                send_frame(output_fd, peer, payload)
                counters["from_udp"] += 1
                counters["bytes_from_udp"] += len(payload)
            if int(time.monotonic() - started) % 5 == 0:
                args.audit.write_text(json.dumps({**counters, "elapsed": time.monotonic() - started}, indent=2))
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
            counters["ssh_exit"] = process.returncode
        if error_log:
            error_log.close()
        args.audit.write_text(json.dumps({**counters, "elapsed": time.monotonic() - started, "closed": True}, indent=2))


if __name__ == "__main__":
    main()
