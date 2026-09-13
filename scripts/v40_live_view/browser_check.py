#!/usr/bin/env python3
"""Bounded CDP evidence for an isolated browser, including Viser worker traffic.

Run from a verifier environment with websockets and Pillow. The CDP endpoint must
belong to the dedicated diagnostic browser: this tool closes it on completion.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit

import websockets


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdp-port", type=int, default=19228)
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--probe-url", default="http://127.0.0.1:18088")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 100
    while True:
        try:
            with urllib.request.urlopen(args.probe_url, timeout=1) as response:
                assert response.status == 200
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("viewer HTTP readiness")
            await asyncio.sleep(.1)
    with urllib.request.urlopen(f"http://127.0.0.1:{args.cdp_port}/json/list", timeout=5) as response:
        target = next(item for item in json.load(response) if item["type"] == "page" and item["url"] == "about:blank")
    target_url = urlsplit(target["webSocketDebuggerUrl"])
    ws_url = urlunsplit(target_url._replace(netloc=f"127.0.0.1:{args.cdp_port}"))
    events, timeline, screenshots = [], [], []
    serial, auxiliary = 0, 100000
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        async def event(message):
            nonlocal auxiliary
            events.append(message)
            if message.get("method") == "Target.attachedToTarget":
                for method in ("Network.enable", "Runtime.runIfWaitingForDebugger"):
                    auxiliary += 1
                    await ws.send(json.dumps({"id": auxiliary, "sessionId": message["params"]["sessionId"],
                                              "method": method, "params": {}}))

        async def command(method, params=None):
            nonlocal serial
            serial += 1
            await ws.send(json.dumps({"id": serial, "method": method, "params": params or {}}))
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                if message.get("id") == serial:
                    if "error" in message:
                        raise RuntimeError(message["error"])
                    return message.get("result", {})
                await event(message)

        await command("Page.enable")
        await command("Runtime.enable")
        await command("Network.enable")
        await command("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True})
        await command("Emulation.setDeviceMetricsOverride", {"width": 1400, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
        await command("Page.navigate", {"url": args.url})
        deadline = time.monotonic() + 100
        last_shot = 0.0
        while time.monotonic() < deadline:
            dom = await command("Runtime.evaluate", {"expression": "document.body.innerText", "returnByValue": True})
            text = dom["result"].get("value", "")
            timeline.append({"time_ns": time.time_ns(), "text": text})
            if ("Canonical V40" in text and time.monotonic() - last_shot > .3
                    and (len(screenshots) < 2 or "TRAINING · LIVE" in text or "ENDED" in text)):
                shot = await command("Page.captureScreenshot", {"format": "png"})
                filename = f"frame-{len(screenshots):03d}.png"
                (args.output / filename).write_bytes(base64.b64decode(shot["data"]))
                screenshots.append({"file": filename, "text": text})
                last_shot = time.monotonic()
            if "ENDED" in text:
                break
            # Pump workers between DOM polls; only the verifier sleeps.
            until = time.monotonic() + .1
            while time.monotonic() < until:
                try:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=until - time.monotonic()))
                except asyncio.TimeoutError:
                    break
                await event(message)
        frames = [e["params"]["response"] for e in events if e.get("method") == "Network.webSocketFrameReceived"]
        errors = [e for e in events if e.get("method") == "Runtime.exceptionThrown"]
        record = {"url": args.url, "timeline": timeline, "screenshots": screenshots,
                  "websocket_frames": len(frames), "runtime_errors": errors,
                  "training_live_observed": any("TRAINING · LIVE" in item["text"] for item in timeline),
                  "ended_observed": any("ENDED" in item["text"] for item in timeline)}
        (args.output / "browser.json").write_text(json.dumps(record, indent=2))
        (args.output / "websocket-frames.json").write_text(json.dumps(frames))
        await command("Browser.close")
        assert not errors and len(screenshots) >= 2 and record["training_live_observed"] and record["ended_observed"], record
        print(json.dumps({key: value for key, value in record.items() if key not in ("timeline", "screenshots")}))


if __name__ == "__main__":
    asyncio.run(main())
