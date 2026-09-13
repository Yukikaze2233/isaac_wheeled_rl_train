#!/usr/bin/env python3
"""Verify Linux mouse events alter the real Windows Kit selection and camera."""
import argparse
import asyncio
import json
from pathlib import Path
import shlex
import subprocess

from client_cdp import ClientCDP


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--state", required=True, help="WSL-readable Windows gui-state.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def state():
        return json.loads(subprocess.check_output([
            "ssh", "-S", args.control_path, "-o", "BatchMode=yes", "-p", str(args.ssh_port),
            args.host, "cat " + shlex.quote(args.state)], timeout=5))

    before = state()
    async with ClientCDP() as client:
        await client.command("Page.bringToFront")
        geometry = await client.evaluate("(()=>{const v=document.querySelector('video'),r=v.getBoundingClientRect();return {width:v.videoWidth,height:v.videoHeight,x:r.x,y:r.y,w:r.width,h:r.height}})()")
        scale = min(geometry["w"] / geometry["width"], geometry["h"] / geometry["height"])
        left = geometry["x"] + (geometry["w"] - geometry["width"] * scale) / 2
        top = geometry["y"] + (geometry["h"] - geometry["height"] * scale) / 2

        async def mouse(kind, x, y, button="left", buttons=0):
            await client.command("Input.dispatchMouseEvent", {
                "type": kind, "x": left + x * scale, "y": top + y * scale,
                "button": button, "buttons": buttons, "clickCount": 1})

        async def click(x, y):
            await mouse("mouseMoved", x, y)
            await mouse("mousePressed", x, y, buttons=1)
            await mouse("mouseReleased", x, y)
            await asyncio.sleep(.5)

        await click(18, 12)
        await client.screenshot(args.output / "native-file-menu.png")
        await click(18, 12)
        await click(1010, 134)
        selected = state()
        await client.screenshot(args.output / "native-selection.png")
        assert "/World/NativeGuiProbe" in selected["selected_paths"], selected
        await mouse("mousePressed", 480, 280, button="right", buttons=2)
        for step in range(1, 11):
            await mouse("mouseMoved", 480 + step * 8, 280 + step * 3, button="right", buttons=2)
            await asyncio.sleep(.03)
        await mouse("mouseReleased", 560, 310, button="right")
        await asyncio.sleep(.7)
        after = state()
        await client.screenshot(args.output / "native-camera-after.png")
        delta = max(abs(a - b) for row_a, row_b in zip(selected["camera_matrix"], after["camera_matrix"])
                    for a, b in zip(row_a, row_b))
        record = {"before": before, "selected": selected, "after": after, "camera_max_delta": delta,
                  "source": "CDP mouse on official Linux client's decoded video; state queried from native Windows Kit"}
        (args.output / "input-proof.json").write_text(json.dumps(record, indent=2))
        assert delta > 1e-4, "camera did not react to Linux right-drag"
        print(json.dumps({"selection": selected["selected_paths"], "camera_max_delta": delta,
                          "native_windows_pid": after["pid"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
