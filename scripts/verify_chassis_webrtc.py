#!/usr/bin/env python3
"""Verify received video, real state freshness and native camera input over WebRTC."""
import argparse
import asyncio
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--remote-gui-state", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-tools", type=Path, default=ROOT.parent / "isaac_wheeled_rl_train-60/scripts/windows_native")
    args = parser.parse_args()
    sys.path.insert(0, str(args.client_tools))
    from client_cdp import ClientCDP
    from connect_client import STATUS
    receipt = json.loads(args.receipt.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    ssh = ["ssh", "-S", receipt["control_path"], "-o", "BatchMode=yes", "-p", str(receipt["ssh_port"]), receipt["host"]]
    code = "from pathlib import Path;print(Path(" + repr(args.remote_gui_state) + ").read_text())"

    def gui_state():
        return json.loads(subprocess.check_output(ssh + ["python3 -B -c " + shlex.quote(code)], text=True, timeout=15))

    async with ClientCDP() as client:
        video_before = await client.evaluate(STATUS)
        before = gui_state()
        await client.command("Input.dispatchMouseEvent", {"type": "mousePressed", "x": 600, "y": 300,
            "button": "right", "buttons": 2, "clickCount": 1})
        for i in range(1, 11):
            await client.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 600 + i * 15,
                "y": 300 + i * 4, "button": "right", "buttons": 2})
            await asyncio.sleep(.08)
        await client.command("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": 750, "y": 340,
            "button": "right", "buttons": 0, "clickCount": 1})
        await asyncio.sleep(2)
        after = gui_state()
        video_after = await client.evaluate(STATUS)
        await client.screenshot(args.output / "native_gui.png")
        camera_delta = max(abs(x - y) for row_x, row_y in zip(before["camera_matrix"], after["camera_matrix"])
                           for x, y in zip(row_x, row_y))
        frames = video_after["videos"][0]["frames"] - video_before["videos"][0]["frames"]
        result = {"video_before": video_before, "video_after": video_after, "gui_before": before, "gui_after": after,
                  "decoded_frames_increased": frames > 0, "camera_matrix_max_change": camera_delta,
                  "native_input_verified": camera_delta > 1e-5,
                  "actual_source_frames_increased": after["source_frames"] > before["source_frames"],
                  "source_is_live": after["status"] == "LIVE"}
        (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("video_before", "video_after", "gui_before", "gui_after")}, indent=2))
        if not result["decoded_frames_increased"] or not result["native_input_verified"]:
            raise SystemExit("Video or camera input did not pass verification")


if __name__ == "__main__":
    asyncio.run(main())
