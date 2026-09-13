#!/usr/bin/env python3
"""Connect the official client and prove actual decoded video rather than TCP readiness."""
import argparse
import asyncio
import json
from pathlib import Path
import time
from urllib.parse import urlencode

from client_cdp import ClientCDP

HOOK = """
window.__nativePeers = [];
const NativePC = window.RTCPeerConnection;
window.RTCPeerConnection = class extends NativePC {
  constructor(...args) { super(...args); window.__nativePeers.push(this); }
};
"""
STATUS = """(async () => ({
  time: performance.now(),
  videos: [...document.querySelectorAll('video')].map(v => {
    const q = v.getVideoPlaybackQuality(), r = v.getBoundingClientRect();
    return {width:v.videoWidth,height:v.videoHeight,time:v.currentTime,frames:q.totalVideoFrames,
      dropped:q.droppedVideoFrames,rect:{x:r.x,y:r.y,width:r.width,height:r.height}};
  }),
  peers: await Promise.all((window.__nativePeers || []).map(async p => ({
    state:p.connectionState,ice:p.iceConnectionState,
    inbound:[...(await p.getStats()).values()].filter(s => s.type==='inbound-rtp').map(s => ({
      kind:s.kind,bytes:s.bytesReceived,frames:s.framesDecoded,fps:s.framesPerSecond,
      width:s.frameWidth,height:s.frameHeight,decoder:s.decoderImplementation
    }))
  })))
}))()"""


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", type=int, default=9229)
    parser.add_argument("--resolution", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=90)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    async with ClientCDP(args.port) as client:
        await client.command("Page.enable")
        await client.command("Page.addScriptToEvaluateOnNewDocument", {"source": HOOK})
        url = client.page_url.split("?", 1)[0] + "?" + urlencode({
            "server": args.server, "signalPort": 49100, "streamPort": 47998,
            "resolution": args.resolution, "autoConnect": 1})
        await client.command("Page.navigate", {"url": url})
        deadline = time.monotonic() + args.seconds
        first = None
        while time.monotonic() < deadline:
            record = await client.evaluate(STATUS)
            records.append(record)
            videos = record["videos"]
            if videos and videos[0]["frames"] > 0:
                if first is None:
                    first = videos[0]["frames"]
                    await client.screenshot(args.output / "first.png")
                if videos[0]["frames"] >= first + 120:
                    await client.screenshot(args.output / "continuous.png")
                    break
            await asyncio.sleep(0.5)
        (args.output / "video.json").write_text(json.dumps({"server": args.server, "samples": records}, indent=2))
        assert first is not None and records[-1]["videos"][0]["frames"] >= first + 120, "No verified continuous decoded video"
        print(json.dumps(records[-1], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
