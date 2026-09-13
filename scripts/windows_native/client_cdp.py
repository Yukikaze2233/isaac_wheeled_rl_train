#!/usr/bin/env python3
"""Inspect/control only the dedicated official Linux streaming client via local CDP."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import urllib.request

import websockets


class ClientCDP:
    def __init__(self, port=9229):
        self.port, self.serial = port, 0
        self.events = []

    async def __aenter__(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=5) as response:
            target = next(item for item in json.load(response)
                          if item["type"] == "page" and item["title"] == "Isaac Sim WebRTC Streaming Client")
        self.page_url = target["url"]
        self.socket = await websockets.connect(target["webSocketDebuggerUrl"], max_size=32_000_000)
        return self

    async def __aexit__(self, *_):
        await self.socket.close()

    async def command(self, method, params=None):
        self.serial += 1
        await self.socket.send(json.dumps({"id": self.serial, "method": method, "params": params or {}}))
        while True:
            reply = json.loads(await asyncio.wait_for(self.socket.recv(), timeout=20))
            if reply.get("id") == self.serial:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply.get("result", {})
            self.events.append(reply)

    async def evaluate(self, expression):
        reply = await self.command("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                          "awaitPromise": True})
        if "exceptionDetails" in reply:
            raise RuntimeError(reply["exceptionDetails"])
        return reply["result"].get("value")

    async def screenshot(self, path):
        reply = await self.command("Page.captureScreenshot", {"format": "png"})
        Path(path).write_bytes(base64.b64decode(reply["data"]))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9229)
    parser.add_argument("--expression", default="({text:document.body.innerText,inputs:[...document.querySelectorAll('input')].map(x=>({type:x.type,value:x.value,placeholder:x.placeholder})),buttons:[...document.querySelectorAll('button')].map(x=>x.innerText),videos:[...document.querySelectorAll('video')].map(x=>({id:x.id,width:x.videoWidth,height:x.videoHeight,time:x.currentTime,quality:x.getVideoPlaybackQuality()}))})")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    async with ClientCDP(args.port) as client:
        print(json.dumps(await client.evaluate(args.expression), indent=2))
        if args.screenshot:
            await client.screenshot(args.screenshot)


if __name__ == "__main__":
    asyncio.run(main())
