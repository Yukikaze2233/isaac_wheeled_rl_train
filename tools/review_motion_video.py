#!/usr/bin/env python3
"""Sample timestamped video frames for qualitative motion review, not metric pose estimation."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3.)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--crop", help="Optional ffmpeg crop w:h:x:y before scaling")
    args = parser.parse_args()
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(args.video)]))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(stream["avg_frame_rate"]))
    stride = max(1, round(args.interval * fps))
    args.output.mkdir(parents=True, exist_ok=False)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-ss", str(args.start), "-i", str(args.video)]
    if args.duration is not None:
        command += ["-t", str(args.duration)]
    filters = f"select=not(mod(n\\,{stride})),"
    if args.crop:
        filters += f"crop={args.crop},"
    command += ["-vf", filters + f"scale={args.width}:-1", "-fps_mode", "vfr", str(args.output / "frame_%04d.png")]
    subprocess.run(command, check=True)
    frames = sorted(args.output.glob("frame_*.png"))
    font = ImageFont.load_default(size=18)
    timestamps = []
    for start in range(0, len(frames), 18):
        sample = Image.open(frames[start]).convert("RGB")
        w, h = sample.size
        sheet = Image.new("RGB", (w * 3, (h + 28) * 6), "#121920")
        draw = ImageDraw.Draw(sheet)
        for idx, path in enumerate(frames[start:start + 18]):
            x, y = (idx % 3) * w, (idx // 3) * (h + 28)
            sheet.paste(Image.open(path).convert("RGB"), (x, y + 28))
            seconds = args.start + (start + idx) * stride / fps
            label = f"{int(seconds)//60:02d}:{seconds % 60:05.2f}  frame {start+idx}"
            draw.text((x + 8, y + 4), label, fill="white", font=font)
            timestamps.append({"file": path.name, "media_seconds": seconds})
        sheet.save(args.output / f"sheet_{start//18:02d}.jpg", quality=93)
    report = {"video": str(args.video), "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
              "width": stream["width"], "height": stream["height"], "fps": fps,
              "duration_s": float(probe["format"]["duration"]), "timestamps": timestamps, "crop": args.crop,
              "scope": "qualitative edited-video review; timestamps are not calibrated real-world motion measurements"}
    (args.output / "frames.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "timestamps"}, indent=2))


if __name__ == "__main__":
    main()
