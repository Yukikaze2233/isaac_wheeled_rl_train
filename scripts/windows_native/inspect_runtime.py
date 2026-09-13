"""Record only package/runtime identity; optional CUDA/NMS verification, never launch Kit."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()
names = ("isaacsim", "isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "torch", "torchvision",
         "warp-lang", "newton", "numpy", "Pillow", "onnx", "onnxruntime", "packaging", "rsl-rl-lib",
         "gymnasium", "hydra-core", "h5py", "tensorboard", "usd-core")
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
result = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
          "versions": versions, "sys_path": sys.path}
if args.cuda:
    import torch
    import torchvision
    assert torch.cuda.is_available()
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device="cuda:0")
    scores = torch.tensor([.9, .5], device="cuda:0")
    kept = torchvision.ops.nms(boxes, scores, .5)
    torch.cuda.synchronize()
    assert kept.tolist() == [0]
    result["cuda"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                      "runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "nms": True}
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
