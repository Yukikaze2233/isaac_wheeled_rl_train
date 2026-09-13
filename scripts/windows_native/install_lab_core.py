"""Use the official core installer, retaining the explicitly audited Sim-6 CUDA build.

The pinned Lab CLI still hardcodes Torch 2.10. This deployment requires the
Sim-6-documented 2.11/cu128 stack. No official source file is modified.
"""
import importlib.metadata as metadata
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "source/IsaacLab/source/isaaclab"))
from isaaclab.cli.commands import install


def require_native_cuda_build():
    expected = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"native CUDA build changed: {actual}")
    print("Retaining audited native CUDA build: " + str(actual), flush=True)


install._ensure_cuda_torch = require_native_cuda_build
install.command_install("core")
