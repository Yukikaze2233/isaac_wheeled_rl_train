"""Exact tensor and RNG restoration shared by command state owners."""
import torch

def _restart_tensor(value, shape, dtype, name, *, device="cpu"):
    """Strict checkpoint transport; conversion happens only after validation."""
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device != torch.device(device) or value.dtype != dtype
            or tuple(value.shape) != tuple(shape) or value.requires_grad):
        raise ValueError(f"Restart {name} requires {dtype} {tuple(shape)} on {device}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"Restart {name} must be finite")
    return value.detach().clone()


def _restart_rng(generator, value):
    expected = generator.get_state()
    state = _restart_tensor(value, expected.shape, torch.uint8, "generator state")
    # Validate the actual Torch engine format without touching caller/global RNG.
    probe = torch.Generator(device=generator.device)
    try:
        probe.set_state(state)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Invalid Torch restart generator state") from exc
    return state
