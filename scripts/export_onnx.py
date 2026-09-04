#!/usr/bin/env python3
"""Export the rsl_rl actor's mean policy to ONNX under the frozen deploy contract.

    /path/to/isaaclab.sh -p scripts/export_onnx.py --checkpoint model_final.pt \
        --output wheeled_biped_flat.onnx

Output: input `obs` float32[1,35] -> output `actions` float32[1,6]
(mean action, no sampling noise — the deploy side must not re-add noise).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402


class PolicyMean(torch.nn.Module):
    def __init__(self, agent: ActorCritic):
        super().__init__()
        self.agent = agent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.agent.act_inference(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_obs, num_actions, hidden = actor_dims_from_ckpt(ckpt["model_state_dict"])
    agent = ActorCritic(num_obs, num_obs, num_actions, hidden, hidden, "elu", 1.0)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    dummy = torch.zeros(1, num_obs)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        PolicyMean(agent), dummy, args.output,
        input_names=["obs"], output_names=["actions"],
        opset_version=13, dynamic_axes=None,  # static [1,35] -> [1,6] per contract
    )
    print(f"exported {args.output} (obs {dummy.shape} -> actions [{num_actions}])")

    # self-verify with onnxruntime when available
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        assert i.shape == [1, num_obs] and o.shape == [1, num_actions], \
            f"contract violation: {i.shape} -> {o.shape}"
        out = sess.run(None, {i.name: np.zeros((1, num_obs), np.float32)})[0]
        print(f"ort check OK: {i.name}{i.shape} -> {o.name}{o.shape}, zero-obs action={out[0]}")
    except ImportError:
        print("onnxruntime not available here; run deploy tools/check_onnx_contract.py later")


if __name__ == "__main__":
    main()
