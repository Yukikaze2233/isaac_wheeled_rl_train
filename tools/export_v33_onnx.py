"""Export the V3.3 MuJoCo-trained policy to ONNX (35D->6D contract).

Wraps obs-normalization + deterministic mean into one graph:
    obs[1,35] -> actions[1,6]
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mujoco_v33 import ActorCritic  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs_v33")


class DeterministicPolicy(nn.Module):
    def __init__(self, ac: ActorCritic, obs_mean, obs_var):
        super().__init__()
        self.ac = ac
        self.register_buffer("obs_mean", torch.from_numpy(obs_mean))
        self.register_buffer("obs_var", torch.from_numpy(obs_var))

    def forward(self, obs):
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        mu, _, _ = self.ac(x)
        return mu


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LOG_DIR, "v33_policy.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(LOG_DIR, "v33_policy.onnx")
    ck = torch.load(ckpt, weights_only=False)
    ac = ActorCritic()
    ac.load_state_dict(ck["ac"])
    ac.eval()
    policy = DeterministicPolicy(ac, ck["obs_mean"], ck["obs_var"])
    policy.eval()
    dummy = torch.zeros(1, 35)
    torch.onnx.export(
        policy, dummy, out,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes=None, opset_version=17,
        use_external_data_format=False,  # single-file ONNX for easy copy/deploy
    )
    print(f"exported {out} (obs[1,35] -> actions[1,6])")


if __name__ == "__main__":
    main()
