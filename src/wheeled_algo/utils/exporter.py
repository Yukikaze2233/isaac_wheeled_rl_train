"""Policy export: rsl_rl/algo_ext checkpoints -> ONNX under the deploy contract."""
import torch
import torch.nn as nn


def actor_dims_from_ckpt(state_dict: dict) -> tuple[int, int, list[int]]:
    """Recover (num_obs, num_actions, hidden_dims) from an ActorCritic state.

    rsl_rl names actor MLP layers actor.<i>.weight; the action dim comes from
    the `std` parameter (state-independent log-std buffer).
    """
    actor_layers = sorted(
        (int(k.split(".")[1]), k) for k in state_dict if k.startswith("actor.") and k.endswith(".weight")
    )
    num_obs = state_dict[actor_layers[0][1]].shape[1]
    hidden = [state_dict[k].shape[0] for _, k in actor_layers[:-1]]
    num_actions = state_dict["std"].shape[0]
    return num_obs, num_actions, hidden


class _PolicyMean(nn.Module):
    def __init__(self, forward_fn):
        super().__init__()
        self._forward = forward_fn

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._forward(obs)


def export_actor_onnx(policy: nn.Module, num_obs: int, output_path: str) -> None:
    """Export a policy exposing act_inference(obs) as static [1,num_obs]->[1,6] ONNX."""
    dummy = torch.zeros(1, num_obs)
    torch.onnx.export(
        _PolicyMean(policy.act_inference), dummy, output_path,
        input_names=["obs"], output_names=["actions"],
        opset_version=13, dynamic_axes=None,
    )
