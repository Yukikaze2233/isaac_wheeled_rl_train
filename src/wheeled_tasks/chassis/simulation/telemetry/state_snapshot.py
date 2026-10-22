"""Read selected live physics for reporting, without modifying training state."""
from typing import NamedTuple

import torch


class SelectedPhysics(NamedTuple):
    poses: list
    origins: list
    commands: list
    episode_steps: list


def select_group_environments(scene_groups, device):
    """Cache the original first-occurrence order once for an immutable scene."""
    selected = tuple(scene_groups.index(name) for name in dict.fromkeys(scene_groups))
    return selected, torch.tensor(selected, dtype=torch.long, device=device)


def read_selected_physics(indices, *, poses, origins, commands, episode_steps):
    """Transfer each field in its own dtype; episode counters remain integers.

    All inputs belong to the same completed policy step. This helper neither
    advances physics nor writes files, consumes RNG, or changes any input.
    """
    return SelectedPhysics(*(value.detach().index_select(0, indices).cpu().tolist()
        for value in (poses, origins, commands, episode_steps)))
