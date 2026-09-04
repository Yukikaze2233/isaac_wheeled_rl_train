"""Unit tests for state machines and curriculums (pure torch, no Isaac Sim)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importlib.util
import pathlib as _pathlib

import torch

from wheeled_tasks.manager.mdp.curriculums import RewardWeightProgression, Stage

# state_machines is pure torch but sits inside the Isaac Lab-facing env package
# (whose __init__ imports isaaclab); load it by path so the unit test runs anywhere.
_sm_spec = importlib.util.spec_from_file_location(
    "state_machines",
    _pathlib.Path(__file__).resolve().parents[1] / "src/wheeled_tasks/direct/wheeled_biped/state_machines/__init__.py")
_sm = importlib.util.module_from_spec(_sm_spec)
_sm_spec.loader.exec_module(_sm)
AirborneStateMachine = _sm.AirborneStateMachine


def test_airborne_state_machine():
    sm = AirborneStateMachine(num_envs=4, device="cpu",
                              cfg={"body_height_threshold": 0.30, "target_height": 0.32})
    base_h = torch.tensor([0.22, 0.35, 0.35, 0.22])
    vel_z = torch.tensor([0.0, 0.5, -0.5, 0.0])
    contact = torch.tensor([1.0, 0.0, 0.0, 1.0])  # wheels loaded / unloaded
    sm.update(base_h, vel_z, contact > 0.5)
    # env0 on ground; env1/2 above threshold with wheels free -> airborne;
    # env3 high but wheels loaded -> still grounded
    assert not sm.in_air[0] and not sm.in_air[3]
    assert sm.in_air[1] and sm.in_air[2]
    flags = sm.mode_flags()
    assert flags[1, 4] == 1.0 and flags[0, 4] == 0.0      # jump flag only in air
    assert torch.all(flags[:, 0] == 1.0)                   # normal bit always set
    bias = sm.height_bias()
    assert bias[1] > 0.0 and bias[0] == 0.0                # height bias only in air
    # landing trajectory timer advances in flight and saturates
    for _ in range(20):
        sm.update(base_h, vel_z, contact > 0.5)
    assert torch.isclose(sm.state_time[1], torch.tensor(0.42), atol=1e-5)  # 21 updates x 0.02 s
    print("airborne FSM OK")


def test_reward_weight_progression():
    stages = [
        Stage(reward_weights={"track_height_exp": 1.0}, threshold=0.4, min_episodes=10),
        Stage(reward_weights={"track_height_exp": 0.5, "track_height_exp_tight": 1.0},
              threshold=0.4, min_episodes=10),
    ]
    prog = RewardWeightProgression(stages, window_size=4)
    # below threshold: stays in stage 0
    for _ in range(10):
        prog.track(0.2, episodes_finished=5)
    _, eff = prog.step()
    assert eff["stage_idx"] == 0 and eff["reward_weights"]["track_height_exp"] == 1.0
    # beat threshold with enough episodes: advances to stage 1
    for _ in range(10):
        prog.track(0.5, episodes_finished=5)
    _, eff = prog.step()
    assert eff["stage_idx"] == 1 and eff["reward_weights"].get("track_height_exp_tight") == 1.0
    # last stage keeps its weights (restore-defaults only fires on re-trigger)
    _, eff = prog.step()
    assert eff["stage_idx"] == 1
    print("curriculum OK")


def test_terrain_flag_mapping():
    from wheeled_tasks.manager.mdp.terrain import terrain_flags_from_types

    # 6 patches row-major; 12 envs spread over them (2 envs per patch)
    patch_names = ["flat", "pyramid_stairs", "pyramid_stairs_inv", "slopes", "slopes_inv", "flat"]
    types = torch.arange(12) % 6
    stair, slope = terrain_flags_from_types(patch_names, types, num_envs=12)
    assert not stair[types == 0].any() and not slope[types == 0].any()      # flat: neither
    assert stair[types == 1].all() and stair[types == 2].all()             # stairs (up/inv)
    assert not slope[types == 1].any()                                     # stairs are not slopes
    assert slope[types == 3].all() and slope[types == 4].all()             # slopes (up/inv)
    assert not stair[types == 4].any()
    print("terrain flags OK")


if __name__ == "__main__":
    test_airborne_state_machine()
    test_reward_weight_progression()
    test_terrain_flag_mapping()
    print("STATE MACHINE + CURRICULUM TESTS PASSED")
