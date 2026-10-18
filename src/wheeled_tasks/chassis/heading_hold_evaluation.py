"""Explicit evaluation-only initial-heading feedback on fixed straight cases.

The native command owner retains reset, RNG, timing, publication and actuation.
Only this opt-in evaluator attaches instance hooks; training uses no hook.
"""
from __future__ import annotations

import math

import torch


MODE = "initial_heading_gain5_limit2pi_v1"


def install_heading_hold(env, protocol, lane_case_ids):
    if "heading_hold" not in protocol:
        return None
    if (protocol["heading_hold"] != MODE or protocol["mode"] != "fixed_measurement"
            or getattr(env, "physical_tick", None) != 0):
        raise ValueError("Heading feedback requires its explicit fresh fixed evaluation")
    if hasattr(env, "evaluation_heading_hold"):
        raise ValueError("Heading feedback already has an owner")
    owner = env.flat_episode.commands
    cases = {case["case_id"]: case["base_request"] for case in protocol["cases"]}
    selected = torch.tensor([cases[name][0] != 0. and cases[name][1] == 0.
                             for name in lane_case_ids], dtype=torch.bool, device=owner.device)
    if not bool(selected.any()) or not bool(owner.fixed_request_mask.all()):
        raise ValueError("Heading feedback requires installed fixed requests and straight lanes")
    controller = InitialHeadingHold(owner, selected)
    env.evaluation_heading_hold = controller
    env.startup_report["evaluation_heading_hold"] = controller.receipt()
    return controller


class InitialHeadingHold:
    def __init__(self, commands, selected):
        self.commands, self.selected = commands, selected
        self.target = torch.zeros(commands.num_envs, dtype=torch.float32, device=commands.device)
        self.ready = torch.zeros_like(selected)
        self.postprocess_calls = self.reset_calls = 0
        self.last_read_physics_step = None
        self._reset, self._postprocess, self._diagnostics = commands.reset, commands._postprocess, commands.diagnostics
        commands.reset = self.reset
        commands._postprocess = self.postprocess
        commands.diagnostics = self.diagnostics

    def reset(self, ids, heading, ground_mask, generator, iteration=0):
        # Capture the actual new episode heading, never a timed resample target.
        values = self.commands._vector(heading, "heading", torch.float32, ids)
        self._reset(ids, heading, ground_mask, generator, iteration)
        self.target[ids] = values
        self.ready[ids] = True
        self.reset_calls += 1

    def postprocess(self, heading):
        if not bool(self.ready[self.selected].all()):
            raise RuntimeError("Heading feedback has no initial episode target")
        self._postprocess(heading)
        # Keep the same wrapped proportional feedback and bounds. Double
        # arithmetic avoids an artificial sign flip near the +/-pi seam.
        error = self.target[self.selected].double() - heading[self.selected].double()
        wrapped = torch.atan2(torch.sin(error), torch.cos(error))
        yaw = (5. * wrapped).clamp(-2. * math.pi, 2. * math.pi).float()
        self.commands.generator_command3[self.selected, 2] = yaw
        self.commands.generator_after_postprocess3[self.selected, 2] = yaw
        self.commands.heading_target[self.selected] = self.target[self.selected]
        self.commands.is_heading[self.selected] = True
        self.postprocess_calls += 1
        self.last_read_physics_step = self.commands.physics_steps

    def diagnostics(self):
        result = self._diagnostics()
        # Requested straight motion and effective outer-loop correction are
        # distinct. Preserve the original request for case accounting only.
        result["base_request_command3"][self.selected] = self.commands.fixed_request_command3[self.selected]
        return result

    def receipt(self):
        return dict(mode=MODE, enabled_lanes=int(self.selected.sum()),
            target="root-link world yaw captured at each episode reset; unchanged by timed resampling",
            yaw_source="fresh simulated root-link world quaternion; not a delayed hardware IMU heading",
            gain_per_second=5., yaw_limit_rad_s=2. * math.pi, outer_loop_hz=50.,
            update_phase="after previous interval payment, before next observation and actor inference",
            transport="existing actor/PD RTT unchanged; no added yaw transport model",
            base_request="original body-forward request with yaw=0; effective yaw is feedback correction",
            reset_calls=self.reset_calls, postprocess_calls=self.postprocess_calls,
            last_read_physics_step=self.last_read_physics_step,
            training_modified=False, ppo_updates=0)
