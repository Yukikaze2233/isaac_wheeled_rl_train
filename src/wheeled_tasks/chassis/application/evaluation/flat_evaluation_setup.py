"""Typed evaluation overlay applied before the first source-owned reset.

The original training contract keeps its byte identity. This overlay selects
measurement requests and checkpoint actor clocks, with explicit recovery
profile lanes or an optional plane regression surface for rough checkpoints. It does not authorize native
execution or provide mechanical admission. Seed identity is checked at
construction, before startup randomization consumes the environment generator.
"""
from dataclasses import dataclass
import hashlib
import json

import torch

from wheeled_tasks.chassis.application.evaluation.flat_policy_evaluation import validate_protocol

PLANE_REGRESSION_SURFACE = "plane_regression_v1"
# Canonical bytes of the existing fixed and spin225 protocols preserve their
# requests, seeds and horizon. This overlay does not introduce new task cases.
PLANE_REGRESSION_PROTOCOL_SHA256 = "ca9922ea4f3bf96377da0e8bb13b20fa40acac02f27273df34bec2c841521825"
PLANE_REGRESSION_PROTOCOL_ENVS = {
    PLANE_REGRESSION_PROTOCOL_SHA256: 480,
    "a794a982430f0b92f6d3fad9318cb41b3af5a07b5fb18aa9424c648e9e586007": 64,
}


@dataclass(frozen=True, init=False)
class FlatEvaluationSetup:
    _protocol_json: str
    lane_case_ids: tuple[str, ...]
    stage_actor_update: int
    global_actor_update: int
    run_role: str
    evaluation_surface: str

    def __init__(self, protocol, lane_case_ids, stage_actor_update, global_actor_update,
                 *, run_role="source_flat_evaluation", evaluation_surface="training"):
        from wheeled_tasks.chassis.application.evaluation.small_rough_evaluation import SURFACE, validate_overlay as validate_small_rough
        checked = validate_protocol(protocol)
        from wheeled_tasks.chassis.application.evaluation.scripted_recovery_evaluation import ROLE, selected, scene_groups
        if selected(checked) != (run_role == ROLE):
            raise ValueError("Recovery evaluation requires its explicit protocol and role together")
        if selected(checked):
            scene_groups(checked, lane_case_ids)
        if run_role not in ("source_flat_evaluation", "source_rough_evaluation", "source_rough_plane_regression", ROLE):
            raise ValueError("Unknown source evaluation role")
        encoded = json.dumps(checked, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if evaluation_surface not in ("training", PLANE_REGRESSION_SURFACE, SURFACE):
            raise ValueError("Unknown evaluation surface")
        plane = evaluation_surface == PLANE_REGRESSION_SURFACE
        if plane != (run_role == "source_rough_plane_regression"):
            raise ValueError("Plane regression requires its explicit surface and role together")
        plane_envs = PLANE_REGRESSION_PROTOCOL_ENVS.get(hashlib.sha256(encoded.encode()).hexdigest())
        if plane and plane_envs is None:
            raise ValueError("Plane regression requires the unchanged foundation 15-case or spin225 protocol")
        if run_role == "source_rough_evaluation" and checked["mode"] != "source_distribution":
            raise ValueError("Rough evaluation requires the unchanged source command distribution")
        if evaluation_surface == SURFACE:
            if run_role != "source_rough_evaluation":
                raise ValueError("Small rough regression requires the matching rough evaluator")
            validate_small_rough(checked, dict(num_envs=len(lane_case_ids), lane_case_ids=list(lane_case_ids),
                record_diagnostics=True, evaluation_surface=SURFACE))
        if (not isinstance(lane_case_ids, tuple) or not lane_case_ids
                or any(not isinstance(case_id, str) for case_id in lane_case_ids)):
            raise ValueError("Evaluation lanes require an explicit nonempty tuple of case IDs")
        cases = {c["case_id"] for c in checked.get("cases", [])} or {"source_distribution"}
        if set(lane_case_ids) != cases:
            raise ValueError("Evaluation lanes must cover exactly the declared cases")
        if plane and (len(lane_case_ids) != plane_envs or any(lane_case_ids.count(name) != 32 for name in cases)):
            raise ValueError(f"Plane regression preserves the existing {len(cases)} by 32 episode allocation")
        for value in (stage_actor_update, global_actor_update):
            if type(value) is not int or value < 0:
                raise ValueError("Evaluation actor clocks must be nonnegative integers")
        if stage_actor_update > global_actor_update:
            raise ValueError("Stage actor clock exceeds the global checkpoint actor clock")
        object.__setattr__(self, "_protocol_json", encoded)
        object.__setattr__(self, "lane_case_ids", lane_case_ids)
        object.__setattr__(self, "stage_actor_update", stage_actor_update)
        object.__setattr__(self, "global_actor_update", global_actor_update)
        object.__setattr__(self, "run_role", run_role)
        object.__setattr__(self, "evaluation_surface", evaluation_surface)

    @property
    def protocol(self):
        # A caller cannot mutate the bound protocol through this property.
        return json.loads(self._protocol_json)

    def validate_constructor(self, config, num_envs, seed, run_role):
        from wheeled_tasks.chassis.application.configuration.flat_episode_selection import validate_flat_episode_selection
        from wheeled_tasks.chassis.application.configuration.reward_selection import SCUT_ROUGH_PROFILES
        if validate_flat_episode_selection(config) is None:
            raise ValueError("Typed flat evaluation requires the source-owned flat episode contract")
        expected_role = ("source_rough_evaluation" if config.get("reward_profile") in SCUT_ROUGH_PROFILES
                         else "source_flat_evaluation")
        from wheeled_tasks.chassis.application.evaluation.scripted_recovery_evaluation import ROLE, selected, scene_groups
        if selected(self.protocol):
            from wheeled_tasks.chassis.application.configuration.flat_episode_selection import validate_scut_flat_recovery
            if (validate_scut_flat_recovery(config) is None
                    or config.get("scene_groups") != scene_groups(self.protocol, self.lane_case_ids)):
                raise ValueError("Recovery constructor requires the explicit protocol scene/profile allocation")
            expected_role = ROLE
        if self.evaluation_surface == PLANE_REGRESSION_SURFACE:
            if config.get("reward_profile") not in SCUT_ROUGH_PROFILES:
                raise ValueError("Plane regression overlay requires the authenticated rough profile")
            expected_role = "source_rough_plane_regression"
        if run_role != self.run_role or run_role != expected_role:
            raise ValueError("Typed source evaluation requires its matching independent run role")
        if type(num_envs) is not int or num_envs != len(self.lane_case_ids):
            raise ValueError("Environment lane count differs from the evaluation overlay")
        protocol = self.protocol
        if type(seed) is not int or seed != protocol["environment_seed"]:
            raise ValueError("Constructor seed differs from the evaluation environment seed")
        expected = {"record_diagnostics": True, "auto_reset": True,
                    "physics_dt": .005, "policy_dt": .02, "episode_seconds": 20.}
        for key, value in expected.items():
            actual = config.get(key, True if key == "auto_reset" else None)
            matching_type = type(actual) is bool if type(value) is bool else type(actual) in (int, float)
            if not matching_type or actual != value:
                raise ValueError("Source evaluation constructor differs: " + key)
        if (config.get("evaluation_exact_cases") or config.get("integrated_runtime_candidate")
                or config.get("integrated_production") or config.get("scut_heading_command", {}).get("enabled")):
            raise ValueError("Typed source flat evaluator cannot select a legacy evaluation/command owner")

    def apply_before_first_reset(self, env):
        self.validate_constructor(env.cfg, env.num_envs, env.cfg["seed"], self.run_role)
        flat = getattr(env, "flat_episode", None)
        if (flat is None or env.physical_tick != 0 or flat.commands.physics_steps != 0
                or bool((flat.episode_id != -1).any()) or bool(env.episode_length_buf.any())):
            raise ValueError("Evaluation setup must precede the first physical/reset episode")
        if env.generator.initial_seed() != self.protocol["environment_seed"]:
            raise ValueError("Startup environment generator seed differs from the overlay")
        from wheeled_tasks.chassis.application.evaluation.scripted_recovery_evaluation import selected, actual_lane_cases
        lanes = actual_lane_cases(self, env) if selected(self.protocol) else self.lane_case_ids
        if self.protocol["mode"] == "fixed_measurement":
            cases = {c["case_id"]: c["base_request"] for c in self.protocol["cases"]}
            requests = torch.tensor([cases[name] for name in lanes],
                dtype=torch.float32, device=flat.device)
            command3 = torch.stack((requests[:, 0], torch.zeros_like(requests[:, 0]), requests[:, 1]), -1)
            from wheeled_tasks.chassis.application.evaluation.temporal_flat_evaluation import selected as temporal_selected, install
            if temporal_selected(self.protocol):
                install(flat.commands, requests)
            else:
                flat.commands.set_fixed_requests(torch.arange(env.num_envs, device=flat.device),
                    command3, requests[:, 2], evaluation_only=True)
            if "heading_hold" in self.protocol:
                from wheeled_tasks.chassis.application.evaluation.heading_hold_evaluation import install_heading_hold
                install_heading_hold(env, self.protocol, lanes)
        env.stage_actor_update = self.stage_actor_update
        env.global_actor_update = self.global_actor_update
        # No manual_seed here: interval/PC startup draws already belong to
        # this generator. Record them rather than restarting their stream.
        rng = env.generator.get_state().cpu().numpy().tobytes()
        env.startup_report["flat_evaluation_setup"] = {
            "assessment": "measurement_only", "native_admission_granted": False,
            "protocol_sha256": hashlib.sha256(self._protocol_json.encode()).hexdigest(),
            "lane_case_ids": list(self.lane_case_ids), "actual_lane_case_ids": list(lanes), "environment_seed": self.protocol["environment_seed"],
            "actor_seed": self.protocol["actor_seed"], "actor_mode": self.protocol["actor_mode"],
            "run_role": self.run_role,
            "evaluation_surface": self.evaluation_surface,
            "stage_actor_update": self.stage_actor_update, "global_actor_update": self.global_actor_update,
            "environment_rng_pre_first_reset_sha256": hashlib.sha256(rng).hexdigest(),
            "environment_rng_reseeded_after_startup": False}
