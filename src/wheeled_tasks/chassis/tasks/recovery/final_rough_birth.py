"""Final reset requests and the separate nominal V7 axis-heading adapter.

Sampling does not admit a native reset. Complete mechanism mapping, collision
placement, command-limit ownership and curriculum roles remain caller-owned.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from wheeled_tasks.chassis.tasks.recovery.nonsteady_initial_state import FlatBirthRequests, FlatBirthSampler


@dataclass(frozen=True)
class FinalRoughBirthBatch:
    requests: FlatBirthRequests
    birth_kind: torch.Tensor  # int8[K]: ordinary=0, ground=1, air=2.
    air_mode_index: torch.Tensor  # long[K]: -1 outside the active air mode list.
    definition_sha256: str


@dataclass(frozen=True)
class FinalRoughBirthRequests(FlatBirthRequests):
    """Accepted-request metadata carried through the existing reset transaction.

    Negative kinds distinguish nominal(-2) and retained flat(-1) requests.
    Final ordinary/ground/air kinds remain0/1/2. The source definition identity
    belongs to the constructed owner rather than a mutable per-row string.
    """
    birth_kind: torch.Tensor
    air_mode_index: torch.Tensor


class FinalRoughBirthSampler(FlatBirthSampler):
    """Sample the final source distribution with explicit reset permissions.

    FlatBirthRequests remain input to the existing complete-chain mapper. The
    extra class/mode labels must travel through accepted/reset/restart state;
    neither root height nor a nonzero velocity is a substitute for those labels.
    No lane allocation, rejection, native write or command payment occurs here.
    RNG ordering is versioned locally; native DR and command draws are external.
    """
    VERSION = "v7-final-rough-birth-requests-v1"

    def __init__(self, num_envs, device):
        from wheeled_tasks.chassis.tasks.terrain.final_rough_definition import load_final_rough_birth_definition
        super().__init__(num_envs, device)
        definition = load_final_rough_birth_definition()
        if (not all(definition[name] is True for name in (
                "use_leg_random_start", "use_predefined_leg_random_start", "use_joint_vel_random_start"))
                or definition["source_root_init_state"]["rot"] != [1., 0., 0., 0.]):
            raise ValueError("Final birth sampler requires its frozen enabled reset and identity root defaults")
        definition["ground_mode_order"] = list(definition["predefined_reset_ground"]["modes"])
        self._definition_canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def definition_sha256(self):
        return hashlib.sha256(self._definition_canonical.encode()).hexdigest()

    def _range_samples(self, ranges, count, generator):
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        bounds = torch.tensor([ranges.get(key, (0., 0.)) for key in keys],
                              dtype=torch.float32, device=self.device)
        return bounds[:, 0] + torch.rand(count, 6, device=self.device, generator=generator) * (
            bounds[:, 1] - bounds[:, 0])

    @staticmethod
    def _yaw_velocity(samples, yaw):
        # The source rotates only linear velocity by sampled yaw. Angular
        # components already address the world axes used by its root API.
        result = samples.clone()
        c, s = yaw.cos(), yaw.sin()
        result[:, 0] = c * samples[:, 0] - s * samples[:, 1]
        result[:, 1] = s * samples[:, 0] + c * samples[:, 1]
        return result

    def sample(self, ids, generator, *, iteration, air_allowed, ground_allowed, axis_allowed):
        ids = self._ids(ids)
        if (not isinstance(generator, torch.Generator)
                or torch.empty(0, device=generator.device).device != self.device):
            raise ValueError("Final birth sampling requires the caller's device RNG")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Final birth sampling requires an explicit nonnegative reset iteration")
        n = ids.numel()
        for mask in (air_allowed, ground_allowed, axis_allowed):
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.device != self.device or mask.shape != (n,)):
                raise ValueError("Final birth permissions require ordered bool[K] masks on the sampler device")
        d = json.loads(self._definition_canonical)
        defaults = d["source_root_init_state"]
        pose = self._range_samples(d["root_reset_event"]["pose_range"], n, generator)
        position = pose[:, :3] + pose.new_tensor(defaults["pos"])
        rpy = pose[:, 3:].clone()
        velocity = self._yaw_velocity(
            self._range_samples(d["root_reset_event"]["velocity_range"], n, generator), rpy[:, 2])
        velocity += velocity.new_tensor(defaults["lin_vel"] + defaults["ang_vel"])
        length = self._uniform((n, 2), *d["leg_length_range"], generator)
        angle = self._uniform((n, 2), *d["leg_angle_range"], generator)
        raw_air = d["predefined_reset_air"]["modes"]
        raw_air = [raw_air] if isinstance(raw_air, dict) else raw_air
        air_modes = [mode for mode in raw_air if iteration >= mode["iteration_start"]
                     and (mode["iteration_end"] < 0 or iteration < mode["iteration_end"])]
        if not d["predefined_reset_air"]["enabled"]:
            air_modes = []
        ground_cfg = d["predefined_reset_ground"]
        ground_modes = [ground_cfg["modes"][name] for name in d["ground_mode_order"]]
        ground_prob = sum(max(mode["prob"], 0.) for mode in ground_modes)
        draws = self._uniform((n,), 0., 1., generator)
        kind = torch.zeros(n, dtype=torch.int8, device=self.device)
        air_index = torch.full((n,), -1, dtype=torch.long, device=self.device)
        leg_mode = torch.full((n, 2), -1, dtype=torch.int8, device=self.device)
        cursor = 0.
        for index, mode in enumerate(air_modes):
            upper = min(cursor + max(mode["prob"], 0.), 1.)
            selected = (draws >= cursor) & (draws < upper) & air_allowed
            kind[selected], air_index[selected] = 2, index
            cursor = upper
            if cursor >= 1.:
                break
        ground = (draws >= cursor) & (draws < min(cursor + ground_prob, 1.)) & ground_allowed
        kind[ground] = 1
        for index, mode in enumerate(air_modes):
            rows = (air_index == index).nonzero(as_tuple=False).flatten()
            if not len(rows):
                continue
            length[rows] = self._uniform((len(rows), 2), *mode["leg_length_range"], generator)
            angle[rows] = self._uniform((len(rows), 2), *mode["leg_angle_range"], generator)
            air_pose = self._range_samples(mode["pose_range"], len(rows), generator)
            position[rows] = air_pose[:, :3] + position.new_tensor(defaults["pos"])
            rpy[rows] = air_pose[:, 3:]
            velocity[rows] = self._yaw_velocity(
                self._range_samples(mode["velocity_range"], len(rows), generator), air_pose[:, 5])
            velocity[rows] += velocity.new_tensor(defaults["lin_vel"] + defaults["ang_vel"])
        rows = ground.nonzero(as_tuple=False).flatten()
        if len(rows):
            leg_draws = self._uniform((len(rows), 2), 0., ground_prob, generator)
            ground_length, ground_angle = torch.empty_like(length[rows]), torch.empty_like(angle[rows])
            ground_mode = torch.full((len(rows), 2), -1, dtype=torch.int8, device=self.device)
            cursor = 0.
            for index, mode in enumerate(ground_modes):
                upper = cursor + max(mode["prob"], 0.)
                selected = (leg_draws >= cursor) & (leg_draws < upper)
                count = int(selected.sum())
                height = self._uniform((count,), *mode["leg_height"], generator)
                sampled_length = self._uniform((count,), *mode["leg_length"], generator)
                ground_length[selected] = sampled_length
                ground_angle[selected] = mode["sign"] * torch.acos(
                    (height / sampled_length.clamp_min(1e-6)).clamp(-1. + 1e-6, 1. - 1e-6))
                ground_mode[selected] = index
                cursor = upper
            if bool((ground_mode < 0).any()):
                raise RuntimeError("Final ground draw was outside every frozen source interval")
            length[rows], angle[rows], leg_mode[rows] = ground_length, ground_angle, ground_mode
            position[rows] = position.new_tensor(defaults["pos"])
            position[rows, 2] = ground_cfg["start_root_height"]
            rpy[rows] = 0.
            velocity[rows] = velocity.new_tensor(defaults["lin_vel"] + defaults["ang_vel"])
        wheels = self._uniform((n, 2), *d["wheel_angle_range"], generator)
        dq = torch.empty(n, 6, dtype=torch.float32, device=self.device)
        dq[:, [0, 1, 3, 4]] = self._uniform((n, 4), *d["leg_joint_vel_range"], generator)
        dq[:, [2, 5]] = self._uniform((n, 2), *d["wheel_joint_vel_range"], generator)
        # Source yaw alignment follows all body/joint draws and leaves world
        # velocity unchanged. It must not be applied again by the nominal path.
        aligned = axis_allowed.nonzero(as_tuple=False).flatten()
        candidates = rpy.new_tensor(d["reset_heading_axis_aligned_candidates_deg"]) * (math.pi / 180.)
        sampled = torch.randint(len(candidates), (len(aligned),), device=self.device, generator=generator)
        rpy[aligned, 2] = candidates[sampled]
        requests = FlatBirthRequests(ids.clone(), ground, leg_mode, length, angle,
            length * angle.cos(), wheels, dq, position, rpy, velocity)
        return FinalRoughBirthBatch(requests, kind, air_index, self.definition_sha256)

    def summary(self):
        d = json.loads(self._definition_canonical)
        return {"version": self.VERSION, "definition_sha256": self.definition_sha256,
            "source_commit": d["source_commit"], "birth_kind": {"ordinary": 0, "ground": 1, "air": 2},
            "ground_modes": d["predefined_reset_ground"]["modes"],
            "ground_mode_order": d["ground_mode_order"],
            "air_configuration": d["predefined_reset_air"],
            "permission_scope": "explicit ordered reset-origin masks from the terrain command owner",
            "disabled_probability": "ordinary fallback without redistribution",
            "velocity_axes": "linear yaw-frame draw rotated to world; angular world components",
            "root_position": "source default plus event/air offsets; ground z replaced",
            "mechanical_mapping": "requires complete own q18/dq18 and collision placement",
            "native_qualified": False, "production_reset_bound": False,
            "historical_bitwise_replay": False}


def apply_nominal_axis_heading(state, commands, generator, *, nominal_mask=None):
    """Snap eligible nominal poses to the four frozen source headings.

    Velocities, joint states, source requests and unselected poses are kept.
    The enclosing reset writes the pose, refreshes native data, then gives its
    actual heading to command reset. The rough support query must use this
    actual quaternion rather than the original sampling request. A mixed reset
    supplies its explicit nominal subset; other birth owners retain their pose.
    """
    ids, pose = state.env_ids, state.root_pose
    if (not isinstance(generator, torch.Generator)
            or torch.empty(0, device=generator.device).device != pose.device):
        raise ValueError("Nominal axis heading requires the caller's pose-device RNG")
    if tuple(commands.reset_heading_candidates_deg) != (0., 90., 180., -90.):
        raise ValueError("Nominal axis heading requires the frozen four source candidates")
    if nominal_mask is None:
        nominal_mask = torch.ones(len(ids), dtype=torch.bool, device=pose.device)
    if (not isinstance(nominal_mask, torch.Tensor) or nominal_mask.shape != (len(ids),)
            or nominal_mask.dtype != torch.bool or nominal_mask.device != pose.device):
        raise ValueError("Nominal axis heading requires an ordered boolean nominal birth mask")
    if (bool(state.requests.ground_birth[nominal_mask].any())
            or bool((state.requests.source_root_rpy_rad[nominal_mask] != 0.).any())
            or bool((state.root_velocity[nominal_mask] != 0.).any())
            or bool((state.joint_velocity[nominal_mask] != 0.).any())
            or not torch.equal(pose[nominal_mask, 3:7],
                pose.new_tensor([0., 0., 0., 1.]).expand_as(pose[nominal_mask, 3:7]))):
        raise ValueError("Axis adapter accepts only a fresh explicit nominal birth")
    selected = commands.axis_reset_mask(ids, use_env_origins=True)
    if selected.shape != (len(ids),) or selected.dtype != torch.bool or selected.device != pose.device:
        raise ValueError("Nominal axis heading requires an ordered per-birth permission mask")
    selected = selected & nominal_mask
    if not bool(selected.any()):
        return selected
    candidates = pose.new_tensor(commands.reset_heading_candidates_deg) * (math.pi / 180.)
    samples = torch.randint(len(candidates), (int(selected.sum()),), device=pose.device, generator=generator)
    yaw = candidates[samples]
    pose[selected, 5] = (yaw * .5).sin()
    pose[selected, 6] = (yaw * .5).cos()
    return selected
