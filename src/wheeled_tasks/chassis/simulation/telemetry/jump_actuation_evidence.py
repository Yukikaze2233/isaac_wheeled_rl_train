"""Read-only 5 ms mechanical measurements, never an active-takeoff classifier.

Native explicit-applied motor Nm and prismatic spring N are separate sources.
Work uses a declared right-endpoint power quadrature. Contact sampled at either
end cannot locate a contact change within the 5 ms interval; both sampled support
windows are reported, without treating either as continuous-contact proof.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from wheeled_tasks.chassis.control.actuator_response import JOINT_NAMES


SCHEMA = "jump-actuation-measurement-v1"
EVENTS = ("accepted_request", "raw_wheel_support_lost", "raw_wheel_support_regained")
SOURCES = {"schedule": 1, "step_jump": 2}
REQUEST_KINDS = {"schedule_mode4_jump": 1, "schedule_manual_step_reference": 2, "step_jump": 3}
WINDOWS = ("all", "support_at_interval_start", "support_at_interval_end")
SCOPES = ("episode", "request", "policy")
MOMENTS = ("signed", "positive", "negative_magnitude")


class JumpActuationEvidence:
    """Observe a worker's actual samples; no commands, state writes or credit.

    Call ``reset`` at each actual physical episode reset; ``accept_request`` at
    the source's real accepted edge; ``begin_policy`` before physical substeps;
    and ``policy_transition`` before reset. A returned checkpoint is a measurement
    record only: there is deliberately no exact-simulator-resume loader.
    """

    def __init__(self, count, device, *, physics_dt, control_joint_names,
                 spring_joint_names, source_identity):
        if type(count) is not int or count <= 0 or type(physics_dt) not in (int, float) or physics_dt != .005:
            raise ValueError("Jump measurements require positive count and explicit 5 ms physics")
        if tuple(control_joint_names) != JOINT_NAMES:
            raise ValueError("Jump measurements require native canonical CONTROL six-axis order")
        if (not isinstance(spring_joint_names, (list, tuple)) or len(spring_joint_names) != 2
                or any(not isinstance(n, str) or not n for n in spring_joint_names)
                or len(set(spring_joint_names)) != 2 or set(spring_joint_names) & set(JOINT_NAMES)):
            raise ValueError("Two distinct named prismatic spring axes are required")
        if not isinstance(source_identity, dict) or not source_identity:
            raise ValueError("Explicit configuration and callsite source identity required")
        self.count, self.device, self.dt = count, torch.device(device), float(physics_dt)
        self._identity = {
            "schema": "jump-actuation-source-v1", "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "configuration_and_callsite": deepcopy(source_identity),
            "physics_dt_s": self.dt, "policy_dt_s": .02,
            "control_joint_names": list(control_joint_names), "spring_joint_names": list(spring_joint_names),
            "motor_effort": "native_explicit_applied_motor_torque_Nm_not_requested_or_total_solver_effort",
            "spring_effort": "native_explicit_applied_prismatic_force_N_separate_from_active_motors",
            "work_quadrature": "right_endpoint_applied_effort_times_native_velocity_times_dt_once",
            "geometry": "native_collision_wheel_center_minus_base_link_origin_in_base_link_frame",
            "com": "caller_native_body_mass_weighted_whole_robot_COM_in_world",
            "support_windows": "start_and_end_sampled_wheel_support_separate_not_continuous_contact_proof",
            "external_intervention_observation": "not_observed_by_this_producer",
            "qualification": "measurement_only_no_active_takeoff_or_skill_pass",
        }
        encoded = json.dumps(self._identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.source_identity_sha256 = hashlib.sha256(encoded).hexdigest()
        self._last_tick = None
        self._open = False
        self._last_transition = None
        self.episode_ids = self._long(-1)
        self.request_ids = self._long(-1)
        self.source_request_ids = self._long(-1)
        self.request_source = self._long(0)
        self.request_kind_ids = self._long(0)
        self.route_stage_ids = self._long(-1)
        self.accepted_tick = self._long(-1)
        self._source_highwater = torch.zeros(count, 2, dtype=torch.long, device=self.device)
        self._accepted_pending = self._long(0)
        self._accepted_pending_tick = self._long(-1)
        self._events = {name: self._long(0) for name in EVENTS}
        self._event_first = {name: self._long(-1) for name in EVENTS}
        self._event_last = {name: self._long(-1) for name in EVENTS}
        self._work = {scope: {window: {"motor": self._float(6, 3), "spring": self._float(2, 3)}
            for window in WINDOWS} for scope in SCOPES}
        self._samples = {scope: self._long(0) for scope in SCOPES}
        self._invalid = {scope: self._long(0) for scope in SCOPES}
        self._start_unknown = {scope: self._long(0) for scope in SCOPES}
        self._air_samples = {scope: self._long(0) for scope in SCOPES}
        self._raw_counts = {scope: {name: self._long(0) for name in EVENTS[1:]} for scope in SCOPES}
        self._previous_valid = self._bool()
        self._previous_contact = self._bool(2)
        self._current_valid = self._bool()
        self._current_contact = self._bool(2)
        self._current_nonwheel = self._bool()
        self._request_active = self._bool()
        self._legacy_phase = self._long(-1)
        self._wheel_b = self._float(2, 3)
        self._previous_wheel_b = self._float(2, 3)
        self._geometry_change = self._float(2, 3)
        self._geometry_change_valid = self._bool()
        self._com_position = self._float(3)
        self._com_velocity = self._float(3)
        self._motor_power = self._float(6)
        self._spring_power = self._float(2)
        self._request_origin_valid = self._bool()
        self._request_origin_tick = self._long(-1)
        self._request_wheel_origin = self._float(2, 3)
        self._request_com_origin = self._float(3)
        self._first_loss_seen = self._bool()
        self._first_loss_tick = self._long(-1)
        self._first_loss_invalid = self._long(0)
        self._first_loss_start_unknown = self._long(0)
        self._first_loss_wheel_delta = self._float(2, 3)
        self._first_loss_com = self._float(3)
        self._first_loss_velocity = self._float(3)
        self._first_loss_work = {window: {"motor": self._float(6, 3), "spring": self._float(2, 3)} for window in WINDOWS}
        self._flight_id = self._long(-1)
        self._flight_active = self._bool()
        self._flight_measurement_valid = self._bool()
        self._flight_request_id = self._long(-1)
        self._liftoff_tick = self._long(-1)
        self._landing_tick = self._long(-1)
        self._landing_request_id = self._long(-1)
        self._liftoff_com = self._float(3)
        self._liftoff_velocity = self._float(3)
        self._landing_com = self._float(3)
        self._landing_velocity = self._float(3)
        self._flight_peak_z = self._float()
        self._air_run = self._long(0)
        self._air_peak = self._long(0)

    def _long(self, fill):
        return torch.full((self.count,), fill, dtype=torch.long, device=self.device)

    def _bool(self, *tail):
        return torch.zeros((self.count, *tail), dtype=torch.bool, device=self.device)

    def _float(self, *tail):
        return torch.zeros((self.count, *tail), dtype=torch.float64, device=self.device)

    def _tensor(self, name, value, shape, dtype=None):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.device != self.device or (value.dtype != dtype if dtype is not None else not value.is_floating_point())):
            raise ValueError(f"{name} has wrong shape/dtype/device")

    def _tick(self, tick):
        if type(tick) is not int or tick < 0:
            raise ValueError("physical_tick must be a nonnegative integer")

    def _ids(self, ids):
        self._tensor("ids", ids, (len(ids),), torch.long)
        if bool(((ids < 0) | (ids >= self.count)).any()) or len(ids.unique()) != len(ids):
            raise ValueError("Episode/request ids must be unique valid lanes")

    def reset(self, ids, *, physical_tick):
        self._tick(physical_tick)
        self._ids(ids)
        if self._last_tick is not None and physical_tick != self._last_tick:
            raise ValueError("Reset must use the current completed physical tick")
        if self._open:
            raise ValueError("Take pre-reset policy_transition before resetting a paid interval")
        self._last_tick = physical_tick
        self.episode_ids[ids] += 1
        for tensor in (self.request_ids, self.source_request_ids, self.route_stage_ids, self.accepted_tick,
                       self._accepted_pending_tick, self._request_origin_tick, self._flight_id,
                       self._flight_request_id, self._liftoff_tick, self._landing_tick, self._landing_request_id):
            tensor[ids] = -1
        for tensor in (self.request_source, self.request_kind_ids, self._source_highwater, self._accepted_pending,
                       self._previous_valid, self._previous_contact, self._current_valid,
                       self._current_contact, self._current_nonwheel, self._request_active,
                       self._geometry_change_valid, self._request_origin_valid, self._flight_active, self._flight_measurement_valid,
                       self._air_run, self._air_peak):
            tensor[ids] = 0
        self._legacy_phase[ids] = -1
        for tensor in (self._wheel_b, self._previous_wheel_b, self._geometry_change,
                       self._com_position, self._com_velocity, self._motor_power, self._spring_power,
                       self._request_wheel_origin, self._request_com_origin, self._liftoff_com,
                       self._liftoff_velocity, self._landing_com, self._landing_velocity, self._flight_peak_z):
            tensor[ids] = 0
        for scope in SCOPES:
            self._clear_scope(scope, ids)
        self._clear_first_loss(ids)
        for name in EVENTS:
            self._events[name][ids] = 0
            self._event_first[name][ids] = self._event_last[name][ids] = -1

    def _clear_scope(self, scope, ids=None):
        for window in WINDOWS:
            for tensor in self._work[scope][window].values():
                tensor.zero_() if ids is None else tensor.__setitem__(ids, 0)
        for counters in (self._samples, self._invalid, self._start_unknown, self._air_samples):
            counters[scope].zero_() if ids is None else counters[scope].__setitem__(ids, 0)
        for tensor in self._raw_counts[scope].values():
            tensor.zero_() if ids is None else tensor.__setitem__(ids, 0)

    def _clear_first_loss(self, ids):
        for tensor in (self._first_loss_seen, self._first_loss_invalid, self._first_loss_start_unknown,
                       self._first_loss_wheel_delta, self._first_loss_com, self._first_loss_velocity):
            tensor[ids] = 0
        self._first_loss_tick[ids] = -1
        for window in WINDOWS:
            for tensor in self._first_loss_work[window].values():
                tensor[ids] = 0

    def begin_policy(self):
        if self._open or self._last_tick is None:
            raise ValueError("Policy interval requires reset and a closed previous interval")
        self._clear_scope("policy")
        for name in EVENTS:
            self._events[name].zero_()
            self._event_first[name].fill_(-1)
            self._event_last[name].fill_(-1)
        self._events["accepted_request"].copy_(self._accepted_pending)
        self._event_first["accepted_request"].copy_(self._accepted_pending_tick)
        self._event_last["accepted_request"].copy_(self._accepted_pending_tick)
        self._accepted_pending.zero_()
        self._accepted_pending_tick.fill_(-1)
        self._open = True
        self._last_transition = None

    def accept_request(self, ids, *, source, request_kind, source_request_ids, physical_tick, route_stage_ids=None):
        self._tick(physical_tick)
        self._ids(ids)
        if source not in SOURCES or self._last_tick != physical_tick:
            raise ValueError("Accepted request needs a known real source and current physical tick")
        if (request_kind not in REQUEST_KINDS or (source == "step_jump") != (request_kind == "step_jump")):
            raise ValueError("Request kind must identify the actual schedule jump/manual step or StepJump source")
        self._tensor("source_request_ids", source_request_ids, (len(ids),), torch.long)
        if route_stage_ids is None:
            route_stage_ids = torch.full_like(source_request_ids, -1)
        self._tensor("route_stage_ids", route_stage_ids, (len(ids),), torch.long)
        if bool(((source_request_ids <= 0) | (route_stage_ids < -1) | (self.episode_ids[ids] < 0)).any()):
            raise ValueError("Accepted source counts/route/episode identity invalid")
        code = SOURCES[source]
        previous = self._source_highwater[ids, code - 1]
        duplicate = source_request_ids == previous
        same = ((self.request_source[ids] == code) & (self.source_request_ids[ids] == source_request_ids)
                & (self.accepted_tick[ids] == physical_tick) & (self.route_stage_ids[ids] == route_stage_ids)
                & (self.request_kind_ids[ids] == REQUEST_KINDS[request_kind]))
        if bool(((source_request_ids < previous) | (duplicate & ~same)).any()):
            raise ValueError("Stale or changed accepted-request identity")
        selected = ~duplicate
        lanes = ids[selected]
        if bool((self.accepted_tick[lanes] == physical_tick).any()):
            raise ValueError("Conflicting accepted requests at one physical boundary")
        self._source_highwater[lanes, code - 1] = source_request_ids[selected]
        self.request_ids[lanes] += 1
        self.source_request_ids[lanes] = source_request_ids[selected]
        self.route_stage_ids[lanes] = route_stage_ids[selected]
        self.request_source[lanes] = code
        self.request_kind_ids[lanes] = REQUEST_KINDS[request_kind]
        self.accepted_tick[lanes] = physical_tick
        self._clear_scope("request", lanes)
        self._clear_first_loss(lanes)
        self._request_origin_valid[lanes] = self._current_valid[lanes]
        self._request_origin_tick[lanes] = torch.where(self._current_valid[lanes], physical_tick, -1)
        self._request_wheel_origin[lanes] = self._wheel_b[lanes]
        self._request_com_origin[lanes] = self._com_position[lanes]
        if self._open:
            pulse = self._bool()
            pulse[lanes] = True
            self._event("accepted_request", pulse, physical_tick)
        else:
            self._accepted_pending[lanes] += 1
            self._accepted_pending_tick[lanes] = physical_tick

    def _event(self, name, pulse, tick):
        first = pulse & (self._events[name] == 0)
        self._events[name] += pulse.long()
        self._event_first[name] = torch.where(first, tick, self._event_first[name])
        self._event_last[name] = torch.where(pulse, tick, self._event_last[name])

    def observe_physics(self, *, physical_tick, applied_motor_nm, motor_velocity_rad_s,
                        applied_spring_n, spring_velocity_m_s, base_link_pose_world,
                        wheel_center_world, com_position_world, com_velocity_world,
                        wheel_contacts, nonwheel_contact, contact_measurement_valid,
                        legacy_phase, request_active):
        self._tick(physical_tick)
        if not self._open or physical_tick != self._last_tick + 1:
            raise ValueError("Exactly one successive physical sample per open policy interval required")
        n = self.count
        numeric = (("applied_motor_nm", applied_motor_nm, (n, 6)),
            ("motor_velocity_rad_s", motor_velocity_rad_s, (n, 6)),
            ("applied_spring_n", applied_spring_n, (n, 2)),
            ("spring_velocity_m_s", spring_velocity_m_s, (n, 2)),
            ("base_link_pose_world", base_link_pose_world, (n, 7)),
            ("wheel_center_world", wheel_center_world, (n, 2, 3)),
            ("com_position_world", com_position_world, (n, 3)),
            ("com_velocity_world", com_velocity_world, (n, 3)))
        for name, tensor, shape in numeric:
            self._tensor(name, tensor, shape)
        for name, tensor, shape in (("wheel_contacts", wheel_contacts, (n, 2)),
                ("nonwheel_contact", nonwheel_contact, (n,)), ("request_active", request_active, (n,)),
                ("contact_measurement_valid", contact_measurement_valid, (n,))):
            self._tensor(name, tensor, shape, torch.bool)
        self._tensor("legacy_phase", legacy_phase, (n,), torch.long)
        valid = (self.episode_ids >= 0) & contact_measurement_valid
        for _, tensor, _ in numeric:
            valid = valid & torch.isfinite(tensor).reshape(n, -1).all(-1)
        pose = base_link_pose_world.double()
        quat = pose[:, 3:]
        norm = quat.norm(dim=-1)
        valid &= norm > 0
        quat = quat / norm.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
        inverse_xyz = -quat[:, None, :3]
        delta = wheel_center_world.double() - pose[:, None, :3]
        cross = torch.linalg.cross(inverse_xyz.expand(-1, 2, -1), delta)
        wheel_b = delta + 2 * (quat[:, None, 3:] * cross + torch.linalg.cross(inverse_xyz.expand(-1, 2, -1), cross))
        motor_power = applied_motor_nm.double() * motor_velocity_rad_s.double()
        spring_power = applied_spring_n.double() * spring_velocity_m_s.double()
        valid &= torch.isfinite(wheel_b).all(-1).all(-1) & torch.isfinite(motor_power).all(-1) & torch.isfinite(spring_power).all(-1)
        supported = wheel_contacts.any(-1)
        previous_supported = self._previous_contact.any(-1)
        edge_valid = valid & self._previous_valid
        lost = edge_valid & previous_supported & ~supported
        regained = edge_valid & ~previous_supported & supported
        self._event("raw_wheel_support_lost", lost, physical_tick)
        self._event("raw_wheel_support_regained", regained, physical_tick)
        windows = {"all": valid, "support_at_interval_start": edge_valid & previous_supported,
                   "support_at_interval_end": valid & supported}
        work_by_axis = {key: torch.stack((power, power.clamp_min(0.), (-power).clamp_min(0.)), -1) * self.dt
                       for key, power in (("motor", motor_power), ("spring", spring_power))}
        for scope in SCOPES:
            selected = (self.request_ids >= 0) if scope == "request" else torch.ones_like(valid)
            self._samples[scope] += selected.long()
            self._invalid[scope] += (selected & ~valid).long()
            self._start_unknown[scope] += (selected & ~edge_valid).long()
            self._air_samples[scope] += (selected & valid & ~supported).long()
            for name, pulse in (("raw_wheel_support_lost", lost), ("raw_wheel_support_regained", regained)):
                self._raw_counts[scope][name] += (selected & pulse).long()
            for window, window_valid in windows.items():
                mask = (selected & window_valid)[:, None, None]
                for key, work in work_by_axis.items():
                    self._work[scope][window][key] += torch.where(mask, work, 0.)
        clean_wheel = torch.where(valid[:, None, None], wheel_b, 0.)
        self._geometry_change = torch.where(edge_valid[:, None, None], clean_wheel - self._previous_wheel_b, 0.)
        self._geometry_change_valid.copy_(edge_valid)
        self._wheel_b.copy_(clean_wheel)
        self._com_position.copy_(torch.where(valid[:, None], com_position_world.double(), 0.))
        self._com_velocity.copy_(torch.where(valid[:, None], com_velocity_world.double(), 0.))
        origin = valid & (self.request_ids >= 0) & ~self._request_origin_valid
        self._request_wheel_origin = torch.where(origin[:, None, None], self._wheel_b, self._request_wheel_origin)
        self._request_com_origin = torch.where(origin[:, None], self._com_position, self._request_com_origin)
        self._request_origin_tick = torch.where(origin, physical_tick, self._request_origin_tick)
        self._request_origin_valid |= origin
        first_loss = lost & (self.request_ids >= 0) & ~self._first_loss_seen
        self._first_loss_seen |= first_loss
        self._first_loss_tick = torch.where(first_loss, physical_tick, self._first_loss_tick)
        self._first_loss_invalid = torch.where(first_loss, self._invalid["request"], self._first_loss_invalid)
        self._first_loss_start_unknown = torch.where(first_loss, self._start_unknown["request"], self._first_loss_start_unknown)
        self._first_loss_wheel_delta = torch.where(first_loss[:, None, None],
            self._wheel_b - self._request_wheel_origin, self._first_loss_wheel_delta)
        self._first_loss_com = torch.where(first_loss[:, None], self._com_position, self._first_loss_com)
        self._first_loss_velocity = torch.where(first_loss[:, None], self._com_velocity, self._first_loss_velocity)
        for window in WINDOWS:
            for key in ("motor", "spring"):
                self._first_loss_work[window][key] = torch.where(first_loss[:, None, None],
                    self._work["request"][window][key], self._first_loss_work[window][key])
        self._motor_power.copy_(torch.where(valid[:, None], motor_power, 0.))
        self._spring_power.copy_(torch.where(valid[:, None], spring_power, 0.))
        self._flight_id += lost.long()
        self._flight_request_id = torch.where(lost, self.request_ids, self._flight_request_id)
        self._liftoff_tick = torch.where(lost, physical_tick, self._liftoff_tick)
        self._liftoff_com = torch.where(lost[:, None], self._com_position, self._liftoff_com)
        self._liftoff_velocity = torch.where(lost[:, None], self._com_velocity, self._liftoff_velocity)
        self._flight_peak_z = torch.where(lost, self._com_position[:, 2], self._flight_peak_z)
        self._flight_measurement_valid = (self._flight_measurement_valid | lost) & (~self._flight_active | valid)
        self._flight_active |= lost
        tracked_air = self._flight_active & valid & ~supported
        self._flight_peak_z = torch.where(tracked_air, torch.maximum(self._flight_peak_z, self._com_position[:, 2]), self._flight_peak_z)
        self._landing_tick = torch.where(regained, physical_tick, self._landing_tick)
        self._landing_request_id = torch.where(regained,
            torch.where(self._flight_active, self._flight_request_id, -1), self._landing_request_id)
        self._landing_com = torch.where(regained[:, None], self._com_position, self._landing_com)
        self._landing_velocity = torch.where(regained[:, None], self._com_velocity, self._landing_velocity)
        self._flight_active &= ~regained & valid
        self._air_run = torch.where(valid & ~supported, self._air_run + 1, 0)
        self._air_peak = torch.maximum(self._air_peak, self._air_run)
        self._current_valid.copy_(valid)
        self._current_contact.copy_(wheel_contacts)
        self._current_nonwheel.copy_(nonwheel_contact)
        self._request_active.copy_(request_active)
        self._legacy_phase.copy_(legacy_phase)
        self._previous_valid.copy_(valid)
        self._previous_contact.copy_(wheel_contacts)
        self._previous_wheel_b.copy_(clean_wheel)
        self._last_tick = physical_tick

    def _work_snapshot(self, scope):
        result = {}
        data = self._first_loss_work if scope == "first_support_loss" else self._work[scope]
        for window in WINDOWS:
            motor, spring = (data[window][name] for name in ("motor", "spring"))
            result[window] = {"motor_work_j": motor.clone(), "spring_work_j": spring.clone(),
                "four_leg_motor_work_j": motor[:, (0, 1, 3, 4)].sum(1),
                "two_wheel_motor_work_j": motor[:, (2, 5)].sum(1)}
        return result

    def policy_transition(self, done, terminated, timeouts):
        for name, tensor in (("done", done), ("terminated", terminated), ("timeouts", timeouts)):
            self._tensor(name, tensor, (self.count,), torch.bool)
        if not self._open:
            if self._last_transition is None:
                raise ValueError("No elapsed policy interval to snapshot")
            return deepcopy(self._last_transition)
        result = {"schema": SCHEMA, "source_identity_sha256": self.source_identity_sha256,
            "physical_tick": self._last_tick, "physics_dt_s": self.dt,
            "scope": "measurement_only", "work_moment_order": list(MOMENTS),
            "request_source_codes": {"none": 0, **SOURCES},
            "request_kind_codes": {"none": 0, **REQUEST_KINDS},
            **{name: getattr(self, name).clone() for name in ("episode_ids", "request_ids", "source_request_ids",
                "request_source", "request_kind_ids", "route_stage_ids", "accepted_tick")},
            "request_active": self._request_active.clone(), "legacy_phase": self._legacy_phase.clone(),
            "measurement_valid": self._current_valid.clone(),
            "policy_physics_samples": self._samples["policy"].clone(),
            "policy_event_counts": {name: value.clone() for name, value in self._events.items()},
            "event_first_tick": {name: value.clone() for name, value in self._event_first.items()},
            "event_last_tick": {name: value.clone() for name, value in self._event_last.items()},
            "work": {scope: self._work_snapshot(scope) for scope in SCOPES},
            "request_first_support_loss": {"observed": self._first_loss_seen.clone(),
                "physical_tick": self._first_loss_tick.clone(),
                "work_since_accept_including_edge_interval": self._work_snapshot("first_support_loss"),
                "invalid_samples": self._first_loss_invalid.clone(), "interval_start_unknown_samples": self._first_loss_start_unknown.clone(),
                "wheel_center_base_frame_delta_m": self._first_loss_wheel_delta.clone(),
                "com_position_world_m": self._first_loss_com.clone(), "com_velocity_world_m_s": self._first_loss_velocity.clone()},
            "sample_counts": {scope: {"observed": self._samples[scope].clone(), "invalid": self._invalid[scope].clone(),
                "interval_start_unknown": self._start_unknown[scope].clone(), "raw_wheel_air": self._air_samples[scope].clone()}
                for scope in SCOPES},
            "raw_event_counts": {scope: {name: value.clone() for name, value in self._raw_counts[scope].items()} for scope in SCOPES},
            "geometry": {"wheel_center_base_frame_m": self._wheel_b.clone(),
                "left_right_base_z_extension_m": -self._wheel_b[:, :, 2],
                "left_right_base_to_center_distance_m": self._wheel_b.norm(dim=-1),
                "delta_wheel_center_base_frame_m": self._geometry_change.clone(),
                "delta_valid": self._geometry_change_valid.clone(),
                "request_origin_valid": self._request_origin_valid.clone(), "request_origin_sample_tick": self._request_origin_tick.clone(),
                "request_delta_wheel_center_base_frame_m": self._wheel_b - self._request_wheel_origin},
            "power": {"motor_w": self._motor_power.clone(), "spring_w": self._spring_power.clone()},
            "contact": {"wheel": self._current_contact.clone(), "nonwheel": self._current_nonwheel.clone()},
            "com_position_world_m": self._com_position.clone(), "com_velocity_world_m_s": self._com_velocity.clone(),
            "request_com_displacement_world_m": self._com_position - self._request_com_origin,
            "raw_flight": {"id": self._flight_id.clone(), "active": self._flight_active.clone(),
                "measurement_valid": self._flight_measurement_valid.clone(),
                "liftoff_request_id": self._flight_request_id.clone(), "liftoff_tick": self._liftoff_tick.clone(),
                "liftoff_com_position_world_m": self._liftoff_com.clone(), "liftoff_com_velocity_world_m_s": self._liftoff_velocity.clone(),
                "landing_tick": self._landing_tick.clone(), "landing_request_id": self._landing_request_id.clone(),
                "landing_com_position_world_m": self._landing_com.clone(), "landing_com_velocity_world_m_s": self._landing_velocity.clone(),
                "sampled_com_rise_m": (self._flight_peak_z - self._liftoff_com[:, 2]).clamp_min(0.),
                "raw_wheel_air_run_s": self._air_run.double() * self.dt, "raw_wheel_air_peak_s": self._air_peak.double() * self.dt},
            "done": done.clone(), "terminated": terminated.clone(), "timeouts": timeouts.clone()}
        self._last_transition = deepcopy(result)
        self._open = False
        return result

    def source_identity(self):
        return deepcopy(self._identity)

    def metrics(self):
        """Stable tensor tags for the existing logger; no capability score."""
        work = self._work["policy"]["all"]
        return {"/jump_actuation/measurement_only/policy_physics_samples": self._samples["policy"].double().sum(),
            "/jump_actuation/measurement_only/invalid_samples": self._invalid["policy"].double().sum(),
            "/jump_actuation/measurement_only/four_leg_positive_work_j": work["motor"][:, (0, 1, 3, 4), 1].sum(),
            "/jump_actuation/measurement_only/two_wheel_positive_work_j": work["motor"][:, (2, 5), 1].sum(),
            "/jump_actuation/measurement_only/spring_positive_work_j": work["spring"][:, :, 1].sum(),
            **{f"/jump_actuation/measurement_only/{name}_count": value.double().sum() for name, value in self._events.items()}}

    def summary(self):
        """Current worker summary; live-episode sums are not lifetime totals."""
        return {"schema": "jump-actuation-worker-summary-v1", "source_identity_sha256": self.source_identity_sha256,
            "source_identity": self.source_identity(), "physical_tick": self._last_tick,
            "scope": "worker_measurement_only_no_exact_resume", "open_policy_interval": self._open,
            "episode_ids": self.episode_ids.detach().cpu().tolist(),
            "current_episode_samples": self._samples["episode"].detach().cpu().tolist(),
            "current_episode_invalid_samples": self._invalid["episode"].detach().cpu().tolist(),
            "current_episode_work": {window: {key: tensor.detach().cpu().tolist() for key, tensor in fields.items()}
                for window, fields in self._work["episode"].items()},
            "work_moment_order": list(MOMENTS)}

    def checkpoint_state(self):
        if self._open:
            raise ValueError("Cannot checkpoint an unfinished physical policy interval")
        return {"schema": "jump-actuation-worker-measurement-checkpoint-v1",
            "exact_resume_supported": False, "summary": self.summary(),
            "last_pre_reset_transition": deepcopy(self._last_transition)}
