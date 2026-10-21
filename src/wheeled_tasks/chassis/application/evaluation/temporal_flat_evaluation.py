"""Inference-only drive/stop/reverse measurements using the production requests."""
from copy import deepcopy
import math


FIELD = "temporal_flat_requests"
PHASE_TICKS = (250, 250, 250, 249)
PHASE_NAMES = ("drive", "stop", "reverse", "final_stop")
SIGNS = (1., 0., -1., 0.)


def selected(protocol):
    return FIELD in protocol


def validate(protocol):
    """Keep the authorized command sequence separate from acceptance criteria."""
    if (protocol.get(FIELD) != {"phase_policy_ticks": list(PHASE_TICKS)}
            or protocol.get("mode") != "fixed_measurement"
            or protocol.get("actor_mode") != "mean" or "heading_hold" in protocol
            or protocol.get("schema") != "flat-policy-evaluation-v1"):
        raise ValueError("Temporal measurements require the reviewed mean-policy four-phase sequence")
    requests = [case["base_request"] for case in protocol["cases"]]
    if (len(requests) != 8 or {r[0] for r in requests} != {
            -.5, .5, -1.35, 1.35, -2.7, 2.7, -3., 3.}
            or any(r[1:] != [0., .30] for r in requests)):
        raise ValueError("Temporal measurements require all eight signed flat requests at 0.30m")


def phase_at_tick(tick):
    """The paid endpoint is one tick after the command's publishing boundary."""
    return sum(tick > end for end in (250, 500, 750))


def request_at_tick(request, tick):
    return [request[0] * SIGNS[phase_at_tick(tick)], request[1], request[2]]


def configuration(parent):
    from wheeled_tasks.chassis.application.evaluation.small_rough_evaluation import retained_flat_configuration
    return retained_flat_configuration(parent,
        [{"name": "temporal_flat_measurement", "terrain": ["flat"], "fraction": 1.}],
        scripted=False)


def install(owner, requests):
    import torch
    owner.set_temporal_training_requests(
        torch.arange(owner.num_envs, device=owner.device), requests,
        phase_policy_ticks=PHASE_TICKS, episode_policy_ticks=999)


class TemporalEpisodeMeasurements:
    """Observe pre-reset states; the bands describe time curves, never passes."""

    def __init__(self, request, position, yaw):
        self.request = list(request)
        self.previous_position = list(position)
        self.previous_yaw = yaw
        self.previous_velocity = None
        self.rows = []
        self.tick = 0

    def observe(self, frame):
        tick = frame["episode_tick"]
        if tick != self.tick + 1:
            raise ValueError("Temporal measurement cannot cross a reset or missing interval")
        self.tick = tick
        phase = phase_at_tick(tick)
        local_tick = tick - sum(PHASE_TICKS[:phase])
        if len(self.rows) <= phase:
            self.rows.append(dict(
                phase=PHASE_NAMES[phase], expected_frames=PHASE_TICKS[phase], frames=0,
                valid_frames=0, invalid_frames=0, command_vx_m_s=self.request[0] * SIGNS[phase],
                entry_root_position_world_m=deepcopy(self.previous_position),
                entry_yaw_rad=self.previous_yaw,
                entry_body_velocity_m_s=deepcopy(self.previous_velocity),
                path_xy_m=0., path_contiguous=self.previous_position is not None,
                endpoint_displacement_xy_m=None, endpoint_forward_displacement_m=None,
                endpoint_body_velocity_m_s=None, vx_absolute_error_sum=0.,
                final_second_vx_sum=0., final_second_frames=0, samples={},
                zero_speed_bands={str(v): dict(consecutive_frames=0, first_held=None, candidate=None)
                    for v in (.05, .1, .2)} if phase in (1, 3) else {},
                reverse_fraction_times_s={str(v): None for v in (.25, .5, .9)} if phase == 2 else {}))
        row = self.rows[phase]
        row["frames"] += 1
        if not frame["measurement_valid"]:
            row["invalid_frames"] += 1
            row["path_contiguous"] = False
            for band in row["zero_speed_bands"].values():
                band.update(consecutive_frames=0, candidate=None)
            self.previous_position = self.previous_yaw = self.previous_velocity = None
            return
        position, velocity = frame["root_link_position_world"], frame["root_com_velocity_body"]
        row["valid_frames"] += 1
        if self.previous_position is not None:
            row["path_xy_m"] += math.dist(position[:2], self.previous_position[:2])
        anchor, heading = row["entry_root_position_world_m"], row["entry_yaw_rad"]
        if anchor is not None:
            delta = [position[i] - anchor[i] for i in range(2)]
            row["endpoint_displacement_xy_m"] = delta
            row["endpoint_forward_displacement_m"] = delta[0] * math.cos(heading) + delta[1] * math.sin(heading)
        row["endpoint_body_velocity_m_s"] = list(velocity)
        row["vx_absolute_error_sum"] += abs(velocity[0] - row["command_vx_m_s"])
        if local_tick > PHASE_TICKS[phase] - 50:
            row["final_second_vx_sum"] += velocity[0]
            row["final_second_frames"] += 1
        witness = dict(time_s=local_tick * .02, body_velocity_m_s=list(velocity),
            path_xy_m=row["path_xy_m"], displacement_xy_m=deepcopy(row["endpoint_displacement_xy_m"]),
            forward_displacement_m=row["endpoint_forward_displacement_m"])
        if local_tick in (25, 50, 150):
            row["samples"][str(local_tick * .02)] = witness
        speed = math.hypot(velocity[0], velocity[1])
        for threshold, band in row["zero_speed_bands"].items():
            if speed <= float(threshold):
                if band["consecutive_frames"] == 0:
                    band["candidate"] = deepcopy(witness)
                band["consecutive_frames"] += 1
                if band["consecutive_frames"] == 26 and band["first_held"] is None:
                    band["first_held"] = dict(band["candidate"], confirmed_at_s=local_tick * .02)
            else:
                band.update(consecutive_frames=0, candidate=None)
        for fraction, reached in row["reverse_fraction_times_s"].items():
            target = row["command_vx_m_s"]
            if reached is None and math.copysign(1., target) * velocity[0] >= abs(target) * float(fraction):
                row["reverse_fraction_times_s"][fraction] = local_tick * .02
        self.previous_position, self.previous_yaw = list(position), frame["yaw_rad"]
        self.previous_velocity = list(velocity)

    def report(self, outcome):
        rows = deepcopy(self.rows)
        for row in rows:
            error_sum = row.pop("vx_absolute_error_sum")
            row["vx_mae_m_s"] = error_sum / row["valid_frames"] if row["valid_frames"] else None
            count = row["final_second_frames"]
            velocity_sum = row.pop("final_second_vx_sum")
            row["final_second_vx_mean_m_s"] = velocity_sum / count if count else None
            for band in row["zero_speed_bands"].values():
                band.pop("candidate")
                band["endpoint_consecutive_samples"] = band.pop("consecutive_frames")
                band["endpoint_continuous_observed_s"] = max(0, band["endpoint_consecutive_samples"] - 1) * .02
        return dict(phases=rows, completed_four_phase_horizon=outcome == "horizon_truncated" and self.tick == 999,
            qualification=False, scope=(
                "One continuous episode; early failures remain incomplete. Drive entry speed is measured, "
                "not assumed to equal the target. Distances use root-link XY sampled every20ms; velocities "
                "use body-frame root COM. The0.05/0.10/0.20m/s planar-speed bands and26 consecutive "
                "samples are descriptive measurement curves, not acceptance thresholds. First-held time "
                "is the first observed in-band endpoint, confirmed25 intervals later. No interpolation, "
                "invalid-gap path bridging or automatic skill qualification."))
