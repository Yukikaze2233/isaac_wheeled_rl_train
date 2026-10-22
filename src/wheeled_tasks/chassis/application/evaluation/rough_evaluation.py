"""Measured rough exposure for ledger-accepted source episodes; no success gate."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import torch

from wheeled_tasks.chassis.application.evaluation.flat_policy_evaluation import command_kind


def _bucket():
    return dict(frames=0, valid=0, invalid=0, path=0., progress=0., segments=0,
        cross_tile=0, cross_region=0, episodes=0, terminated=0, boundary=0, timeout=0,
        errors={name: [0., 0., 0.] for name in ("vx", "yaw", "height")},
        observed={name: 0. for name in ("vx", "yaw", "height", "root_world_z", "scan_world_z")},
        reasons=Counter(), owners=Counter(), scan_hits=Counter(), command_kinds={})


class RoughTerrainMeasurements:
    """One world grid, retaining command ownership separately from geometry.

    prepare() performs one batched CPU readback and has no state effects.
    accept() is called only after the existing episode ledger accepts the batch.
    """

    @classmethod
    def from_env(cls, env, policy_dt):
        terrain = env.shared_terrain
        summary = terrain.summary()
        grid = terrain.importer.terrain_origins
        columns = summary["column_names"]
        if (summary["type"] != "rough" or grid.ndim != 3 or grid.shape[-1] != 3
                or grid.shape[1] != len(columns) or not math.isfinite(policy_dt) or policy_dt <= 0.):
            raise ValueError("Rough measurements require the actual generated terrain grid")
        self = cls()
        self.num_envs, self.dt = env.num_envs, float(policy_dt)
        self.rows, self.cols = grid.shape[:2]
        self.tile_size = tuple(summary["tile_size_m"])
        center = grid[0, 0, :2].detach().cpu().tolist()
        self.start = tuple(center[i] - self.tile_size[i] / 2 for i in range(2))
        self.columns = tuple(columns)
        self.bounds = deepcopy(summary["mesh_bounds_world_m"])
        self.owner_names = tuple(env.flat_episode.commands.summary()["terrain_profiles"])
        self.source = {key: deepcopy(summary[key]) for key in (
            "source", "source_commit", "source_env_sha256", "mesh_sha256", "geometry_seed")}
        self.by_type = {name: _bucket() for name in (*dict.fromkeys(columns), "border", "outside_mesh", "invalid_position")}
        self.by_row = {str(row): _bucket() for row in range(self.rows)}
        self.by_row.update({name: _bucket() for name in ("border", "outside_mesh", "invalid_position")})
        self.by_type_row, self.previous = {}, {}
        # First-reset positions are valid anchors, but never connect later resets.
        initial = torch.cat((env.robot.data.root_link_pose_w.torch[:, :2].detach().double(),
            env.flat_episode.episode_id[:, None].double(), env.episode_length_buf[:, None].double()), -1).cpu().tolist()
        for lane, (x, y, episode, tick) in enumerate(initial):
            if math.isfinite(x) and math.isfinite(y):
                self.previous[lane] = (int(episode), int(tick), x, y, self._cell(x, y))
        return self

    def _cell(self, x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            return ("invalid_position", None, None)
        if any(value < self.bounds[0][axis] or value > self.bounds[1][axis]
               for axis, value in enumerate((x, y))):
            return ("outside_mesh", None, None)
        row = math.floor((x - self.start[0]) / self.tile_size[0])
        col = math.floor((y - self.start[1]) / self.tile_size[1])
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return ("border", None, None)
        return (self.columns[col], row, col)

    def prepare(self, diagnostics, active_lanes):
        lanes = tuple(active_lanes)
        if (len(set(lanes)) != len(lanes) or any(type(lane) is not int or not 0 <= lane < self.num_envs for lane in lanes)):
            raise ValueError("Rough measurement lanes must be unique active ledger indices")
        fields = {name: diagnostics[name] for name in (
            "episode_id", "episode_tick", "measurement_valid_mask", "root_link_position_world",
            "root_yaw_rad", "effective_command", "root_COM_velocity_body", "angular_velocity_body",
            "height", "root_height_world", "reward_ground_height_world", "rough_base_scan_valid_hit_count",
            "done", "terminated", "timeout", "boundary", "command_override_active")}
        fields["command_owner_key"] = diagnostics["source_flat_episode"]["terrain_key_ids"]
        fields.update({"reason:" + name: mask for name, mask in diagnostics["reasons"].items()})
        vectors = {"root_link_position_world": 3, "effective_command": 3,
                   "root_COM_velocity_body": 3, "angular_velocity_body": 3}
        blocks, slices, width, device = [], {}, 0, None
        for name, value in fields.items():
            size = vectors.get(name, 1)
            expected = (self.num_envs, size) if name in vectors else (self.num_envs,)
            if not isinstance(value, torch.Tensor) or value.shape != expected:
                raise ValueError("Rough diagnostic shape mismatch: " + name)
            if device is not None and value.device != device:
                raise ValueError("Rough diagnostics must share one device")
            device = value.device
            blocks.append(value.detach().reshape(self.num_envs, size).double())
            slices[name] = (width, width + size)
            width += size
        # A single device-to-host transfer for every lane and field, including masks.
        rows = torch.cat(blocks, -1).cpu().tolist()
        pending = []
        for lane in lanes:
            row = rows[lane]
            values = {name: row[a:b] if name in vectors else row[a] for name, (a, b) in slices.items()}
            for name in ("episode_id", "episode_tick", "command_owner_key", "rough_base_scan_valid_hit_count"):
                value = values[name]
                if not math.isfinite(value) or value != int(value):
                    raise ValueError("Rough integer diagnostic is invalid: " + name)
                values[name] = int(value)
            if not 0 <= values["rough_base_scan_valid_hit_count"] <= 9:
                raise ValueError("Rough source base scan has exactly nine rays")
            values["cell"] = self._cell(*values["root_link_position_world"][:2])
            values["kind"] = ("capture_override" if values["command_override_active"]
                              else command_kind(values["effective_command"]))
            if values["measurement_valid_mask"]:
                measured = [*values["root_link_position_world"], values["root_yaw_rad"],
                    *values["effective_command"], *values["root_COM_velocity_body"], *values["angular_velocity_body"],
                    values["height"], values["root_height_world"], values["reward_ground_height_world"]]
                if not all(math.isfinite(value) for value in measured):
                    raise ValueError("Valid rough measurements must be finite")
            pending.append((lane, values))
        return tuple(pending)

    def accept(self, pending):
        for lane, value in pending:
            cell = value["cell"]
            name, row, _ = cell
            row_name = str(row) if row is not None else name
            key = (name, row_name)
            if key not in self.by_type_row:
                self.by_type_row[key] = _bucket()
            bucket = self.by_type_row[key]
            buckets = (self.by_type[name], self.by_row[row_name], bucket)
            valid = bool(value["measurement_valid_mask"])
            episode, tick = value["episode_id"], value["episode_tick"]
            x, y = value["root_link_position_world"][:2]
            previous = self.previous.get(lane)
            continuous = valid and previous is not None and previous[:2] == (episode, tick - 1)
            path = progress = 0.
            cross_tile = cross_region = 0
            if continuous:
                dx, dy = x - previous[2], y - previous[3]
                path = math.hypot(dx, dy)
                vx = value["effective_command"][0]
                if vx != 0.:
                    yaw = value["root_yaw_rad"]
                    progress = math.copysign(1., vx) * (dx * math.cos(yaw) + dy * math.sin(yaw))
                old_cell = previous[4]
                cross_tile = int(old_cell[1] is not None and row is not None and old_cell[1:] != cell[1:])
                cross_region = int(old_cell != cell)
            owner_id = value["command_owner_key"]
            owner = self.owner_names[owner_id] if 0 <= owner_id < len(self.owner_names) else f"unresolved:{owner_id}"
            for out in buckets:
                out["frames"] += 1
                out["valid"] += valid
                out["invalid"] += not valid
                out["owners"][owner] += 1
                out["scan_hits"][value["rough_base_scan_valid_hit_count"]] += 1
                kind = out["command_kinds"].setdefault(value["kind"], dict(frames=0, valid_frames=0, path_m=0., signed_command_progress_m=0.))
                kind["frames"] += 1
                kind["valid_frames"] += valid
                if valid:
                    out["path"] += path
                    out["progress"] += progress
                    out["segments"] += continuous
                    out["cross_tile"] += cross_tile
                    out["cross_region"] += cross_region
                    kind["path_m"] += path
                    kind["signed_command_progress_m"] += progress
                    observed = {"vx": value["root_COM_velocity_body"][0],
                        "yaw": value["angular_velocity_body"][2], "height": value["height"],
                        "root_world_z": value["root_height_world"], "scan_world_z": value["reward_ground_height_world"]}
                    for metric, command in zip(("vx", "yaw", "height"), value["effective_command"]):
                        error = observed[metric] - command
                        sums = out["errors"][metric]
                        sums[0] += error
                        sums[1] += abs(error)
                        sums[2] += error * error
                    for metric, measured in observed.items():
                        out["observed"][metric] += measured
                if value["done"]:
                    out["episodes"] += 1
                    out["terminated"] += bool(value["terminated"])
                    out["boundary"] += bool(value["boundary"])
                    out["timeout"] += bool(value["timeout"])
                    for key, active in value.items():
                        if key.startswith("reason:") and active:
                            out["reasons"][key[7:]] += 1
            if valid and not value["done"]:
                self.previous[lane] = (episode, tick, x, y, cell)
            else:
                self.previous.pop(lane, None)

    def _report_bucket(self, bucket):
        n = bucket["valid"]
        return dict(accepted_frames=bucket["frames"], valid_frames=n, invalid_frames=bucket["invalid"],
            valid_lane_seconds=n * self.dt, invalid_lane_seconds=bucket["invalid"] * self.dt,
            continuous_position_intervals=bucket["segments"], root_link_planar_path_m=bucket["path"],
            signed_command_direction_progress_m=bucket["progress"], cross_tile_observations=bucket["cross_tile"],
            cross_region_observations=bucket["cross_region"], episodes_ending=bucket["episodes"],
            terminated_episodes=bucket["terminated"], boundary_episodes=bucket["boundary"], timeout_episodes=bucket["timeout"],
            termination_reasons=dict(bucket["reasons"]), command_owner_frames=dict(bucket["owners"]),
            scan_hit_count_frames={str(i): bucket["scan_hits"][i] for i in range(10)},
            by_command_kind=deepcopy(bucket["command_kinds"]),
            tracking={metric: dict(signed_error_mean=sums[0] / n if n else None,
                mae=sums[1] / n if n else None, rmse=math.sqrt(sums[2] / n) if n else None)
                for metric, sums in bucket["errors"].items()},
            observed_means={metric: total / n if n else None for metric, total in bucket["observed"].items()})

    def report(self):
        return {"schema": "rough-terrain-measurements-v1", "assessment": "measurement_only",
            "source": deepcopy(self.source), "grid_start_world_xy_m": self.start,
            "tile_size_m": self.tile_size, "rows": self.rows, "columns": self.cols,
            "column_terrain_types": self.columns, "mesh_bounds_world_m": deepcopy(self.bounds),
            "unobserved_type_rows": [dict(terrain_type=name, row=row)
                for name in dict.fromkeys(self.columns) for row in range(self.rows)
                if self.by_type_row.get((name, str(row)), {}).get("valid", 0) == 0],
            "scope": {"frames": "ledger-accepted pre-reset frames for lanes still within episode quota, including terminal frames",
                "terrain": "root world XY in half-open generated tile cells; mesh border and outside mesh are separate",
                "command_owner": "paid command terrain_key_ids retained separately; never used as physical exposure",
                "height": "root_link_world_z minus actual nine-ray base scan; error against effective_command[2]",
                "velocity": "root COM body vx and body omega_z; error against effective command",
                "tracking_units": {"vx": "m/s", "yaw": "rad/s", "height": "m"},
                "path": "world XY chords between consecutive valid frames of one episode; interval assigned to endpoint cell",
                "signed_progress": "XY chord projected on endpoint root yaw times sign(effective vx); zero when vx=0",
                "crossings": "observed cell changes across consecutive valid frames, not completed terrain traversal",
                "reset": "initial native reset anchors first interval; later reset jumps and gaps are excluded",
                "termination": "raw source/mechanical reasons at terminal location; attitude-or-contact is not a pure fall count",
                "scan": "hit-count histogram includes invalid frames; zero hits use the existing origin-z fallback",
                "capability": "exposure, motion and failures only; standing horizon survival is not a stair pass"},
            "by_terrain_type": {name: self._report_bucket(bucket) for name, bucket in self.by_type.items()},
            "by_row": {name: self._report_bucket(bucket) for name, bucket in self.by_row.items()},
            "by_terrain_and_row": {f"{name}/row:{row}": self._report_bucket(bucket)
                for (name, row), bucket in self.by_type_row.items()}}
