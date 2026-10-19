"""Bounded numerical event evidence; optional explicit worker fast-stop."""
import math
from numbers import Real
from pathlib import Path
import uuid

import torch


def classify_source_numerical(active_joint_velocity, root_com_velocity_body, angular_velocity_body):
    """The existing source termination's three tests, with separate evidence.

    Threshold equality is allowed, as in ScutFlatRewardRuntime. No position,
    contact or closed-chain-gap threshold is introduced here.
    """
    result = {}
    combined = torch.zeros(len(active_joint_velocity), dtype=torch.bool,
                           device=active_joint_velocity.device)
    for name, value, limit in (("active_joint_velocity_rad_s", active_joint_velocity, 500.),
            ("root_com_velocity_body_m_s", root_com_velocity_body, 100.),
            ("root_angular_velocity_body_rad_s", angular_velocity_body, 200.)):
        nonfinite = ~torch.isfinite(value).all(-1)
        over = value.abs().amax(-1) > limit
        result[name] = {"limit": limit, "nonfinite": nonfinite, "over_limit": over,
                        "axis_nonfinite": ~torch.isfinite(value), "axis_over_limit": value.abs() > limit}
        combined |= nonfinite | over
    return combined, result


class NumericalFailureCapture:
    """Up to 64 prior states; preserve first and latest reset-mode events.

    The reused ring retains 0.32 seconds at a 5ms physical step.
    Prior frames are captured before submitting each next physical step, so
    their state times precede the first abnormal post-step frame. Requested
    targets/effort belong to the explicitly labelled following interval; native
    applied effort and contact readbacks belong to the completed state interval.
    """
    HISTORY_FRAMES = 64
    CONTROL_FIELDS = ("leg_position_target", "wheel_velocity_target", "feedback_q", "feedback_dq",
        "sent_effort", "active_effort_after_filter_and_gate", "requested_motor_effort",
        "motor_effort_bounds", "raw_actor_action", "clipped_control_action",
        "previous_control_action", "commands", "spring_sent_effort",
        "policy_input35", "policy_mean_action", "policy_std_action",
        "pressure_generated_physical_tick", "pressure_strength", "pressure_body_com_positions_world",
        "pressure_mount_distance_m",
        "pressure_compression_m", "pressure_force_n", "pressure_mount_positions_world",
        "pressure_forces_world", "pressure_torques_about_com_world")

    def __init__(self, directory, count, metadata, *, stop_on_first_event=False,
            stop_on_nonfinite=False, closure_observation_threshold_m=.03, energy_readback=None,
            energy_contact_snapshot=None):
        if type(stop_on_first_event) is not bool:
            raise ValueError("Numerical first-event stop requires an explicit boolean")
        if type(stop_on_nonfinite) is not bool:
            raise ValueError("Numerical nonfinite stop requires an explicit boolean")
        if (isinstance(closure_observation_threshold_m, bool)
                or not isinstance(closure_observation_threshold_m, Real)
                or not math.isfinite(closure_observation_threshold_m)
                or closure_observation_threshold_m <= 0):
            raise ValueError("Closure observation threshold requires a finite positive distance in metres")
        self.closure_observation_threshold_m = float(closure_observation_threshold_m)
        self.path = Path(directory) / "numerical_failure_first.pt"
        if not self.path.parent.is_dir() or self.path.exists():
            raise ValueError("First numerical capture requires the existing fresh formal trace directory")
        self.count, self.metadata = count, dict(metadata)
        self.stop_on_first_event = stop_on_first_event
        self.stop_on_nonfinite = stop_on_nonfinite
        self.capture_latest = stop_on_nonfinite and not stop_on_first_event
        self.latest_path = Path(directory) / "numerical_failure_latest.pt"
        if self.latest_path.exists():
            raise ValueError("Latest numerical capture requires a fresh event file")
        self._active_numerical_lanes = None
        self.nonfinite_path = Path(directory) / "numerical_nonfinite_first.pt"
        if self.nonfinite_path.exists():
            raise ValueError("Nonfinite capture requires a fresh event file")
        self.nonfinite_recorded = False
        self.frames = []
        self.cursor = 0
        self.recorded = False
        self.failed_ids = None
        self.closure_first_path = Path(directory) / "closure_first_substep.pt"
        self.closure_peak_path = Path(directory) / "closure_peak_substep.pt"
        if self.closure_first_path.exists() or self.closure_peak_path.exists():
            raise ValueError("Closure capture requires fresh event files")
        # Observation-only: this threshold never resets a lane or stops the worker.
        self.closure_peak_m = self.closure_observation_threshold_m
        self.summary = {"enabled": True, "file": str(self.path), "history_frames": self.HISTORY_FRAMES,
            "recorded": False, "source_policy_end_observed": False,
            "termination_changed": stop_on_first_event or stop_on_nonfinite,
            "thresholds": [500., 100., 200.],
            "state_device": None, "diagnostic_only": not (stop_on_first_event or stop_on_nonfinite),
            "failure_policy": ("worker_abort_first_5ms_crossing" if stop_on_first_event else
                "source_episode_reset_finite_worker_abort_nonfinite" if stop_on_nonfinite else "record_only"),
            "stop_on_nonfinite": stop_on_nonfinite,
            "nonfinite": {"recorded": False, "file": str(self.nonfinite_path)},
            "latest": {"enabled": self.capture_latest, "file": str(self.latest_path),
                "recorded": False, "additional_events_observed": 0, "replacements": 0,
                "saved_physical_tick": None, "saved_policy_tick": None, "saved_lane_ids": [],
                "selection": "newly_abnormal_lanes_after_the_first_event",
                "termination_changed": False},
            "not_root_cause_fix": True, "worker_abort_raised": False,
            "all_joint_nonfinite_checked": False, "root_pose_nonfinite_checked": False,
            "closure_nonfinite_checked": False,
            "closure": {"threshold_m": self.closure_observation_threshold_m, "recorded": False,
                "termination_changed": False, "first_file": str(self.closure_first_path),
                "peak_file": str(self.closure_peak_path), "peak_replacements": 0,
                "first_dump_succeeded": False, "latest_dump_succeeded": False,
                "saved_peak_m": None, "saved_peak_physical_tick": None}}
        self.energy = energy_readback
        if energy_contact_snapshot is not None and (self.energy is None or not callable(energy_contact_snapshot)):
            raise ValueError("Energy contact evidence requires an energy reader and a callable snapshot")
        self.energy_contact_snapshot = energy_contact_snapshot
        self.energy_paths = {}
        if self.energy is not None:
            from .finite_energy_readback import PEAKS
            if self.energy.count != count:
                raise ValueError("Energy readback and capture must share native lane membership")
            # Healthy cold resumes may reuse the run directory. Preserve the
            # prior worker's evidence without imposing a new resume blocker.
            energy_worker = uuid.uuid4().hex
            self.energy_paths = {name: Path(directory) / f"energy_{energy_worker}_{name}_peak.pt" for name in PEAKS}
            if any(path.exists() for path in self.energy_paths.values()):
                raise ValueError("Energy peak capture requires fresh diagnostic files")
            self.summary["finite_energy"] = {"enabled": True, "worker_id": energy_worker,
                "readback": self.energy.summary(),
                "peak_captures": {name: {"file": str(path), "saved": False, "replacements": 0}
                                  for name, path in self.energy_paths.items()},
                "termination_changed": False, "ppo_mask_changed": False}

    def _save_energy_peak(self, index, frame, physical_tick, policy_tick, state_factory):
        from .finite_energy_readback import PEAKS
        name = PEAKS[index]
        path = self.energy_paths[name]
        ids = frame["peak_lane_ids"][index:index + 1]
        report = self.summary["finite_energy"]["peak_captures"][name]
        report.update(observed_value=float(frame["peak_values"][index]),
            observed_physical_tick=physical_tick, lane_ids=ids.cpu().tolist(), latest_dump_succeeded=False)
        temporary = path.with_suffix(".pt.tmp")
        try:
            payload = dict(schema="finite-energy-peak-event-v1", metadata=self.metadata,
                selection=name, lane_ids=ids.cpu().clone(),
                dump_physical_tick=physical_tick, dump_policy_tick=policy_tick,
                termination_changed=False, ppo_mask_changed=False,
                value=float(frame["peak_values"][index]),
                native_properties_selected={"mass_kg": self.energy.mass[ids].cpu().clone(),
                    "inertia_link_about_com_kg_m2": self.energy.inertia[ids].cpu().clone()},
                energy_metrics=self._lanes_cpu(frame["metrics"], ids),
                previous_post_physics_tick=frame["previous_tick"],
                previous_post_physics_state=(None if frame["previous"] is None
                    else self._lanes_cpu(frame["previous"], ids)),
                prior_frames=[{**{key:value for key,value in row.items() if key != "state"},
                    "state": self._lanes_cpu(row["state"], ids)}
                    for row in sorted(self.frames, key=lambda row: row["physical_tick"])],
                current={"physical_tick": physical_tick, "policy_tick": policy_tick,
                         "state": self._current_lanes_cpu(state_factory(), ids, physical_tick)},
                scope="largest observed finite metric; no new safety/sample threshold; work balance is partial")
            if self.energy_contact_snapshot is not None:
                try:
                    payload["native_contact"] = self.energy_contact_snapshot(environment_ids=ids)
                    payload["native_contact"]["physical_tick"] = physical_tick
                except Exception as error:
                    payload["native_contact"] = {"status": "error", "error": repr(error),
                        "physical_tick": physical_tick}
            torch.save(payload, temporary)
            temporary.replace(path)
            report.update(saved=True, saved_value=payload["value"], saved_physical_tick=physical_tick,
                          latest_dump_succeeded=True)
            report.pop("dump_error", None)
            report["replacements"] += 1
        except Exception as error:
            report["dump_error"] = repr(error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                report["cleanup_error"] = repr(error)

    def refresh_energy_summary(self):
        if self.energy is not None:
            self.summary["finite_energy"]["readback"] = self.energy.summary()

    def remember_before_physics(self, physical_tick, policy_tick, state):
        # Reuse the bounded ring for later closure maxima, even if a
        # record-only numerical event has already been saved.
        if len(self.frames) < self.HISTORY_FRAMES:
            frame = {"state": {key: value.detach().clone() for key, value in state.items()}}
            self.frames.append(frame)
        else:
            frame = self.frames[self.cursor]
            for key, value in state.items():
                frame["state"][key].copy_(value.detach())
        frame.update(physical_tick=physical_tick, policy_tick=policy_tick,
            next_physical_tick=physical_tick + 1, state_phase="before_next_physics")
        self.cursor = (self.cursor + 1) % self.HISTORY_FRAMES
        self.summary["state_device"] = str(next(iter(state.values())).device)

    def _lanes_cpu(self, state, ids):
        return {key: value.detach().index_select(0, ids).cpu().clone() for key, value in state.items()}

    def _current_lanes_cpu(self, state, ids, physical_tick):
        current = self._lanes_cpu(state, ids)
        # Transport.observe may mutate feedback views after physics. The
        # inputs that produced this interval's effort are in the existing ring.
        prior = next((frame for frame in self.frames
            if frame["next_physical_tick"] == physical_tick), None)
        if prior is not None:
            current.update(self._lanes_cpu({key: prior["state"][key] for key in self.CONTROL_FIELDS
                if key in prior["state"]}, ids))
        return current

    def observe_closure(self, physical_tick, policy_tick, gap, state_factory, *, crossing_confirmed=False):
        """Save first crossing and every new maximum at its actual 5ms state.

        Only the affected maximum lane(s) leave the device, and only two files
        are retained. Prior frames include episode ids so a reset between
        frames is visible rather than mistaken for physical motion.
        """
        if gap is None:
            return
        finite_gap = torch.where(torch.isfinite(gap), gap, -torch.inf)
        if not crossing_confirmed and not bool((finite_gap > self.closure_peak_m).any()):
            return
        peak = finite_gap.max()
        ids = (finite_gap == peak).nonzero(as_tuple=False).flatten()
        self.closure_peak_m = float(peak)
        report = self.summary["closure"]
        first = not report["recorded"]
        report.update(recorded=True, observed_peak_m=self.closure_peak_m,
            physical_tick=physical_tick, policy_tick=policy_tick,
            lane_ids=ids.cpu().tolist(), latest_dump_succeeded=False)
        if first:
            report.update(first_physical_tick=physical_tick, first_peak_m=self.closure_peak_m)
        temporary = self.closure_peak_path.with_suffix(".pt.tmp")
        try:
            payload = {"schema": "closed-chain-substep-event-v1", "metadata": self.metadata,
                "lane_ids": ids.cpu().clone(), "gap_m": gap[ids].detach().cpu().clone(),
                "dump_physical_tick": physical_tick, "dump_policy_tick": policy_tick,
                "threshold_m": self.closure_observation_threshold_m, "termination_changed": False,
                "state_phase": "after_physics_and_contact_refresh_before_reset",
                "selection": "maximum_gap_lanes_at_first_crossing_or_new_run_maximum",
                "prior_frames": [{**{key: value for key, value in frame.items() if key != "state"},
                    "state": self._lanes_cpu(frame["state"], ids)} for frame in
                    sorted(self.frames, key=lambda frame: frame["physical_tick"])],
                "current": {"physical_tick": physical_tick, "policy_tick": policy_tick,
                    "control_targets_apply_to_interval_ending_at": physical_tick,
                    "state": self._current_lanes_cpu(state_factory(), ids, physical_tick)}}
            # Atomic replacement preserves the prior complete peak if writing fails.
            torch.save(payload, temporary)
            if first:
                # Exclusive atomic publication; never leave a half-written first file.
                self.closure_first_path.hardlink_to(temporary)
                report["first_dump_succeeded"] = True
            temporary.replace(self.closure_peak_path)
            report["latest_dump_succeeded"] = True
            report.update(saved_peak_m=self.closure_peak_m, saved_peak_physical_tick=physical_tick)
            report["peak_replacements"] += 1
        except Exception as error:
            report["dump_error"] = repr(error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                report["temporary_cleanup_error"] = repr(error)

    def _save_event(self, path, report, physical_tick, policy_tick, ids,
            classification, state_factory, *, trigger, aborts_worker, replace=False):
        report.update(recorded=True, physical_tick=physical_tick, policy_tick=policy_tick,
            lane_ids=ids.cpu().tolist(), dump_succeeded=False)
        temporary = path.with_suffix(".pt.tmp")
        try:
            payload = {"schema": "source-numerical-first-event-v1", "metadata": self.metadata,
                "lane_ids": ids.cpu().clone(), "dump_physical_tick": physical_tick,
                "dump_policy_tick": policy_tick, "trigger": trigger,
                "original_policy_termination_evaluated_at_dump": False,
                "diagnostic_only_does_not_terminate": not aborts_worker,
                "failure_policy": self.summary["failure_policy"], "not_root_cause_fix": True,
                "classification": {name: {key: value.detach().index_select(0, ids).cpu().clone()
                    if isinstance(value, torch.Tensor) else value for key, value in row.items()}
                    for name, row in classification.items()},
                "prior_frames": [{**{key: value for key, value in frame.items() if key != "state"},
                    "state": self._lanes_cpu(frame["state"], ids)} for frame in
                    sorted(self.frames, key=lambda frame: frame["physical_tick"])],
                "current": {"physical_tick": physical_tick, "policy_tick": policy_tick,
                    "state_phase": "after_physics_and_contact_refresh_before_reset",
                    "control_targets_apply_to_interval_ending_at": physical_tick,
                    "state": self._current_lanes_cpu(state_factory(), ids, physical_tick)}}
            if replace:
                # Retain the last complete later event if storage fails.
                torch.save(payload, temporary)
                temporary.replace(path)
                report.update(saved_physical_tick=physical_tick, saved_policy_tick=policy_tick,
                    saved_lane_ids=report["lane_ids"])
                report["replacements"] += 1
            else:
                # Exclusive create preserves the first event even if an
                # external caller accidentally invokes us twice.
                with path.open("xb") as stream:
                    torch.save(payload, stream)
            report["dump_succeeded"] = True
            report.pop("dump_error", None)
        except Exception as error:
            report["dump_error"] = repr(error)
        finally:
            if replace:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    report["temporary_cleanup_error"] = repr(error)

    def observe_after_physics(self, physical_tick, policy_tick, signals, state_factory,
            *, closure_gap=None, joint_position=None, joint_velocity=None, root_pose=None):
        energy_frame = None
        if self.energy is not None:
            before = self.frames[(self.cursor - 1) % len(self.frames)] if self.frames else None
            control = (before["state"] if before is not None
                and before["next_physical_tick"] == physical_tick else None)
            energy_frame = self.energy.observe(physical_tick, state_factory(), control)
        check_signals = not self.recorded or self.stop_on_nonfinite
        mask, classification = classify_source_numerical(*signals) if check_signals else (None, {})
        nonfinite_mask = None if mask is None else torch.zeros_like(mask)
        for row in classification.values():
            nonfinite_mask |= row["nonfinite"]
        # Passive coordinates have different units; only finiteness is shared.
        for name, value in (("all_joint_position", joint_position),
                ("all_joint_velocity", joint_velocity), ("root_pose_world", root_pose)):
            if value is not None and mask is not None:
                nonfinite = ~torch.isfinite(value)
                lane_nonfinite = nonfinite.any(-1)
                mask |= lane_nonfinite
                nonfinite_mask |= lane_nonfinite
                classification[name] = {"nonfinite": lane_nonfinite, "axis_nonfinite": nonfinite}
                self.summary["root_pose_nonfinite_checked" if name == "root_pose_world"
                    else "all_joint_nonfinite_checked"] = True
        gap_finite = None if closure_gap is None else torch.isfinite(closure_gap)
        if gap_finite is not None and mask is not None:
            nonfinite = ~gap_finite
            mask |= nonfinite
            nonfinite_mask |= nonfinite
            classification["closure_gap_m"] = {"nonfinite": nonfinite, "value_m": closure_gap}
            self.summary["closure_nonfinite_checked"] = True
        gap_crossing = (None if closure_gap is None else
            (gap_finite & (closure_gap > self.closure_peak_m)).any())
        # One host decision covers existing guards and observation-only peaks.
        scalar_false = torch.zeros((), dtype=torch.bool, device=signals[0].device)
        checks = [scalar_false if mask is None else mask.any(),
                  scalar_false if nonfinite_mask is None else nonfinite_mask.any(),
                  scalar_false if gap_crossing is None else gap_crossing]
        if energy_frame is not None:
            checks.extend(energy_frame["new_peaks"].unbind())
        events = torch.stack(checks).cpu().tolist()
        numerical_event, nonfinite_event, closure_event = events[:3]
        if energy_frame is not None:
            for index, changed in enumerate(events[3:]):
                if changed:
                    self._save_energy_peak(index, energy_frame, physical_tick, policy_tick, state_factory)
        if numerical_event:
            self.refresh_energy_summary()
        if closure_event:
            self.observe_closure(physical_tick, policy_tick, closure_gap, state_factory,
                crossing_confirmed=True)
        if not numerical_event:
            self._active_numerical_lanes = None
            return
        abort_nonfinite = self.stop_on_nonfinite and nonfinite_event
        was_recorded = self.recorded
        if self.capture_latest and not abort_nonfinite:
            # Repeated bad substeps in one uninterrupted crossing need only
            # one snapshot. Another lane entering the mask is a new event.
            new_lanes = (mask if self._active_numerical_lanes is None else
                mask & ~self._active_numerical_lanes)
            self._active_numerical_lanes = mask.detach().clone()
            if was_recorded:
                ids = new_lanes.nonzero(as_tuple=False).flatten()
                if len(ids):
                    report = self.summary["latest"]
                    report["additional_events_observed"] += 1
                    self._save_event(self.latest_path, report, physical_tick, policy_tick,
                        ids, classification, state_factory,
                        trigger="later_physical_substep_crossing", aborts_worker=False, replace=True)
        if not was_recorded:
            # Never replace the first event, even if its dump fails.
            self.recorded = True
            ids = mask.nonzero(as_tuple=False).flatten()
            self.failed_ids = ids.detach().clone()
            self.summary["original_policy_numerical_termination_at_dump"] = "not_yet_evaluated"
            self._save_event(self.path, self.summary, physical_tick, policy_tick, ids,
                classification, state_factory, trigger="first_physical_substep_crossing",
                aborts_worker=self.stop_on_first_event or abort_nonfinite)
        if abort_nonfinite and not self.nonfinite_recorded:
            self.nonfinite_recorded = True
            report = self.summary["nonfinite"]
            if was_recorded:
                ids = nonfinite_mask.nonzero(as_tuple=False).flatten()
                self._save_event(self.nonfinite_path, report, physical_tick, policy_tick, ids,
                    classification, state_factory, trigger="first_nonfinite_physical_substep",
                    aborts_worker=True)
            else:
                # The first numerical file already contains this nonfinite event.
                report.update({name: self.summary[name] for name in
                    ("physical_tick", "policy_tick", "dump_succeeded")},
                    lane_ids=nonfinite_mask.nonzero(as_tuple=False).flatten().cpu().tolist(),
                    recorded=True, file=str(self.path))
                if "dump_error" in self.summary:
                    report["dump_error"] = self.summary["dump_error"]
        if (self.stop_on_first_event and not was_recorded) or abort_nonfinite:
            self.summary["worker_abort_raised"] = True
            reason = "nonfinite physical substep" if abort_nonfinite else "first physical substep"
            saved = self.summary["nonfinite"] if abort_nonfinite else self.summary
            raise FloatingPointError(f"Formal V7 native numerical safety at {reason} "
                f"{physical_tick}; snapshot_saved={saved['dump_succeeded']}; "
                "policy-end termination not evaluated; not a root-cause fix")

    def observe_original_policy_end(self, physical_tick, policy_tick, numerical_mask):
        self.refresh_energy_summary()
        if self._active_numerical_lanes is not None:
            # The existing source reset ends these crossings. Do not require
            # a healthy frame before observing a newly reset lane again.
            self._active_numerical_lanes &= ~numerical_mask
        if not self.recorded or self.summary["source_policy_end_observed"]:
            return
        self.summary.update(source_policy_end_observed=True,
            source_policy_end_physical_tick=physical_tick, source_policy_end_tick=policy_tick,
            recorded_lanes_original_numerical=numerical_mask[self.failed_ids].detach().cpu().tolist(),
            original_numerical_lane_ids=numerical_mask.nonzero(as_tuple=False).flatten().cpu().tolist())
