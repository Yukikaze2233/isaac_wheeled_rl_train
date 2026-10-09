#!/usr/bin/env python3
"""Record an unchanged native recovery prefix in one isolated CPU replay.

The frozen replay driver still authenticates both models, its fixed projection,
the native constructor, and completion. Instance wrappers copy existing values;
each original method runs once and returns its original result. No training
runtime, controller, physical clock, validator, or reward is replaced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ADAPTER_COMMIT = "bcc65de507ed0ec877bb17c9401aeff13b52da2a"
DRIVER_SHA = "3f8a61f76a4ca680234407545818e55173cac3b2c742c4655ed4b94999065fa0"
PREP_SHA = "9a9fed4f095b97c740f6edaef63af08422c5e98b2d2a6d19c74845b15290a857"
PROJECTION_SHA = "03bf51b6b9f45fc2aaa450af7600d14e68ff767dd192b652775b4be5e8840e9d"
LANES = (12, 13, 16, 17, 19)
CONDITIONS = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_process_identity(pid):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [value.decode() for value in (directory / "cmdline").read_bytes().split(b"\0") if value]
    return dict(pid=pid, starttime=int(stat[19]), command=argv)


def wait_process_identity(process, command, timeout=5.):
    """A new Linux process can briefly have an empty cmdline during exec."""
    deadline = time.monotonic() + timeout
    attempts = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError("Owned child exited before its process identity was observed")
        attempts += 1
        observed = read_process_identity(process.pid)
        if observed["command"] == command:
            observed["identity_read_attempts"] = attempts
            return observed
        time.sleep(.01)
    raise ValueError("Owned child process identity did not match within the bounded startup window")


class SelectedBuffer:
    """Bounded same-device snapshots; export only after the native replay."""

    def __init__(self, lanes=LANES, capacity=1600, *, device="cpu"):
        import torch
        if capacity != 1600 or tuple(lanes) != LANES or device != "cpu":
            raise ValueError("Only the original five-lane, eight-second CPU prefix is allowed")
        self.indices = torch.tensor(lanes, dtype=torch.long, device=device)
        self.capacity = capacity
        self.arrays = {}
        self.counts = {}

    def capture(self, name, value, row):
        import torch
        if not 0 <= row < self.capacity or value.shape[0] != 28 or value.device.type != "cpu":
            raise ValueError("Diagnostic shape, device or capacity differs")
        selected = value.detach().index_select(0, self.indices)
        if name not in self.arrays:
            self.arrays[name] = torch.empty((self.capacity, *selected.shape), dtype=selected.dtype,
                                           device=selected.device)
            self.counts[name] = 0
        array = self.arrays[name]
        if (self.counts[name] != row or array.shape[1:] != selected.shape
                or array.dtype != selected.dtype):
            raise ValueError("Diagnostic field was duplicated, skipped, or changed shape")
        array[row].copy_(selected)
        self.counts[name] += 1

    def export(self, directory, expected_rows):
        import numpy as np
        if expected_rows != self.capacity or set(self.counts.values()) != {expected_rows}:
            raise ValueError("Diagnostic did not observe every original prefix interval")
        arrays = {name: value[:expected_rows].numpy() for name, value in self.arrays.items()}
        nonfinite = {}
        for name, value in arrays.items():
            if np.issubdtype(value.dtype, np.floating):
                count = int((~np.isfinite(value)).sum())
                if count:
                    nonfinite[name] = count
                    # +inf means the probe has not established evidence yet.
                    if name != "probe_age_s" or not np.all(np.isfinite(value) | np.isposinf(value)):
                        raise ValueError("Unexpected nonfinite diagnostic field: " + name)
        path = Path(directory) / "script_prefix_200hz.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return dict(file=str(path), sha256=digest(path), rows=expected_rows,
                    lanes=list(LANES), fields={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                                               for name, value in arrays.items()},
                    nonfinite_counts=nonfinite,
                    uncompressed_bytes=sum(value.nbytes for value in arrays.values()))


@contextmanager
def observe_method(instance, name, before=None, after=None):
    """Restore the exact instance dictionary, including inherited methods."""
    original = getattr(instance, name)
    inherited = name not in instance.__dict__
    saved = instance.__dict__.get(name)

    def wrapper(*args, **kwargs):
        if before is not None:
            before(*args, **kwargs)
        result = original(*args, **kwargs)
        if after is not None:
            after(result, *args, **kwargs)
        return result

    setattr(instance, name, wrapper)
    try:
        yield
    finally:
        if inherited:
            delattr(instance, name)
        else:
            setattr(instance, name, saved)


class ScriptRecorder:
    def __init__(self, env):
        import torch
        from contextlib import ExitStack
        if (env.num_envs != 28 or env.device != "cpu" or env.design_preflight.get("role") != "baseline"
                or env.dt != .005 or env.policy_dt != .02 or env.recovery is None
                or env.recovery.reference_stride != 1 or env.torque_monitor is None
                or env.cfg.get("auto_reset") is not False):
            raise ValueError("Diagnostic installation requires the authenticated original fixed CPU replay")
        if [env.scene_groups[i] for i in LANES] != ["recovery_left_side_90_stand"] * 2 + [
                "recovery_right_side_90_stand"] * 3:
            raise ValueError("Original lane order differs")
        self.env = env
        self.buffer = SelectedBuffer()
        self.stack = ExitStack()
        self.ordinal = -1
        self.pending = False
        self.stages = {}
        self.alive = torch.ones(28, dtype=torch.bool, device="cpu")

    def capture(self, name, value):
        if self.pending:
            self.buffer.capture(name, value, self.ordinal)

    def enter(self, stage):
        if self.pending:
            previous = self.stages.get(stage, 0)
            if previous != self.ordinal:
                raise ValueError("Original method order or number of calls differs: " + stage)
            self.stages[stage] = previous + 1

    def reference_before(self, q, dq, gravity, gyro):
        recovery = self.env.recovery
        self.ordinal += 1
        self.pending = self.ordinal < self.buffer.capacity
        self.enter("reference")
        self.capture("feedback_q_rad", q)
        self.capture("feedback_dq_rad_s", dq)
        self.capture("feedback_gravity_body", gravity)
        self.capture("feedback_gyro_body_rad_s", gyro)
        self.capture("outer_elapsed_before_s", recovery.elapsed)
        self.capture("phase_before", recovery.fsm.phase)
        self.capture("target_before", recovery.fsm.script_targets)
        self.capture("first_episode_alive", self.alive)

    def reference_after(self, result, *args):
        recovery, fsm = self.env.recovery, self.env.recovery.fsm
        for name, value in (
            ("encoder_continuous_q_rad", recovery.encoders.q),
            ("fsm_continuous_q_rad", fsm.continuous_q), ("target_after", fsm.script_targets),
            ("phase_after", fsm.phase), ("route", fsm.route), ("age_ticks", fsm.age_ticks),
            ("phase_ticks", fsm.phase_ticks), ("ready", fsm.ready), ("ready_ticks", fsm.ready_ticks),
            ("stable", fsm.stable_pass), ("blend", fsm.blend),
            ("release_finished", recovery.release_finished), ("observed_height_m", recovery.observed_height),
            ("observed_support", recovery.observed_support),
            ("wheel_balance_active", recovery.wheel_balance_active),
            ("probe_torque_nm", recovery.wheel_probe_torque), ("probe_evidence", recovery.probe.evidence),
            ("probe_age_s", recovery.probe.age), ("probe_bad_samples", recovery.probe.bad_samples),
        ):
            self.capture(name, value)
        for key in CONDITIONS:
            self.capture("condition_" + key, fsm.stable_pass_masks[key])

    def torque_after(self, result, q, dq, rl_torque):
        self.enter("torque")
        self.capture("requested_after_ownership_nm", self.env.v5.requested_motor_effort)
        self.capture("bounds_at_script_nm", self.env.v5.current_motor_bounds)
        self.capture("script_blended_output_nm", result)
        self.capture("outer_elapsed_after_s", self.env.recovery.elapsed)
        self.capture("learning_mask_after_torque", self.env.recovery.learning_mask)

    def channel_before(self, channel, torque, *args, **kwargs):
        self.enter(channel)
        self.capture(channel + "_input_nm", torque)

    def channel_after(self, channel, result, *args, **kwargs):
        self.capture(channel + "_output_nm", result)

    def drive_before(self, *args, **kwargs):
        self.enter("issued")
        target = kwargs["target"] if "target" in kwargs else args[0]
        self.capture("issued_motor_effort_nm", target[:, self.env.ids])

    def monitor_before(self, applied, velocity, gas_force, gas_velocity, compression, requested, bounds):
        self.enter("post_physics")
        self.capture("post_physics_applied_motor_nm", applied)
        self.capture("post_physics_motor_dq_rad_s", velocity)
        self.capture("post_physics_gas_force_n", gas_force)
        self.capture("post_physics_gas_velocity_m_s", gas_velocity)
        self.capture("post_physics_compression_m", compression)
        self.capture("post_physics_q_rad", self.env.robot.data.joint_pos.torch[:, self.env.ids])

    def install(self):
        recovery = self.env.recovery
        try:
            self.stack.enter_context(observe_method(recovery, "advance_reference", self.reference_before,
                                                    self.reference_after))
            self.stack.enter_context(observe_method(recovery, "torque", after=self.torque_after))
            for name in ("command_transport", "usb_transport", "actuator_response"):
                channel = getattr(self.env, name, None)
                if channel is not None:
                    self.stack.enter_context(observe_method(channel, "apply_torque",
                        before=lambda *a, _name=name, **kw: self.channel_before(_name, *a, **kw),
                        after=lambda result, *a, _name=name, **kw: self.channel_after(_name, result, *a, **kw)))
            self.stack.enter_context(observe_method(self.env.robot, "set_joint_effort_target_index",
                                                    before=self.drive_before))
            self.stack.enter_context(observe_method(self.env.torque_monitor, "observe", before=self.monitor_before))
        except BaseException:
            self.stack.close()
            raise

    def close(self):
        self.stack.close()


def load_driver(adapter):
    adapter = Path(adapter).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(adapter), "rev-parse", "HEAD"], text=True).strip()
    if commit != ADAPTER_COMMIT:
        raise ValueError("Replay adapter is not the frozen source")
    driver = adapter / "scripts/run_rec_speed_mean_replay.py"
    if digest(driver) != DRIVER_SHA or digest(driver.with_name("prepare_rec_speed_mean_replay.py")) != PREP_SHA:
        raise ValueError("Frozen replay adapter bytes differ")
    sys.path.insert(0, str(driver.parent))
    spec = importlib.util.spec_from_file_location("original_recovery_diagnostic_driver", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or len(args.candidate) != 2 or args.output.exists():
        raise ValueError("Explicit execute, two authentic candidates and an exclusive output are required")
    driver = load_driver(args.adapter_root)
    owned_source = Path(__file__).resolve()
    repo = owned_source.parent.parent
    committed = subprocess.check_output(["git", "-C", str(repo), "show",
                                        "HEAD:scripts/replay_recovery_script_diagnostic.py"])
    if hashlib.sha256(committed).hexdigest() != digest(owned_source):
        raise ValueError("Diagnostic tool must be frozen before native execution")
    holder = {}
    original_readback, original_interval = driver.verify_physics, driver.observe_interval

    def readback(env, root, config):
        result = original_readback(env, root, config)
        if holder:
            raise ValueError("Diagnostic must use one fresh native constructor")
        recorder = ScriptRecorder(env)
        recorder.install()
        holder["recorder"] = recorder
        return result

    def interval(events, *a, **kw):
        result = original_interval(events, *a, **kw)
        holder["recorder"].alive.copy_(events.native.alive)
        return result

    driver.verify_physics, driver.observe_interval = readback, interval
    command = ["--runtime-root", str(args.runtime_root), "--execute", "--protocol", "nominal",
        "--device", "cpu", "--candidate-index", "0", "--prepared-projection", str(args.projection),
        "--projection-receipt-sha256", PROJECTION_SHA, "--output", str(args.output)]
    for pair in args.candidate:
        command += ["--candidate", *pair]
    try:
        result = driver.main(command)
        recorder = holder["recorder"]
        if result["status"] != "native_mean_replay_completed_unscored":
            raise ValueError("Original native completion did not pass")
        sidecar = recorder.buffer.export(args.output, 1600)
        if any(value != 1600 for value in recorder.stages.values()):
            raise ValueError("Diagnostic method counts differ")
        write(args.output / "script_diagnostic.json", dict(status="native_script_diagnostic_completed_unscored",
            schema="native-recovery-five-lane-200hz-observation-v1", native_result_sha256=digest(args.output / "native_replay.json"),
            diagnostic_source_sha256=digest(owned_source), adapter_commit=ADAPTER_COMMIT, driver_sha256=DRIVER_SHA,
            original_method_calls=recorder.stages, sidecar=sidecar, policy_hz=50, physics_hz=200,
            reference_feedback_phase="before_physics_integration", effort_phase="issued_before_physics_integration",
            applied_phase="after_physics_before_contact_refresh", target_units="legs:rad; wheels:rad/s",
            remaining_available=False, original_controller_changed=False, PPO_updates=0, training_signals=0,
            optimizer_constructed=False, qualification=None, hardware_deployment_ready=False))
    finally:
        driver.verify_physics, driver.observe_interval = original_readback, original_interval
        if "recorder" in holder:
            holder["recorder"].close()


if __name__ == "__main__":
    main()
