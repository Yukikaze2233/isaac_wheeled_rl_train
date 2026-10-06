#!/usr/bin/env python3
"""One fresh-process arm of the authenticated v18 training prefix, inference only.

--preflight-only uses CPU checkpoint loading and never imports an App or env.
Native execution is an explicit separate invocation. No fixed-case projection,
PPO, optimizer, reward edit, command edit, or extra controller update is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SOURCE_COMMIT = "712d701473f38ef6797725f8efdda9e6beb85e87"
CHECKPOINT_SHA = "7f91d355d1e033891d895661dd63365820eb8553728a878c226667fe374dbcca"
CONTRACT_SHA = "7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013"
SOURCE_MANIFEST_SHA = "efc4cf55f98ee1c4efcda9b6fd9193a9170762fb589f7611a5b01aab2bc7c268"
SOURCE_HASHES = {
    "contracts/v6_task_campaign_v18.json": "027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022",
    "contracts/v6_task_campaign_recovery_worker_v18.json": CONTRACT_SHA,
    "scripts/train_chassis.py": "4326490e93e56f2240afa50d7ed99b82a6c4fdc3362b821c74af186956bdb73b",
    "src/wheeled_tasks/chassis/env.py": "a0eeb8da7a406aa79f8103f2857fa97ee36585a2bfddc1246fa0599b50c48cf3",
    "src/wheeled_tasks/chassis/episode_metrics.py": "26438717a12b3ab37b7279cbcae09dda91c549438c7a6090e7d6876c7a046511",
    "src/wheeled_tasks/chassis/learning_audit.py": "0ac7a3502199c0912bebf6a1689cf3d388ffe03e23fcb3217628a9a70d324715",
    "src/wheeled_tasks/chassis/recovery_prefix_curriculum.py": "c11cfe901744d943e27ac940c48e668be628c95ce57d6ea3f1461bca055b8bd4",
    "src/wheeled_tasks/chassis/recovery_training.py": "e6d0452ef90be408929cbfd56bba682171ea6e17ce54244cc395e6061626fba4",
    "src/wheeled_tasks/chassis/recovery_curriculum.py": "6617b7667dc703447b10c0ccd701424f833df51cf828f54e67b63e875b8e2a78",
    "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py": "b3424cf23f9d7afec294ad00576d8fc2f8c9de4f2d3560df3bb852c46f9a4dff",
}
CLOCKS = dict(task_actor_update=2579, stage_actor_update=15947,
              global_actor_update=15947, actor_updates_consumed=29360,
              successful_updates_total=29500, learning_lineage_updates=16007,
              critic_warmup_remaining=0)
COUNT, POLICY_STEPS, SEED, BLOCK = 4096, 600, 290927, 0
PHYSICS_DT, POLICY_DT = .005, .02
LESSONS = ("prefix_stand", "prefix_release", "original_chain", "ordinary_base")
GATES = ("valid", "support", "body_clear", "tilt", "gyro", "height", "wheel_speed", "leg_speed")
ROW_FIELDS = (
    "recovery_enabled", "recovery_prefix_lesson_id", "recovery_actual_profile_id",
    "recovery_phase", "recovery_route", "recovery_failed", "recovery_failure_code",
    "recovery_pure_rl", "recovery_holding_rl", "recovery_motion_released",
    "recovery_stable_seconds", "recovery_stable_peak_seconds", "recovery_stable_break_count",
    "recovery_rl_stable_s", "recovery_followup_s", "recovery_true_success",
    "recovery_platform_retained", "recovery_handover_count", "recovery_ever_blended",
    "recovery_script_seconds", "recovery_prefix_release_s",
    "commands", "velocity", "omega", "height", "gravity", "planar_speed_world",
    "episode_ticks", "done", "terminated", "success",
)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_value(value):
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite receipt value")
    return value


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_inputs(args, identity):
    return (digest(args.checkpoint) == identity["checkpoint_sha256"]
            and digest(args.checkpoint.with_suffix(".contract.json")) == identity["checkpoint_contract_sha256"]
            and digest(args.checkpoint.with_suffix(".checkpoint.json")) == identity["checkpoint_seal_sha256"]
            and digest(args.source_manifest) == identity["source_manifest_sha256"]
            and digest(Path(__file__)) == identity["wrapper_sha256"])


def snapshot_sources(root, config, inventory):
    # Hash a superset of local implementation dependencies, plus every asset.
    paths = {root / name for name in inventory}
    bundle = root / config["asset_directory"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    paths |= {bundle / "manifest.json", *(bundle / name for name in manifest["files_sha256"])}
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def authenticate(args):
    """Exact same-contract full-state identity; no Git checkout assumption."""
    import torch
    root = args.source_root.resolve()
    source_manifest = args.source_manifest.resolve()
    require(digest(source_manifest) == SOURCE_MANIFEST_SHA, "Independent460-file source manifest SHA differs")
    pinned = json.loads(source_manifest.read_text())
    require(pinned.get("source_commit") == SOURCE_COMMIT and pinned.get("files_count") == 460
            and len(pinned.get("files", {})) == 460, "Independent source manifest catalog differs")
    require(all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in pinned["files"]),
            "Source manifest path escape")
    require({name: digest(root / name) for name in pinned["files"]} == pinned["files"],
            "Actual source root differs from authenticated712 archive bytes")
    require({name: digest(root / name) for name in SOURCE_HASHES} == SOURCE_HASHES,
            "Source019 pinned ten-file identity differs")
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.ordinary_posture_revision import canonical_worker
    contract = root / "contracts/v6_task_campaign_recovery_worker_v18.json"
    config, manifest = preflight(contract)
    require(config == canonical_worker("recovery", root), "Canonical v18 train worker differs")
    require(digest(contract) == CONTRACT_SHA, "Worker contract SHA differs")
    require(not config.get("evaluation_exact_cases") and config.get("record_diagnostics") is True,
            "Exact native training diagnostics required")
    require(config.get("auto_reset", True) is True and "recovery_takeover" not in config,
            "Unmodified v18 autoreset and global rewards required")
    require(tuple(config[k] for k in ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim"))
            == (.005, .02, 35, 81, 6), "Native clocks or ABI differ")
    require(config["recovery_training"]["prefix_curriculum"]["enabled"] is True,
            "Native prefix lessons absent")
    binding = config["design_preflight"]
    design_path = root / binding["file"]
    require(digest(design_path) == binding["sha256"], "Independent design SHA differs")
    design = json.loads(design_path.read_text())
    invocation = design["stages"][binding["stage"]]["roles"]["train"]["invocation"]
    rule = invocation["seed"]
    seed = rule if type(rule) is int else rule["root_seed"] + BLOCK
    require(type(rule) is int or rule.get("rule") == "root_plus_block_index", "Wrong seed rule")
    require(invocation["num_envs"] == COUNT and seed == SEED and binding.get("block_index", 0) == BLOCK,
            "Exact 4096/block0/seed290927 invocation differs")
    require(config["enabled_stages"] == ["mixed"], "Actual mixed train stage required")
    control = load_chassis_control(root / config["control_math_source"])
    receipt = design_preflight(config, control, root, stage="mixed", role="train", seed=seed,
                               num_envs=COUNT, block_index=BLOCK)
    require(receipt is not None, "Independent train design receipt absent")
    pt = args.checkpoint.resolve()
    seal_path, side_contract = pt.with_suffix(".checkpoint.json"), pt.with_suffix(".contract.json")
    require(digest(pt) == CHECKPOINT_SHA and digest(side_contract) == CONTRACT_SHA,
            "Exact sealed recovery_update29500 checkpoint/contract required")
    seal = json.loads(seal_path.read_text())
    require(seal.get("sealed") is True and seal.get("schema") == "chassis-learning-checkpoint-v1"
            and seal.get("checkpoint_sha256") == CHECKPOINT_SHA and seal.get("contract_sha256") == CONTRACT_SHA,
            "Per-file seal identity differs")
    saved = torch.load(pt, map_location="cpu", weights_only=True)
    require(set(saved) >= {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "infos", "iter"},
            "Complete learning state required")
    infos = saved["infos"]
    require(all(infos.get(k) == value for k, value in CLOCKS.items()), "Sealed checkpoint infos clocks differ")
    for key in ("task_actor_update", "stage_actor_update", "global_actor_update", "actor_updates_consumed",
                "learning_lineage_updates"):
        require(seal.get(key) == infos[key], "Seal/infos clock mismatch: " + key)
    require(seal["learning_updates"] == infos["successful_updates_total"] == 29500,
            "Consumed-update clock differs")
    require(infos.get("contract_sha256") == CONTRACT_SHA
            and infos.get("task_campaign") == config["task_campaign"] == seal["task_campaign"],
            "Checkpoint campaign/contract differs")
    require(all(infos.get(key) == config[key] for key in ("asset_manifest_sha256", "control_math_sha256")),
            "Checkpoint physics identity differs")
    expected_actor = {"distribution.std_param": (6,)}
    expected_critic = {}
    for shape, name, target in [((35, 256, 128, 64, 6), "actor", expected_actor),
                                ((81, 256, 128, 64, 1), "critic", expected_critic)]:
        for i, (input_dim, output_dim) in enumerate(zip(shape, shape[1:])):
            target[f"mlp.{2*i}.weight"] = (output_dim, input_dim)
            target[f"mlp.{2*i}.bias"] = (output_dim,)
        state = saved[name + "_state_dict"]
        require(set(state) == set(target), "Native " + name + " parameter catalog differs")
        require(all(isinstance(v, torch.Tensor) and tuple(v.shape) == target[k]
                    and v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for k, v in state.items()),
                "Native " + name + " shape/dtype/finite state differs")
    std = saved["actor_state_dict"]["distribution.std_param"]
    require(bool((std > 0).all()), "Saved Gaussian std must stay positive")
    optimizer = saved["optimizer_state_dict"]
    require(len(optimizer["param_groups"]) == 1 and len(optimizer["state"]) == 17,
            "Full Adam population differs")
    group = optimizer["param_groups"][0]
    parameters = [*saved["actor_state_dict"].values(), *saved["critic_state_dict"].values()]
    require(group["params"] == list(range(17)) and group["lr"] == config["learning_rate"],
            "Native Adam parameter order/rate differs")
    for index, parameter in enumerate(parameters):
        state = optimizer["state"][index]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Full Adam moment fields differ")
        require(all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in state.values()),
                "Nonfinite Adam state")
        require(state["step"].numel() == 1 and float(state["step"]) > 0
                and state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape,
                "Adam step or moment shape differs")
    require(isinstance(infos.get("rng_state"), torch.Tensor)
            and isinstance(infos.get("env_rng_state"), torch.Tensor) and infos.get("cuda_rng_states"),
            "Native saved resume RNG states required")
    source_snapshot = snapshot_sources(root, config, pinned["files"])
    identity = dict(source_commit_label=SOURCE_COMMIT, source_identity_scope="pinned_bytes_no_git_assumption",
        source_root=str(root), pinned_source_sha256=SOURCE_HASHES, source_files_sha256=source_snapshot,
        source_manifest=str(source_manifest), source_manifest_sha256=SOURCE_MANIFEST_SHA,
        archive_source_files=pinned["files"], archive_source_file_count=460,
        wrapper_sha256=digest(Path(__file__)), contract_sha256=CONTRACT_SHA,
        checkpoint=str(pt), checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_contract_sha256=digest(side_contract), checkpoint_seal_sha256=digest(seal_path),
        checkpoint_saved_iter_field=saved["iter"], checkpoint_clocks=CLOCKS,
        full_actor_critic_Adam_finite=True, actor_tensor_sha256={k: tensor_sha(v) for k, v in saved["actor_state_dict"].items()},
        saved_std=std.tolist(), stage="mixed", seed=seed, sampling_seed=seed, block_index=BLOCK,
        design_preflight=receipt, design_sha256=binding["sha256"], torch_version=torch.__version__,
        rsl_rl_version=version("rsl-rl-lib"), ordinary_eligible_expected=1449,
        native_scene_run=False, runtime_preflight_passed=False)
    return config, manifest, control, saved, identity


class FirstBirthLedger:
    """Pure sparse ledger: a first done closes a lane permanently."""
    def __init__(self, births):
        self.births = births
        self.terminals = {}

    def close(self, lane, tick, row):
        require(type(lane) is int and 0 <= lane < len(self.births), "Invalid birth lane")
        require(lane not in self.terminals and row["done"], "First birth reopened or lacks native done")
        reasons = row["termination_reasons"]
        partition = ("failure" if row["terminated"] else "success" if row["success"] else
                     "boundary_truncation" if "boundary" in reasons or "blocked" in reasons else
                     "timeout" if row["timeout"] else None)
        require(partition is not None, "Native done outcome lacks a partition")
        self.terminals[lane] = dict(row, partition=partition, terminal_policy_tick=tick)

    def finish(self, tick, remaining):
        require(set(remaining) == set(range(len(self.births))) - set(self.terminals),
                "Final first-birth censor population differs")
        rows = []
        for lane, birth in enumerate(self.births):
            ended = lane in self.terminals
            outcome = self.terminals[lane] if ended else dict(remaining[lane],
                partition="right_censored_unfinished", censor_policy_tick=tick)
            rows.append(dict(birth, **outcome))
        partitions = dict(Counter(row["partition"] for row in rows))
        require(sum(partitions.values()) == len(self.births), "First-birth partition count differs")
        by_lesson = {}
        for lesson in LESSONS:
            members = [row for row in rows if row["lesson"] == lesson]
            by_lesson[lesson] = dict(fixed_birth_denominator=len(members),
                partitions=dict(Counter(row["partition"] for row in members)),
                ever_handed_over=sum(row.get("first_handover_reference_tick", -1) >= 0 for row in members),
                ever_motion_released=sum(row.get("first_motion_release_reference_tick", -1) >= 0 for row in members))
        return dict(fixed_birth_denominator=len(rows), partitions=partitions, rows=rows,
                    by_lesson=by_lesson,
                    censor_is_not_failure=True, observations_are_not_qualification=True)


class NativeObserver:
    """Read original update results once, without changing a controller output."""
    METRICS = ("frames", "whole_interval_actor_valid_frames", "hold_at_interval_start_frames",
               "nonzero_vx_command_frames", "nonzero_yaw_command_frames", "vx_command_sum_m_s",
               "yaw_command_sum_rad_s", "vx_actual_sum_m_s", "yaw_actual_sum_rad_s",
               "planar_speed_sum_m_s", "vx_abs_error_sum_m_s", "yaw_abs_error_sum_rad_s",
               "actor_valid_and_nonzero_vx_command_frames", "actor_valid_and_nonzero_yaw_command_frames")

    def __init__(self, env, output):
        import torch
        self.t, self.env, self.recovery = torch, env, env.recovery
        r, n, device = self.recovery, env.num_envs, env.device
        self.live = torch.ones(n, dtype=torch.bool, device=device)
        self.tick = self.reference_calls = self.torque_calls = 0
        self.expected = torch.ones_like(self.live)
        self.hold_start = torch.zeros_like(self.live)
        self.handover_ref = torch.full((n,), -1, dtype=torch.long, device=device)
        self.release_ref = self.handover_ref.clone()
        self.last = {}
        self.previous_handover = torch.zeros(n, device=device)
        self.previous_release = torch.zeros_like(self.live)
        self.gates = torch.zeros(2, 4, 9, dtype=torch.long, device=device)
        self.policy_gates = torch.zeros_like(self.gates)
        self.metrics = torch.zeros(2, 4, len(self.METRICS), dtype=torch.float64, device=device)
        self.birth_counts = torch.zeros(4, dtype=torch.long, device=device)
        self.later_terminals = torch.zeros(4, 4, dtype=torch.long, device=device)
        self.sparse = gzip.open(output / "events.jsonl.gz", "wt", encoding="utf-8")
        level = r.pose_curriculum.level_ids.cpu().tolist()
        profile = r.pose_curriculum.profile_ids.cpu().tolist()
        enabled = r.enabled.cpu().tolist()
        births = [dict(lane=i, birth_id=1, group=env.scene_groups[i], terrain=env.kinds[i],
                       recovery_enabled=enabled[i], lesson=LESSONS[level[i] if enabled[i] else 3],
                       sampled_profile=(r.pose_curriculum.profile_names[profile[i]] if enabled[i] else None))
                  for i in range(n)]
        self.ledger = FirstBirthLedger(births)
        self.original_begin, self.original_torque, self.original_update = r.begin_policy, r.torque, r.fsm.update
        def begin():
            self.original_begin()
            self.expected.copy_(r.pure_rl)
            self.hold_start.copy_(r.policy_motion_hold)
            self.torque_calls = 0
        def torque(q, dq, effort):
            self.expected &= ~r.enabled | (r.release_finished & (r.fsm.pure_rl | r.fsm.failed))
            self.torque_calls += 1
            return self.original_torque(q, dq, effort)
        def update(*args, **kwargs):
            result = self.original_update(*args, **kwargs)
            self.reference_calls += 1
            self.accumulate_gates(self.gates, r.fsm.holding_rl, r.pose_curriculum.level_ids,
                                  r.enabled, r.fsm.stable_pass_masks)
            # During passive initial release the disabled FSM intentionally has
            # default RL/motion_released flags. Adapter release_finished is
            # essential: these placeholders are not a physical handover.
            entered = self.live & r.enabled & r.release_finished & r.fsm.pure_rl & ~r.fsm.failed & (self.handover_ref < 0)
            released = self.live & r.enabled & r.release_finished & r.fsm.motion_released & (self.release_ref < 0)
            self.handover_ref[entered] = self.reference_calls
            self.release_ref[released] = self.reference_calls
            return result
        r.begin_policy, r.torque, r.fsm.update = begin, torque, update

    def accumulate_gates(self, target, hold, levels, enabled, masks):
        t = self.t
        for lesson in range(3):
            members = hold & enabled & (levels == lesson)
            values = t.stack((members, *(members & masks[name] for name in GATES)), -1).long()
            target[0, lesson] += values.sum(0)
            target[1, lesson] += (values * self.live[:, None]).sum(0)

    def rows(self, ids, data, extras):
        available = {k: data[k][ids].detach().cpu().tolist() for k in ROW_FIELDS}
        available["timeout"] = extras["time_outs"][ids].cpu().tolist()
        available["whole_interval_actor_valid"] = extras["policy_learning_mask"][ids].cpu().tolist()
        refs = {"first_handover_reference_tick": self.handover_ref[ids].cpu().tolist(),
                "first_motion_release_reference_tick": self.release_ref[ids].cpu().tolist()}
        reasons = {k: v[ids].cpu().tolist() for k, v in data["reasons"].items()}
        result = {}
        for j, lane in enumerate(ids.cpu().tolist()):
            row = {k: v[j] for k, v in available.items()}
            row.update({k: v[j] for k, v in refs.items()})
            row.update(done=bool(row["done"]), terminated=bool(row["terminated"]), success=bool(row["success"]),
                       timeout=bool(row["timeout"]), termination_reasons=[k for k, v in reasons.items() if v[j]])
            result[lane] = row
        return result

    def observe(self, tick, extras):
        t, r = self.t, self.recovery
        data = extras["diagnostics"]
        require(all(k in data for k in ROW_FIELDS), "Missing native first-cohort diagnostic field")
        require(self.torque_calls == 4 and t.equal(self.expected, extras["policy_learning_mask"]),
                "Actual whole-policy recovery ownership differs from substeps")
        require(not bool((extras["policy_bootstrap_mask"] & data["terminated"]).any()),
                "Native physical terminal bootstraps")
        timing = self.env.startup_report["control_timing"]["runtime_substep_verification"]
        require(timing["pd_updates"] == timing["feedback_refreshes"] == 4,
                "Actual 200Hz control timing differs")
        require(all(bool(t.isfinite(data[k]).all()) for k in ("commands", "velocity", "omega", "height", "gravity")),
                "Nonfinite native policy diagnostic")
        self.tick = tick
        levels, enabled = data["recovery_prefix_lesson_id"], data["recovery_enabled"].bool()
        masks = {k: data[f"recovery_stable_{k}_pass"].bool() for k in GATES}
        self.accumulate_gates(self.policy_gates, data["recovery_holding_rl"].bool(), levels, enabled, masks)
        commands, velocity, omega = data["commands"], data["velocity"], data["omega"]
        values = t.stack((t.ones_like(data["height"]), extras["policy_learning_mask"], self.hold_start,
                         commands[:, 0].abs() > .001, commands[:, 1].abs() > .001,
                         commands[:, 0], commands[:, 1], velocity[:, 0], omega[:, 2], data["planar_speed_world"],
                         (commands[:, 0]-velocity[:, 0]).abs(), (commands[:, 1]-omega[:, 2]).abs(),
                         extras["policy_learning_mask"] & (commands[:, 0].abs() > .001),
                         extras["policy_learning_mask"] & (commands[:, 1].abs() > .001)), -1).double()
        for lesson in range(4):
            members = (enabled & (levels == lesson)) if lesson < 3 else ~enabled
            self.metrics[0, lesson] += values[members].sum(0)
            self.metrics[1, lesson] += values[members & self.live].sum(0)
        done = data["done"].bool()
        event_mask = self.live & enabled & ((data["recovery_handover_count"] > self.previous_handover)
                     | (data["recovery_motion_released"].bool() & ~self.previous_release) | done)
        event_ids = event_mask.nonzero().flatten()
        for lane, row in self.rows(event_ids, data, extras).items():
            self.sparse.write(json.dumps(dict(policy_tick=tick, observed_at_s=tick*POLICY_DT,
                lane=lane, group=self.env.scene_groups[lane], birth_id=1, native=row), allow_nan=False)+"\n")
        first_done = (self.live & done).nonzero().flatten()
        for lane, row in self.rows(first_done, data, extras).items():
            self.ledger.close(lane, tick, row)
        later = done & ~self.live
        for lesson in range(4):
            members = later & ((enabled & (levels == lesson)) if lesson < 3 else ~enabled)
            terminated, success = data["terminated"].bool(), data["success"].bool()
            boundary = data["reasons"]["boundary"] | data["reasons"].get("blocked", t.zeros_like(done))
            self.later_terminals[lesson] += t.stack(((members & terminated).sum(),
                (members & ~terminated & success).sum(), (members & ~terminated & ~success & boundary).sum(),
                (members & ~terminated & ~success & ~boundary & extras["time_outs"]).sum()))
        self.live &= ~done
        current_levels, current_enabled = r.pose_curriculum.level_ids, r.enabled
        for lesson in range(4):
            self.birth_counts[lesson] += (done & ((current_enabled & (current_levels == lesson))
                if lesson < 3 else ~current_enabled)).sum()
        self.previous_handover.copy_(data["recovery_handover_count"])
        self.previous_release.copy_(data["recovery_motion_released"])
        self.last = {k: data[k].clone() for k in ROW_FIELDS}
        self.last["reasons"] = {k: v.clone() for k, v in data["reasons"].items()}
        self.last_extras = {k: extras[k].clone() for k in ("time_outs", "policy_learning_mask")}

    def finish(self, output):
        ids = self.live.nonzero().flatten()
        first = self.ledger.finish(self.tick, self.rows(ids, self.last, self.last_extras))
        with gzip.open(output / "first_birth_rows.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in first.pop("rows"):
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        self.sparse.close()
        gates = {}
        for clock, values in (("reference_200Hz", self.gates), ("diagnostic_50Hz", self.policy_gates)):
            gates[clock] = {scope: {name: dict(hold_frames=row[0],
                gates={g: dict(passed=row[i+1], denominator=row[0], fraction=row[i+1]/row[0] if row[0] else None)
                       for i,g in enumerate(GATES)}) for name,row in zip(LESSONS, array)}
                for scope,array in zip(("all_native_births", "first_birth_only"), values.cpu().tolist())}
        metrics = {scope: {name: dict(zip(self.METRICS, row)) for name,row in zip(LESSONS, array)}
                   for scope,array in zip(("all_native_births", "first_birth_only"), self.metrics.cpu().tolist())}
        return dict(first_birth=first, raw_instantaneous_hold_gates=gates, policy_diagnostics=metrics,
            reference_update_calls=self.reference_calls,
            subsequent_births_by_lesson=dict(zip(LESSONS, self.birth_counts.cpu().tolist())),
            subsequent_terminal_partition_order=["failure", "success", "boundary_truncation", "timeout"],
            subsequent_terminal_counts=dict(zip(LESSONS, self.later_terminals.cpu().tolist())),
            denominator_scope="raw lane-frame counts, not TensorBoard rollout means",
            command_sample_clock="pre-autoreset 50Hz policy diagnostics; x4 is only physics-equivalent exposure",
            reference_event_time="reference tick k observes controller state before physics step k; time=(k-1)*.005s",
            gate_scope="native holding_rl mask from original reference update; historical peaks are not final stability",
            no_wide_state_trace=True)


def run_native(args, config, manifest, control, saved, identity, report):
    import torch
    os.environ.update(ENABLE_CAMERAS="0", LIVESTREAM="0")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(dict(headless=True, enable_cameras=False, device="cuda:0",
        kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
    completed = False
    observer = None
    try:
        from wheeled_tasks.chassis.env import ChassisEnv
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        from locomotion_exploration_sampling import load_native_actor, NativeActionSampler
        torch.manual_seed(SEED)
        env = ChassisEnv(config, manifest, control, args.source_root.resolve(), stage_name="mixed", num_envs=COUNT,
                         device="cuda:0", level=1., seed=SEED, run_role="train", block_index=BLOCK)
        require(env.startup_report.get("gpu_dynamics_enabled") is True
                and env.startup_report.get("runtime_preflight", {}).get("status") == "runtime_configuration_passed",
                "Native GPU/runtime independent preflight did not pass")
        require(env.recovery.has_tasks and env.recovery.pose_curriculum is not None
                and env.recovery.reference_stride == 1 and env.decimation == 4,
                "Actual native200Hz prefix differs")
        require(env.ordinary_posture.enabled and int(env.ordinary_posture.eligible.sum()) == 1449,
                "Actual native ordinary coverage differs from1449")
        cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
        cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        agent_config = cfg.to_dict()
        agent_config["actor"]["distribution_cfg"]["init_std"] = config["initial_noise_std"]
        actor = load_native_actor(saved, agent_config, config, "cuda:0")
        actor_before = {k: v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        # Reuse the training entry's resume protocol. Neither private sampling
        # stream nor observing hooks advance these restored global/env streams.
        torch.set_rng_state(saved["infos"]["rng_state"])
        env.generator.set_state(saved["infos"]["env_rng_state"])
        torch.cuda.set_rng_state_all(saved["infos"]["cuda_rng_states"])
        env.training_transitions = saved["infos"]["training_transitions"]
        env.confirmed_jump_bins_m = saved["infos"]["confirmed_jump_bins_m"]
        for key in ("task_actor_update", "stage_actor_update", "global_actor_update"):
            setattr(env, key, CLOCKS[key])
        env.reset(torch.arange(COUNT, device=env.device))
        birth_state = {
            "root_link_pose_w": env.robot.data.root_link_pose_w.torch,
            "root_link_lin_vel_w": env.robot.data.root_link_lin_vel_w.torch,
            "root_com_ang_vel_b": env.robot.data.root_com_ang_vel_b.torch,
            "full_joint_pos": env.robot.data.joint_pos.torch,
            "full_joint_vel": env.robot.data.joint_vel.torch,
            "origins": env.origins,
            "sampled_lesson_ids": env.recovery.pose_curriculum.level_ids,
            "sampled_profile_ids": env.recovery.pose_curriculum.profile_ids,
        }
        birth_hashes = {name: dict(sha256=tensor_sha(value), shape=list(value.shape), dtype=str(value.dtype))
                        for name,value in birth_state.items()}
        birth_hashes["scene_group_and_terrain_order"] = hashlib.sha256(json.dumps(
            list(zip(env.scene_groups, env.kinds)), separators=(",", ":")).encode()).hexdigest()
        initial_birth_hash = hashlib.sha256(json.dumps(birth_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report.update(native_startup=env.startup_report, runtime_preflight_passed=True,
            source_before_native_unchanged=snapshot_sources(args.source_root.resolve(), config, identity["archive_source_files"]) == identity["source_files_sha256"],
            native_group_population=dict(Counter(env.scene_groups)),
            initial_physical_birth_tensor_sha256=birth_hashes,
            initial_physical_birth_sha256=initial_birth_hash,
            initial_env_rng_sha256=tensor_sha(env.generator.get_state()),
            first_birth_protocol="constructor then native saved-RNG/clock resume then one native reset, as train_chassis",
            actor_factory="load_native_actor/MLPModel, no PPO runner or optimizer instantiated")
        require(report["source_before_native_unchanged"], "Sources changed before inference")
        require(len(report["native_group_population"]) == 95 and sum(report["native_group_population"].values()) == COUNT,
                "Native mixed95 catalog differs")
        observer = NativeObserver(env, args.output)
        sampler = NativeActionSampler(actor, SEED, "cuda:0")
        require(verify_inputs(args, identity), "Immutable probe inputs changed before inference")
        report["sampling_rng_initial_sha256"] = tensor_sha(sampler.generator.get_state())
        observations = env.get_observations()
        first_sampling = last_sampling = None
        action_square_sum = torch.zeros(3, 6, dtype=torch.float64, device=env.device)
        effective_std = None
        effective_std_first = None
        started = time.monotonic()
        with torch.inference_mode():
            for tick in range(1, POLICY_STEPS+1):
                require(launcher.app.is_running(), "App exited before600policy intervals")
                draw = sampler.draw(observations)
                action = draw["mu"] if args.arm == "mean" else draw["sample"]
                action_square_sum[0] += draw["mu"].double().square().sum(0)
                action_square_sum[1] += draw["potential_sample_minus_mean"].double().square().sum(0)
                action_square_sum[2] += (action-draw["mu"]).double().square().sum(0)
                effective_std = draw["effective_std"][0].clone()
                if effective_std_first is None:
                    require(torch.equal(draw["effective_std"], saved["actor_state_dict"]["distribution.std_param"].to(env.device).expand(COUNT,6)),
                            "Native effective six-axis std differs from saved Gaussian")
                    effective_std_first = effective_std.clone()
                observations, reward, done, extras = env.step(action)
                require(action.shape == (COUNT,6) and observations["policy"].shape == (COUNT,35)
                        and observations["critic"].shape == (COUNT,81), "Native actor/observation ABI differs")
                require(all(bool(torch.isfinite(v).all()) for v in (*observations.values(), reward)),
                        "Nonfinite native observation/reward")
                observer.observe(tick, extras)
                require(all(getattr(env,k) == CLOCKS[k] for k in ("task_actor_update", "stage_actor_update", "global_actor_update")),
                        "Inference advanced curriculum clock inside rollout")
                sampling = {k: draw[k] for k in ("rng_before_sha256", "rng_after_sha256", "draw_counter", "sample_element_count")}
                first_sampling = first_sampling or sampling
                last_sampling = sampling
        require(observer.reference_calls == 2400 and sampler.draw_counter == 600
                and sampler.sample_element_count == COUNT*6*600, "Full600policy/2400physics sampling coverage absent")
        require(all(getattr(env,k) == CLOCKS[k] for k in ("task_actor_update", "stage_actor_update", "global_actor_update")),
                "Inference advanced curriculum actor clocks")
        require(env.training_transitions == saved["infos"]["training_transitions"], "Inference advanced learning transition clock")
        require(all(torch.equal(v.detach().cpu(), actor_before[k]) for k,v in actor.state_dict().items()),
                "Inference altered actor/std tensors")
        report.update(observer.finish(args.output), policy_steps_completed=600, physical_steps_completed=2400,
            simulated_seconds=12., observer_included_wall_seconds=time.monotonic()-started,
            first_native_sampling=first_sampling, final_native_sampling=last_sampling,
            sampling_rng_final_sha256=tensor_sha(sampler.generator.get_state()),
            actor_tensors_and_saved_std_unchanged=True, curriculum_clocks_unchanged=True,
            learning_transition_clock_unchanged=True, final_env_rng_sha256=tensor_sha(env.generator.get_state()))
        require(torch.equal(effective_std_first, effective_std), "Native effective std changed during inference")
        report.update(native_effective_std=effective_std, native_effective_std_matches_saved=True,
            raw_policy_action_rms_by_axis=(action_square_sum/(COUNT*POLICY_STEPS)).sqrt(),
            raw_policy_action_rms_row_order=["mean", "potential_native_sample_minus_mean", "applied_action_minus_mean"],
            action_evidence_scope="raw native policy axes, before original controller decode/clipping")
        require(verify_inputs(args, identity)
                and snapshot_sources(args.source_root.resolve(), config, identity["archive_source_files"]) == identity["source_files_sha256"],
                "Checkpoint/source changed during native inference")
        report.update(status="native600_completed_preclose", checkpoint_unchanged=True, source_files_unchanged=True,
                      seal_contract_source_manifest_wrapper_unchanged=True)
        report["artifact_sha256"] = {p.name: digest(p) for p in args.output.glob("*.jsonl.gz")}
        write_json(args.output / "report.json", report)
        completed = True
    finally:
        if observer is not None:
            observer.sparse.close()
        close_error = None
        try:
            launcher.app.close()
        except BaseException:
            close_error = traceback.format_exc()
        # Preserve the terminal-before-close report byte-for-byte.
        write_json(args.output / "process_completion.json", dict(native600_completed=completed,
            app_close_returned=close_error is None, app_close_error=close_error,
            report_sha256=(digest(args.output / "report.json") if (args.output / "report.json").exists() else None),
            exit0_is_not_behavior_success=True))
        require(close_error is None, "App.close failed; native report is preclose only")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("mean", "stochastic"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path,
                        default=Path(__file__).with_name("source712_native_probe_manifest_019.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="native-v18-prefix-mean-stochastic-019-v1", status="starting", arm=args.arm,
        started_at_utc=datetime.now(timezone.utc).isoformat(), qualification=None, acceptance="not_assessed",
        formal_updates_consumed=0, optimizer_steps=0, optimizer_instances=0, PPO_updates=0,
        policy_steps_requested=600, physics_steps_requested=2400, requested_seconds=12.,
        reward_controller_command_overrides=[], evaluation_projection=False, clock_advance=False)
    try:
        config, manifest, control, saved, identity = authenticate(args)
        write_json(args.output / "identity.json", identity)
        report["identity_sha256"] = digest(args.output / "identity.json")
        if args.preflight_only:
            require(verify_inputs(args, identity)
                    and snapshot_sources(args.source_root.resolve(), config, identity["archive_source_files"]) == identity["source_files_sha256"],
                    "Preflight source files changed")
            report.update(status="preflight_passed_scene_not_run", native_scene_run=False,
                runtime_preflight_passed=False, policy_steps_completed=0, physical_steps_completed=0)
            write_json(args.output / "report.json", report)
        else:
            write_json(args.output / "report.json", report)
            run_native(args, config, manifest, control, saved, identity, report)
        return 0
    except BaseException:
        # An already completed preclose receipt must not be overwritten when
        # shutdown fails; failures before completion get an explicit failure.
        if report.get("status") != "native600_completed_preclose":
            report.update(status="failed", error=traceback.format_exc())
            write_json(args.output / "report.json", report)
        else:
            write_json(args.output / "postclose_error.json", dict(error=traceback.format_exc()))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
