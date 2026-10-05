"""Synthetic CPU reader tests; none of these fixtures is a physics result."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("compare_physical_ab.py")
SPEC = importlib.util.spec_from_file_location("physical_ab_reader", PATH)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def mutate(path, function):
    value = json.loads(path.read_text())
    function(value)
    write(path, value)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root/"A_reports", self.root/"B_reports"
        self.sa, self.sb = self.root/"A_source", self.root/"B_source"
        self.sa.mkdir()
        def git(root, *args):
            return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout.decode().strip()
        git(self.sa, "init", "-q")
        git(self.sa, "config", "user.email", "fixture@example.invalid")
        git(self.sa, "config", "user.name", "CPU fixture")
        self.source_paths = (reader.RECOVERY_FILE, "src/wheeled_tasks/chassis/preflight.py", "src/wheeled_tasks/chassis/env.py")
        for name, content in zip(self.source_paths, (reader.PREPARE_CHANGE["old"]+"\n", "# synthetic preflight\n", "# synthetic env\n")):
            path = self.sa/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        git(self.sa, "add", ".")
        git(self.sa, "commit", "-qm", "Synthetic CPU reader H")
        self.h = git(self.sa, "rev-parse", "HEAD")
        self.base_core = {name: reader.sha(self.sa/name) for name in self.source_paths}
        subprocess.run(["git", "clone", "-q", str(self.sa), str(self.sb)], check=True, capture_output=True)
        git(self.sb, "config", "user.email", "fixture@example.invalid")
        git(self.sb, "config", "user.name", "CPU fixture")
        path = self.sb/reader.RECOVERY_FILE
        path.write_text(path.read_text().replace(reader.PREPARE_CHANGE["old"], reader.PREPARE_CHANGE["new"]))
        git(self.sb, "commit", "-qam", "Synthetic CPU reader C")
        self.c = git(self.sb, "rev-parse", "HEAD")
        self.engineering = self.root/"engineering_fixture.json"
        write(self.engineering, {"note": "synthetic reader test, not native engineering authentication"})
        self.expected_sha = reader.sha(self.engineering)
        self.patcher = patch.object(reader, "ENGINEERING_SHA", self.expected_sha)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.make_run(self.a, self.sa, "baseline", self.h)
        self.make_run(self.b, self.sb, "prepare_slew_1_candidate", self.c)

    def make_run(self, out, source, mode, revision):
        out.mkdir()
        runtime = {name: reader.sha(source/name) for name in self.source_paths}
        identity = {"schema": "recovery-prefix-source-identity-v1", "mode": mode, "source_root": str(source),
                    "source_revision": revision, "base_core_revision": reader.CORE_REVISION, "harness_revision": self.h,
                    "core_files_sha256": runtime, "base_core_files_sha256": self.base_core,
                    "runtime_sources_sha256": runtime, "candidate_diff_sha256": "d"*64,
                    "candidate_change": None if mode == "baseline" else reader.PREPARE_CHANGE,
                    "engineering_expectations_sha256": self.expected_sha}
        identity_path = self.root/(mode+"_identity.json")
        write(identity_path, identity)
        receipt = {"revision": revision, "base_revision": reader.CORE_REVISION, "harness_revision": self.h,
                   "mode": mode, "source_identity_path": str(identity_path), "source_identity_sha256": reader.sha(identity_path),
                   "core_files_sha256": runtime, "runtime_sources_sha256": runtime,
                   "candidate_change": identity["candidate_change"], "candidate_diff_sha256": "d"*64,
                   "engineering_expectations_sha256": self.expected_sha,
                   "actual_preflight_sha256": runtime["src/wheeled_tasks/chassis/preflight.py"]}
        (out/"engineering_expectations.json").write_bytes(self.engineering.read_bytes())
        write(out/"effective_contract.json", {"physics_dt": .005, "recovery_training": {"enabled": True},
              "design_preflight": {"file": str(out/"engineering_expectations.json"), "sha256": self.expected_sha}})
        cases = [f"synthetic_case_{i}" for i in range(4)]
        events, frames, policies, rows = [], [], [], []
        for lane, name in enumerate(cases):
            events.extend({"lane": lane, "case": name, "birth_id": 1, "event": event, "physical_tick": tick,
                           "sim_time_s": tick*.005, "snapshot_index": lane*2+index}
                          for index, (event, tick) in enumerate((("birth_complete", 0), ("terminal_pre_autoreset", 4))))
            for tick in range(1, 5):
                frames.append({"lane": lane, "case": name, "physical_tick": tick, "sim_time_s": tick*.005,
                    "phase_used": 7 if tick in (2,3) else 0, "effective_control_ownership": "script",
                    "q18": [0.]*18, "dq18": [float(tick)]*18,
                    "root_pose_world_xyzw": [0., 0., .305, 0., 0., 0., 1.],
                    "root_angular_velocity_body_rad_s": [float(tick), 0., 0.], "script_targets6": [tick*.01]*6,
                    "gates": {g: g != "support" for g in reader.GATES}, "ready_ticks": 1 if tick == 3 else 0})
            policies.append({"lane": lane, "case": name, "sim_time_s": .02, "native_recovery": {"recovery_ready": False}})
            rows.append({"lane": lane, "case": name, "replica": 0, "seed": 290929, "birth_id": 1,
                "events": {"birth_complete": 0., "terminal_pre_autoreset": .02}, "partition": "failure",
                "terminal": {"time_s": .02, "terminated": True, "success": False, "boundary": False, "timeout": False, "reasons": {}}})
        for name, values in (("events.jsonl", events), ("physical_frames.jsonl", frames), ("policy_frames.jsonl", policies)):
            (out/name).write_text("".join(json.dumps(v)+"\n" for v in values))
        (out/"event_snapshots.pt").write_bytes(b"synthetic provenance bytes, never deserialized")
        startup = {"runtime_preflight": {"status": "runtime_configuration_passed"}, "gpu_dynamics_enabled": True,
                   "active_joint_order": list(reader.ACTIVE_ORDER), "scene_group_counts": {n: 1 for n in cases},
                   **{k: {} for k in ("dynamics_randomization", "contact_domain", "command_transport", "actuator_response", "recovery_training")},
                   "mass_kg_each": [13.8]*4}
        report = {"schema": "v6-native-recovery-physical-prefix-v1", "execution_status": "completed",
            "status": "engineering_probe_completed", "num_envs": 4, "run_role": "full", "formal_updates_consumed": 0,
            "optimizer_steps": 0, "actor_tensors_unchanged": True, "checkpoint_unchanged": True,
            "actor_weights_loaded_strictly": True, "source_files_changed": [], "qualification": False,
            "physx_internal_state_restore_claim": False, "randomization_or_delay_overrides": [],
            "source_worker_sha256": reader.WORKER_SHA, "checkpoint": {"checkpoint_sha256": reader.PT_SHA,
                "actor_only_deterministic_inference": True, "seal_and_infos_authenticated": True,
                "checkpoint_contract_sha256": "a"*64, "checkpoint_learning_updates": 14020, "checkpoint_clocks": {}},
            "requested_episode_seconds": 20., "signal_delay_enabled": False, "command_transport_enabled": True,
            "control_timing": {"physics_dt": .005, "policy_dt": .02, "pd_updates": 4, "feedback_refreshes": 4,
                               "target_decodes": 1, "physics_steps": 4},
            "physics_steps": 4, "policy_steps": 1, "elapsed_physics_s": .02, "wall_seconds": .1, "startup": startup,
            "native_joint_names": list(reader.ACTIVE_ORDER)+[f"aux{i}" for i in range(12)], "joint_position_units": ["rad"]*18,
            "source_identity_final_verified": True, "frozen_physical_control_sources": receipt, "source_files_sha256": runtime,
            "contract_sha256": reader.sha(out/"effective_contract.json"), "case_names": cases, "seed": 290929, "stage": "mixed",
            "summary": {"fixed_birth_denominator": 4, "rows": rows, "event_count": len(events), "physical_frame_count": len(frames),
                        "policy_frame_count": len(policies), "partitions": {p: 4 if p == "failure" else 0 for p in reader.PARTITIONS},
                        "all_first_cohorts_closed": True, "terminal_pending_lanes": [], "actual_blend_physical_substeps": {},
                        "events_per_seeded_cohort": {e: {"numerator": 4, "denominator": 4, "fraction": 1.}
                                                   for e in ("birth_complete", "terminal_pre_autoreset")}}}
        write(out/"report.json", report)

    def fail_report(self, mutate_fn, message):
        mutate(self.b/"report.json", mutate_fn)
        with self.assertRaisesRegex(ValueError, message):
            reader.compare(self.a, self.b)

    def test_distinct_roots_pre_post_and_native_gates(self):
        result = reader.compare(self.a, self.b)
        self.assertFalse(result["qualification"])
        p = result["cases"][0]["candidate"]["prepare"]
        self.assertEqual(p["pre_physics"]["gyro_body_norm_rad_s"]["mean"], 1.5)
        self.assertEqual(p["post_physics"]["gyro_body_norm_rad_s"]["mean"], 2.5)
        self.assertEqual(p["native_gate_counts"]["support"]["passed"], 0)
        self.assertAlmostEqual(p["leg_target_slew_rad_s"]["L_joint1"]["mean"], 2.)
        self.assertEqual(result["candidate"]["fixed_birth_denominator"], 4)
        self.assertTrue(result["cases"][0]["prefix_before_first_prepare"]["all_recorded_prefix_fields_exactly_equal"])
        self.assertTrue(result["cases"][0]["prefix_before_first_prepare"]["first_prepare_entry_frame_exactly_equal"])
        self.assertEqual(result["candidate_disposition"], "candidate_not_merged_or_deployed")

    def test_failed_execution_rejected(self):
        self.fail_report(lambda r: r.update(execution_status="failed"), "completed")

    def test_tensor_change_rejected(self):
        self.fail_report(lambda r: r.update(actor_tensors_unchanged=False), "tensors changed")

    def test_formal_update_rejected(self):
        self.fail_report(lambda r: r.update(formal_updates_consumed=1), "formal updates")

    def test_external_source_verification_required(self):
        self.fail_report(lambda r: r.update(source_identity_final_verified=False), "finally verified")

    def test_changed_seed_rejected(self):
        self.fail_report(lambda r: r.update(seed=7), "cohort identity")

    def test_partition_denominator_rejected(self):
        self.fail_report(lambda r: r["summary"].update(fixed_birth_denominator=3), "denominator")

    def test_event_ledger_tamper_rejected(self):
        self.fail_report(lambda r: r["summary"]["rows"][0]["events"].update(phase_BLEND=.01), "Events differ")

    def test_effective_contract_difference_rejected(self):
        mutate(self.b/"effective_contract.json", lambda c: c.update(physics_dt=.01))
        mutate(self.b/"report.json", lambda r: r.update(contract_sha256=reader.sha(self.b/"effective_contract.json")))
        with self.assertRaisesRegex(ValueError, "Effective contracts differ"):
            reader.compare(self.a, self.b)

    def test_dirty_source_checkout_rejected(self):
        (self.sb/"untracked").write_text("changed")
        with self.assertRaisesRegex(ValueError, "no longer clean"):
            reader.compare(self.a, self.b)

    def test_manifest_sha_tamper_rejected(self):
        mutate(self.root/"prepare_slew_1_candidate_identity.json", lambda x: x.update(harness_revision="b"*40))
        with self.assertRaisesRegex(ValueError, "identity SHA"):
            reader.compare(self.a, self.b)

    def test_nonfinite_physical_value_rejected(self):
        path = self.b/"physical_frames.jsonl"
        values = [json.loads(s) for s in path.read_text().splitlines()]
        values[1]["dq18"][0] = float("inf")
        path.write_text("".join(json.dumps(v)+"\n" for v in values))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            reader.compare(self.a, self.b)


if __name__ == "__main__":
    unittest.main()
