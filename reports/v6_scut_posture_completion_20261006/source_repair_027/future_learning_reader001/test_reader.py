"""CPU-only synthetic fixtures; these are not adoption or runtime receipts."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import reader

ROOT = Path(__file__).resolve().parents[4]
KERNEL_PATH = ROOT / "reports/v6_scut_posture_completion_20261006/learning_audit/read_recovery_window_019.py"


class ReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = reader.load_kernel(KERNEL_PATH)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source_59c201678b37"
        self.run = self.root / "train/speed/attempt_011/train"
        self.run.mkdir(parents=True)
        self.source.mkdir()
        for name in reader.SOURCE_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        self.recipe = reader.read_json(ROOT / "contracts/v6_integrated_runtime_recipe_v25.json")
        self.worker = reader.read_json(ROOT / "contracts/v6_task_campaign_speed_worker_v25.json")
        self.pins = {}

        def save(name, path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2) + "\n")
            self.pins[name] = dict(path=str(path), sha256=reader.digest(path))

        # Preserve actual frozen recipe/worker bytes instead of JSON reserializing.
        for name, original in (("recipe", ROOT / "contracts/v6_integrated_runtime_recipe_v25.json"),
                               ("worker", ROOT / "contracts/v6_task_campaign_speed_worker_v25.json")):
            path = self.source / "contracts" / original.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original.read_bytes())
            self.pins[name] = dict(path=str(path), sha256=reader.digest(path))
        revision = dict(schema="task-plan-revision-v25", source_commit=reader.SOURCE_COMMIT,
                        new_source_root=str(self.source), source_archive_sha256=reader.ARCHIVE_SHA)
        save("revision", self.root / "source_revision_027.json", revision)
        save("adoption", self.root / "boundary_continuation_027.receipt.json",
             dict(status="launch_submitted", revision=revision, reason="rec_boundary_native_continuation_authenticated"))
        self.selection = dict(domain="speed", lineage_id="source027-joint-heading-v1", mode="rebase_from_normal_REC",
                              plan_sha256="b" * 64, origin_checkpoint_sha256="a" * 64,
                              fresh_joint_actor_updates=0,
                              checkpoint=dict(path="synthetic.pt", sha256="a" * 64),
                              checkpoint_consumed_updates=8000, fresh_domain_actor_updates=0,
                              critic_warmup_remaining=10, contract_sha256=self.pins["worker"]["sha256"])
        native_clock_file = self.source / "src/wheeled_tasks/chassis/integrated_runtime.py"
        helpers = reader.clock_helpers(native_clock_file)
        source_clocks = helpers["_clock_document"](self.selection, {"speed": 0}, 0, 0)
        admission = dict(schema="v6-integrated-production-admission-v1", qualification=None,
                         target_contract_sha256=self.pins["worker"]["sha256"], selection=self.selection,
                         allowance=dict(admitted=True, consumed_updates=9000), component_clocks_at_source=source_clocks)
        save("selection", self.run.parent / "integrated_source_selection.json", admission)
        admission_path = self.run / "integrated_admission.json"
        admission_path.write_bytes(Path(self.pins["selection"]["path"]).read_bytes())
        self.pins["admission"] = dict(path=str(admission_path), sha256=reader.digest(admission_path))
        argv = ["python", "-B", str(self.source / "scripts/train_chassis.py"),
                "--run-dir", str(self.run), "--contract", self.pins["worker"]["path"],
                "--integrated-source-selection-file", self.pins["selection"]["path"],
                "--integrated-source-selection-sha256", self.pins["selection"]["sha256"],
                "--consumed-updates", "9000", "--task-actor-update", "0", "--stage-actor-update", "15000",
                "--actor-updates-consumed", "8900"]
        self.owner = dict(pid=4242, ppid=4241, start_ticks=123456, command=argv)
        save("invocation", self.run.parent / "invocation.json", dict(command=argv))
        config = reader.materialize(self.worker, self.recipe,
                                   self.source / "src/wheeled_tasks/chassis/integrated_production.py", admission)
        groups = [row["name"] for row in config["scene_groups"]]
        self.population = config["integrated_sampling_candidate"]["allocation"]["counts"]
        scope = dict(row_population=self.population,
                     row_ownership=config["integrated_runtime_candidate"]["row_ownership"],
                     clocks_at_source=admission["component_clocks_at_source"])
        save("design", self.run / "integrated_design.json", dict(source_receipt=admission,
             selection=dict(sha256=self.pins["selection"]["sha256"]), runtime_expected=dict(integrated_curriculum=scope)))
        counts = {}
        for group, n in self.population.items():
            pool = config["behavior_pool_membership"][group]
            counts[pool] = counts.get(pool, 0) + n
        save("startup", self.run / "startup.json", dict(contract_sha256=self.pins["worker"]["sha256"],
             integrated_admission_sha256=self.pins["selection"]["sha256"], args=dict(run_dir=str(self.run)),
             source_sha256=reader.SOURCE_FILES, startup=dict(scene_group_counts=self.population,
             behavior_pool_counts=counts, runtime_preflight=dict(status="runtime_configuration_passed",
             scoped_validation="v6-integrated-preflight-candidate-v1", observed=dict(physics_dt=.005,
             policy_dt=.02, decimation=4, integrated_curriculum_observed=dict(schema="v6-integrated-preflight-candidate-v1",
             task_actor_updates=source_clocks["domain_actor_updates"], component_actor_updates=source_clocks["component_actor_updates"]))))))
        save("resume", self.run / "resume_verification.json", dict(parent_checkpoint_sha256="a" * 64,
             actor_exact=True, critic_exact=True, optimizer_exact=True, parent_updates=9000, restored_checkpoint_updates=8000))
        save("agent", self.run / "agent_config.json", dict(num_steps_per_env=24))
        self.progress = dict(pid=4242, num_envs=4096, successful_updates=210, actor_updates_in_block=200,
                             parent_updates=9000, task_actor_update=200, stage_actor_update=15200,
                             global_budget_actor_updates=9100, attempted_updates=211)
        current_clocks = helpers["advance_clock_document"](source_clocks, 200)
        self.progress["integrated_runtime_clocks"] = current_clocks
        self.progress["integrated_lineage"] = {k: current_clocks[k] for k in
            ("lineage_id", "plan_sha256", "origin_checkpoint_sha256", "domain", "joint_actor_updates", "domain_actor_updates")}
        self.progress["integrated_lineage"]["domain_complete"] = False
        (self.run / "progress.json").write_text(json.dumps(self.progress))
        self.binding = dict(schema="v25-readonly-actual-authority-v1", reviewed_actual_adoption=True,
                            source_commit=reader.SOURCE_COMMIT, campaign_root=str(self.root),
                            source_root=str(self.source), run_dir=str(self.run), files=self.pins, owner=self.owner)

    def authority(self):
        with patch.object(reader, "read_owner", return_value=self.owner), patch.object(reader, "campaign_workers", return_value=[self.owner]):
            return reader.authenticate(self.binding)

    def change(self, name, value):
        path = Path(self.pins[name]["path"])
        path.write_text(json.dumps(value))
        self.pins[name]["sha256"] = reader.digest(path)

    def series(self, authority, window):
        k, data = self.kernel, {}
        def add(tag, step, value):
            data.setdefault(tag, {})[step] = [float(value), float(step)]
        for group in authority["groups"]:
            n = authority["population"][group]
            for step in window["behavior_steps"]:
                completed = step - authority["clocks"]["parent_updates"] + 1
                frames = completed * n * 24
                values = dict(frames=frames, observed_frames=frames, episodes=completed,
                              successes=completed // 2, failures=0, timeouts=completed-completed//2,
                              boundary_truncations=0, blocked_truncations=0)
                for f, value in values.items():
                    add(f"Behavior/{group}/{f}", step, value)
                for f in (*k.MEANS, "tilt_max_deg"):
                    add(f"Behavior/{group}/{f}", step, .1)
            for step in window["dense_steps"]:
                for field in k.STATIONARY_FIELDS:
                    add(f"LearningAudit/{group}/stationary_10s/{field}", step, 0)
        for step in window["dense_steps"]:
            for f in k.TIMING_FIELDS:
                add("Timing/" + f, step, 10 if f == "iteration_wall_seconds" else 1)
            for g in ("forward_05", "backward_05"):
                for f in reader.PAID_FIELDS:
                    add(f"LearningAudit/{g}/{f}", step, 24 * authority["population"][g] if f == "rollout_samples" else step)
            for g in reader.stop_groups(authority):
                for phase in ("braking", "settled_stop"):
                    for f in k.STOP_FIELDS:
                        add(f"LearningAudit/{g}/{phase}/{f}", step, 1 if f == "samples" else .02)
            add("RecoveryRollout/valid_transitions", step, 90000)
            add("RecoveryRollout/script_or_blend_transitions", step, 8304)
            add("RecoveryRollout/valid_fraction", step, 90000 / 98304)
        for lesson in k.LESSONS:
            before, after = ([17,9,3,5], [25,12,6,7]) if lesson == "original_chain" else ([0,0,0,0], [0,0,0,0])
            for step, counts in ((window["baseline_step"], before), (window["end_step"], after)):
                for f, value in zip(reader.OUTCOME_FIELDS, counts):
                    add(f"RecoveryOutcomes/{lesson}/{f}", step, value)
        return data

    def test_dynamic_runtime_catalog_and_charged_parent(self):
        a = self.authority()
        self.assertEqual(len(self.worker["scene_groups"]), 135)
        self.assertEqual(len(a["groups"]), 189)
        self.assertEqual(sum(a["population"].values()), 4096)
        self.assertEqual(a["clocks"]["parent_updates"], 9000)

    def test_staged_default_refused(self):
        self.binding["reviewed_actual_adoption"] = False
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.authority()

    def test_waiting_boundary_refused(self):
        x = reader.read_json(self.pins["adoption"]["path"]); x["status"] = "waiting"
        self.change("adoption", x)
        with self.assertRaisesRegex(ValueError, "staged/waiting"):
            self.authority()

    def test_missing_runtime_metadata_refused(self):
        del self.binding["files"]["design"]
        with self.assertRaisesRegex(ValueError, "pins"):
            self.authority()

    def test_live_account_growth_is_not_startup_readmission(self):
        # Reader never opens or authenticates mutable durable account totals.
        account = self.root / "campaign.account.json"
        account.write_text(json.dumps(dict(consumed_updates=12345)))
        self.assertTrue(self.authority()["identity_verified"])
        account.write_text(json.dumps(dict(consumed_updates=23456)))
        self.assertTrue(self.authority()["identity_verified"])

    def test_wrong_start_ticks_refused(self):
        self.binding["owner"] = {**self.owner, "start_ticks": 99999}
        with self.assertRaisesRegex(ValueError, "owner"):
            self.authority()

    def test_duplicate_worker_refused(self):
        with patch.object(reader, "read_owner", return_value=self.owner), patch.object(reader, "campaign_workers", return_value=[self.owner, self.owner]):
            with self.assertRaisesRegex(ValueError, "owner"):
                reader.authenticate(self.binding)

    def test_static_catalog_cannot_replace_runtime(self):
        x = reader.read_json(self.pins["startup"]["path"])
        x["startup"]["scene_group_counts"] = {r["name"]: 1 for r in self.worker["scene_groups"]}
        self.change("startup", x)
        with self.assertRaisesRegex(ValueError, "catalog"):
            self.authority()

    def test_mutually_matching_but_wrong_lane_allocation_refused(self):
        startup = reader.read_json(self.pins["startup"]["path"])
        design = reader.read_json(self.pins["design"]["path"])
        counts = dict(self.population)
        a, b = list(counts)[:2]; counts[a] += 1; counts[b] -= 1
        startup["startup"]["scene_group_counts"] = counts
        design["runtime_expected"]["integrated_curriculum"]["row_population"] = counts
        self.change("startup", startup); self.change("design", design)
        with self.assertRaisesRegex(ValueError, "catalog"):
            self.authority()

    def test_missing_joint_metadata_or_wrong_scoped_clock_refused(self):
        bad = dict(self.progress); del bad["integrated_lineage"]
        (self.run / "progress.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "joint/component"):
            self.authority()
        (self.run / "progress.json").write_text(json.dumps(self.progress))
        startup = reader.read_json(self.pins["startup"]["path"])
        startup["startup"]["runtime_preflight"]["observed"]["integrated_curriculum_observed"]["component_actor_updates"]["recovery"] += 1
        self.change("startup", startup)
        with self.assertRaisesRegex(ValueError, "startup component"):
            self.authority()

    def test_attempted_clock_does_not_complete_window(self):
        a = self.authority()
        self.assertEqual(a["progress"]["attempted_updates"], 211)
        reader.check_progress(a, self.progress)
        bad = {**self.progress, "successful_updates": 211}
        with self.assertRaisesRegex(ValueError, "clocks"):
            reader.check_progress(a, bad)

    def test_cadence_and_actual_critic_clock(self):
        a = self.authority()
        w = reader.window_spec(a, 200)
        self.assertEqual((w["baseline_step"], w["start_step"], w["end_step"]), (9109, 9110, 9209))
        a["clocks"]["critic_warmup_updates"] = 3
        self.assertEqual(reader.window_spec(a, 207)["end_step"], 9209)
        with self.assertRaisesRegex(ValueError, "cadence"):
            reader.window_spec(a, 200)

    def test_complete100_payment_halves_and_exact_L2_outcomes(self):
        a = self.authority(); w = reader.window_spec(a, 200)
        result = reader.summarize(self.kernel, self.series(a, w), a, {}, w)
        self.assertTrue(result["complete"])
        self.assertEqual(result["recovery_outcomes"]["original_chain"]["success_fraction"], 3/8)
        self.assertIsNone(result["recovery_outcomes"]["prefix_stand"]["success_fraction"])
        self.assertEqual(result["learning_ownership"]["observed_transitions"], 9830400)
        self.assertIn("release_stop_translate", result["conditional_stop"])
        self.assertNotEqual(a["group_pools"]["release_stop_translate"], "transitions")
        for group in result["payment"].values():
            self.assertIn("continuous", group["command_semantics"])
            for field in group["fields"].values():
                self.assertEqual([field[n]["records"] for n in ("full100","first50","second50")], [100,50,50])

    def test_missing_payment_and_outcome_are_not_zero(self):
        a = self.authority(); w = reader.window_spec(a, 200); s = self.series(a, w)
        del s["LearningAudit/forward_05/reward_density/wheel_power"][w["dense_steps"][0]]
        del s["RecoveryOutcomes/original_chain/ended"][w["baseline_step"]]
        result = reader.summarize(self.kernel, s, a, {}, w)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["payment"]["forward_05"]["fields"]["reward_density/wheel_power"]["full100"]["mean"])
        self.assertIsNone(result["recovery_outcomes"]["original_chain"]["success_fraction"])

    def test_outcome_partition_or_worker_reset_rejected(self):
        a = self.authority(); w = reader.window_spec(a, 200); s = self.series(a, w)
        s["RecoveryOutcomes/original_chain/ended"][w["end_step"]][0] = 26
        with self.assertRaisesRegex(ValueError, "partition"):
            reader.summarize(self.kernel, s, a, {}, w)
        s = self.series(a, w)
        for f in reader.OUTCOME_FIELDS:
            s[f"RecoveryOutcomes/original_chain/{f}"][w["end_step"]][0] = 0
        with self.assertRaisesRegex(ValueError, "regressed"):
            reader.summarize(self.kernel, s, a, {}, w)

    def test_wanted_tags_include_only_reported_fields(self):
        a = self.authority()
        tags = reader.wanted_tags(self.kernel, a)
        self.assertNotIn("Policy/mean_std", tags)
        self.assertFalse(any(tag.startswith("/recovery/lesson/") for tag in tags))
        self.assertFalse(any("reward_density/pitch" in tag for tag in tags))
        self.assertIn("RecoveryOutcomes/original_chain/ended", tags)
        self.assertIn("LearningAudit/release_stop_translate/settled_stop/samples", tags)

    def test_incomplete_actor_window_prevents_event_read(self):
        a = self.authority(); a["progress"]["actor_updates_in_block"] = 190
        with patch.object(reader, "authenticate", return_value=a), patch.object(reader, "load_kernel") as parser:
            with self.assertRaisesRegex(ValueError, "has not completed"):
                reader.read_window(self.binding, KERNEL_PATH, 200)
            parser.assert_not_called()

    def test_original_crc_kernel_and_suffix_limit(self):
        k = self.kernel
        def frame(step):
            e = k.Event(step=step, wall_time=step)
            e.summary.value.add(tag="Timing/iteration_wall_seconds", simple_value=12.)
            payload = e.SerializeToString(); length = struct.pack("<Q", len(payload))
            return length + struct.pack("<I", k.masked_crc32c(length)) + payload + struct.pack("<I", k.masked_crc32c(payload))
        path = self.root / "synthetic.events"
        path.write_bytes(b"".join(frame(i) for i in range(300)))
        series, audit = k.parse_events(path, {"Timing/iteration_wall_seconds"}, 299, min_step=290, max_read_bytes=2048)
        self.assertEqual(len(series["Timing/iteration_wall_seconds"]), 10)
        self.assertGreater(audit["source_region_start_offset"], 0)
        self.assertLessEqual(audit["read_complete_record_bytes"], 2048)
        self.assertFalse(audit["earlier_records_crc_verified"])
        broken = bytearray(frame(300)); broken[-1] ^= 1; path.write_bytes(broken)
        with self.assertRaisesRegex(ValueError, "CRC"):
            k.parse_events(path, {"Timing/iteration_wall_seconds"}, 300, min_step=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
