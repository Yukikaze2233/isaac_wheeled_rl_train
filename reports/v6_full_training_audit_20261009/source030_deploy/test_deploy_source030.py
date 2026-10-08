"""CPU-only deployment boundary tests; no simulator, production process or SSH.

Synthetic artifact bytes exercise deployment guards only, not native full-state
qualification. The existing production migration/full-state tests remain separate.
"""
import copy
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("deploy030", HERE / "deploy_source030.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def exclusive(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = SimpleNamespace(run_root=root, old_source_root=root / "source029",
            new_source_root=root / "source030", source_archive=root / "source030.tar.gz",
            source_commit="b" * 40, source_archive_sha256="", controller_intent=None,
            controller_receipt_sha256=None, runtime_script=root / "runtime.sh")
        self.args.old_source_root.mkdir()
        (root / "train").mkdir()
        self.directory = root / "source030_deploy"
        self.native = SimpleNamespace(owners=lambda root: [], publish=exclusive,
            validate_revision=lambda *args: None)
        self.claim = root / "campaign.account.json"
        put(self.claim, {})

    def source_archive(self, *, bad=None):
        put(self.args.old_source_root / d.OLD_CONTRACT, {"old": True})
        members = {name: b'{}' for name in d.REQUIRED}
        members[str(d.OLD_CONTRACT)] = (self.args.old_source_root / d.OLD_CONTRACT).read_bytes()
        with tarfile.open(self.args.source_archive, "w:gz", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": self.args.source_commit}) as archive:
            for name, content in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            if bad:
                archive.addfile(bad, io.BytesIO(b""))
        self.args.source_archive_sha256 = d.sha(self.args.source_archive)

    def closed_fixture(self):
        args, root = self.args, self.args.run_root
        args.controller_intent = root / "boundary.intent.json"
        source, train = args.old_source_root, root / "train"
        worker_dir = train / "speed/attempt_003/train"
        worker_dir.mkdir(parents=True)
        checkpoint = worker_dir / "model_final.pt"
        checkpoint.write_bytes(b"synthetic full state for deployment guard only")
        put(checkpoint.with_suffix(".checkpoint.json"), {"completed": True})
        put(checkpoint.with_suffix(".contract.json"), {"domain": "speed"})
        put(worker_dir / "completion.json", {"status": "completed", "successful_updates": 14907})
        reference = {"path": str(checkpoint), "sha256": d.sha(checkpoint)}
        command = [sys.executable, "-B", str(source / "scripts/run_task_campaign.py"),
            "--contract", str(source / d.OLD_CONTRACT), "--run-dir", str(train), "--num-envs", "4096", "--research"]
        job = {"pid": 900001, "command": command}
        put(root / "job.json", job)
        intent = {"schema": "native-completed-speed-supervisor-hold-v1",
            "operation": "hold_supervisor_until_normal_speed_end", "worker_signals": 0, "qualification": None,
            "supervisor": {"pid": job["pid"], "command": command},
            "worker": {"ppid": job["pid"], "command": [str(source / "scripts/train_chassis.py"),
                "--run-dir", str(worker_dir)]}}
        put(args.controller_intent, intent)
        totals = {"consumed_updates": 56300, "actor_updates_consumed": 56120,
            "domain_consumed_updates": {"speed": 20000, "terrain": 0}}
        record = {"status": "joint_domain_finished_unassessed", "candidate_checkpoint": str(checkpoint),
            "candidate_checkpoint_sha256": reference["sha256"]}
        put(train / "campaign_state.json", dict(totals, domains={"speed": record}))
        put(train / "consumption.json", totals)
        receipt = {"status": "native_supervisor_stopped_after_normal_speed", "worker_signals": 0,
            "next_domain_dispatched": False, "qualification": None,
            "native_completion_recovery": {"status": "closed_native_completed_speed", "PPO_updates": 0,
                "next_domain_dispatched": False, "qualification": None, "checkpoint": reference, "account": totals},
            "evidence": {"checkpoint": reference, "selection": {"checkpoint": reference, "domain": "speed",
                "domain_complete": True, "critic_warmup_remaining": 0}, "PPO_updates_by_controller": 0}}
        put(args.controller_intent.with_suffix(".receipt.json"), receipt)
        args.controller_receipt_sha256 = d.sha(args.controller_intent.with_suffix(".receipt.json"))
        files = [checkpoint, checkpoint.with_suffix(".checkpoint.json"), checkpoint.with_suffix(".contract.json"),
            worker_dir / "completion.json", root / "job.json", train / "campaign_state.json", train / "consumption.json"]
        archive = root / "delivery.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for path in files:
                output.add(path, arcname=str(path.relative_to(root)))
        delivery = {"status": "ready", "archive": archive.name, "exit_code": 0,
            "archive_bytes": archive.stat().st_size, "archive_sha256": d.sha(archive),
            "files": {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": d.sha(p)} for p in files}}
        put(root / "delivery.json", delivery)
        put(root / "archiver_status.json", {"status": "finished", "pid": job["pid"]})
        return receipt, checkpoint

    def rewrite_receipt(self, value):
        path = self.args.controller_intent.with_suffix(".receipt.json")
        put(path, value)
        self.args.controller_receipt_sha256 = d.sha(path)

    def test_static_stage_does_not_read_live_job_or_bind_parent(self):
        self.source_archive()
        with patch.object(d, "load_native", return_value=self.native):
            result = d.stage(self.args)
        self.assertFalse(result["dynamic_parent_bound"])
        self.assertFalse((self.args.run_root / "source_revision_030.json").exists())
        self.assertFalse((self.args.run_root / "job.json").exists())
        with self.assertRaisesRegex(ValueError, "Prior"):
            d.stage(self.args)

    def test_archive_sha_rejected_before_stage_mutation(self):
        self.source_archive()
        self.args.source_archive_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "Archive SHA"):
            d.stage(self.args)
        self.assertFalse(self.directory.exists())

    def test_archive_link_traversal_duplicate_and_pax_rejected(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("linked", tarfile.SYMTYPE),
                           (str(d.CONTRACT), tarfile.REGTYPE)):
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "/etc/passwd"
                self.source_archive(bad=member)
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    d.archive_files(self.args.source_archive, self.args.source_commit)
        self.source_archive()
        with self.assertRaisesRegex(ValueError, "PAX"):
            d.archive_files(self.args.source_archive, "c" * 40)

    def test_legacy_contract_bytes_and_only_twelve_new_enforced(self):
        self.source_archive()
        files = d.archive_files(self.args.source_archive, self.args.source_commit)
        d.contract_inventory(self.args.old_source_root, files)
        for bad in (dict(files, **{str(d.OLD_CONTRACT): "f" * 64}),
                    dict(files, **{"contracts/unsolicited.json": "f" * 64})):
            with self.assertRaises(ValueError):
                d.contract_inventory(self.args.old_source_root, bad)

    def test_group_writable_archive_and_changed_executable_permission_rejected(self):
        member = tarfile.TarInfo("bad_permission.py")
        member.mode = 0o664
        self.source_archive(bad=member)
        with self.assertRaisesRegex(ValueError, "permission"):
            d.archive_files(self.args.source_archive, self.args.source_commit)
        self.source_archive()
        with patch.object(d, "load_native", return_value=self.native):
            d.stage(self.args)
            d.validate_stage(self.args)
            (self.args.new_source_root / d.CONTRACT).chmod(0o755)
            with self.assertRaisesRegex(ValueError, "permission"):
                d.validate_stage(self.args)

    def test_closed_archive_allows_post_archive_mutable_ledger_change(self):
        self.closed_fixture()
        state = self.args.run_root / "train/campaign_state.json"
        put(state, dict(d.read(state), status="native_zero_ppo_closed"))
        put(self.args.run_root / "train/consumption.json", {"new_native_close_bytes": True})
        result = d.closed_boundary(self.args, self.native)
        self.assertEqual(result["closed_account"]["domain_consumed_updates"]["speed"], 20000)

    def test_actual_final_seal_mutation_rejected(self):
        _, checkpoint = self.closed_fixture()
        checkpoint.with_suffix(".checkpoint.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "immutable speed artifact"):
            d.closed_boundary(self.args, self.native)

    def test_controller_parent_or_incomplete_or_postdispatch_rejected(self):
        receipt, _ = self.closed_fixture()
        for mutate in (lambda v: v["evidence"]["selection"].update(domain_complete=False),
                       lambda v: v["native_completion_recovery"]["checkpoint"].update(sha256="0" * 64),
                       lambda v: v.update(next_domain_dispatched=True),
                       lambda v: v["native_completion_recovery"].update(PPO_updates=1)):
            value = copy.deepcopy(receipt)
            mutate(value)
            self.rewrite_receipt(value)
            with self.assertRaises(ValueError):
                d.closed_boundary(self.args, self.native)

    def test_live_owner_and_stale_archiver_and_archive_corruption_rejected(self):
        self.closed_fixture()
        with patch.object(self.native, "owners", return_value=[{"role": "wrapper"}]):
            with self.assertRaisesRegex(ValueError, "still alive"):
                d.closed_boundary(self.args, self.native)
        path = self.args.run_root / "archiver_status.json"
        original = d.read(path)
        put(path, dict(original, pid=original["pid"] + 1))
        with self.assertRaisesRegex(ValueError, "wrapper"):
            d.closed_boundary(self.args, self.native)
        put(path, original)
        with (self.args.run_root / "delivery.tar.gz").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "wrapper"):
            d.closed_boundary(self.args, self.native)

    def test_budget_refund_or_foreign_job_rejected(self):
        self.closed_fixture()
        state = self.args.run_root / "train/campaign_state.json"
        original = d.read(state)
        put(state, dict(original, consumed_updates=original["consumed_updates"] - 1))
        with self.assertRaisesRegex(ValueError, "clocks"):
            d.closed_boundary(self.args, self.native)
        put(state, original)
        path = self.args.run_root / "job.json"
        value = d.read(path)
        value["command"][-1] = "--evaluate-only"
        put(path, value)
        with self.assertRaisesRegex(ValueError, "exact old"):
            d.closed_boundary(self.args, self.native)

    def test_lock_order_and_account_not_touched_if_controller_busy(self):
        train = self.args.run_root / "train"
        calls = []
        original = fcntl.flock
        def observe(fd, operation):
            calls.append(Path(os.readlink(f"/proc/self/fd/{fd.fileno()}")))
            return original(fd, operation)
        with patch.object(d.fcntl, "flock", side_effect=observe):
            with d.boundary_locks(train, self.claim):
                pass
        self.assertEqual(calls, [train / "native_speed_boundary_controller.lock", self.claim.with_suffix(".lock")])
        script = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); sys.stdin.read()"
        child = subprocess.Popen([sys.executable, "-c", script, str(calls[0])], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "locked")
            calls.clear()
            with patch.object(d.fcntl, "flock", side_effect=observe):
                with self.assertRaises(BlockingIOError):
                    with d.boundary_locks(train, self.claim):
                        self.fail("Must not obtain live controller lock")
            self.assertEqual(calls, [train / "native_speed_boundary_controller.lock"])
        finally:
            child.communicate("", timeout=5)

    def test_account_lock_failure_releases_controller_lock(self):
        train = self.args.run_root / "train"
        with self.claim.with_suffix(".lock").open("a") as occupied:
            fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with d.boundary_locks(train, self.claim):
                    self.fail("Must not obtain account lock")
        with d.boundary_locks(train, self.claim):
            pass

    def test_prepare_bad_boundary_never_mutates_native_account(self):
        self.directory.mkdir()
        put(self.directory / "preflight.json", {"status": "CPU_preflight_passed"})
        with patch.object(d, "closed_boundary", side_effect=ValueError("wrong parent")), patch.object(d, "native_admission") as admit:
            with self.assertRaisesRegex(ValueError, "wrong parent"):
                d.prepare(self.args, self.directory, self.native, self.claim)
        admit.assert_not_called()
        self.assertFalse((self.directory / "prepare.intent.json").exists())

    def test_prepare_native_failure_retains_intent_refuses_reentry(self):
        self.directory.mkdir()
        put(self.directory / "preflight.json", {"status": "CPU_preflight_passed"})
        boundary = {"job": {}, "checkpoint": {"path": "actual"}}
        with patch.object(d, "closed_boundary", return_value=boundary), patch.object(d, "revision_receipt", return_value={"command": []}), \
                patch.object(d, "native_admission", side_effect=ValueError("native rejected parent")) as admit:
            with self.assertRaisesRegex(ValueError, "native rejected"):
                d.prepare(self.args, self.directory, self.native, self.claim)
            with self.assertRaisesRegex(ValueError, "Prior"):
                d.prepare(self.args, self.directory, self.native, self.claim)
        self.assertEqual(admit.call_count, 1)
        self.assertFalse((self.args.run_root / "source_revision_030.json").exists())

    def test_launch_requires_prepared_state_and_never_retries_ambiguous_intent(self):
        self.directory.mkdir()
        put(self.directory / "launch.intent.json", {"status": "ambiguous"})
        with patch.object(d.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Prior"):
                d.launch(self.args, self.directory, self.native, self.claim)
        popen.assert_not_called()

    def test_launch_changed_account_rejected_before_submission(self):
        self.directory.mkdir()
        saved = {"status": "boundary_prepared_not_launched"}
        for key in ("admission", "source_revision", "account", "state"):
            path = self.args.run_root / (key + ".json")
            put(path, {"initial": True})
            saved[key] = d.pin(path)
        put(self.directory / "prepare.json", saved)
        put(saved["account"]["file"], {"modified": True})
        with patch.object(d.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Prepared"):
                d.launch(self.args, self.directory, self.native, self.claim)
        popen.assert_not_called()

    def test_native_adapter_calls_actual_api_order_and_refuses_wrong_origin(self):
        """Exercise adapter wiring with spy native endpoints; no full-state claim."""
        calls = []
        boundary = {"checkpoint": {"path": "future-real-final", "sha256": "a" * 64},
            "closed_account": {"consumed_updates": 56300}}
        receipt = {"selection": {"domain": "terrain", "checkpoint": boundary["checkpoint"],
            "fresh_domain_actor_updates": 0, "critic_warmup_remaining": 10},
            "allowance": {"admitted": True, "charged_critic_updates": 10},
            "reconciled_account": boundary["closed_account"]}
        row = {"name": "terrain", "contract": "contracts/terrain.json"}
        put(self.args.new_source_root / row["contract"], {})
        class FakeCampaign:
            def __init__(inner, args):
                calls.append("construct")
                inner.plan, inner.root = {"domains": [row]}, args.run_dir
                inner.claim, inner.records, inner._joint_selection = self.claim, {"retained": 1}, {}
            def open_account(inner):
                calls.append("open_account")
            def learning_source(inner, actual_row):
                self.assertEqual(actual_row, row)
                calls.append("learning_source")
        runner = ModuleType("run_task_campaign")
        runner.TaskCampaign = FakeCampaign
        joint = ModuleType("wheeled_tasks.chassis.integrated_production")
        def make(*args, **kwargs):
            calls.append("make_admission")
            self.assertEqual(kwargs["snapshot_directory"], self.args.run_root / "train/revisions/probe")
            return receipt
        joint.make_admission = make
        joint.authenticate_receipt = lambda *args: calls.append("authenticate_receipt")
        chassis = ModuleType("wheeled_tasks.chassis")
        chassis.integrated_production = joint
        package = ModuleType("wheeled_tasks")
        package.chassis = chassis
        with patch.dict(sys.modules, {"run_task_campaign": runner, "wheeled_tasks": package,
                "wheeled_tasks.chassis": chassis, "wheeled_tasks.chassis.integrated_production": joint}):
            result = d.native_admission(self.args, boundary, self.args.run_root / "train/revisions/probe")
            self.assertIs(result, receipt)
            self.assertEqual(calls, ["construct", "open_account", "learning_source", "make_admission", "authenticate_receipt"])
            receipt["selection"]["fresh_domain_actor_updates"] = 1
            with self.assertRaisesRegex(ValueError, "Initial terrain"):
                d.native_admission(self.args, boundary, self.args.run_root / "train/revisions/probe")

    def test_prepare_then_mocked_launch_preserves_lock_handoff_and_once_only(self):
        self.closed_fixture()
        self.directory.mkdir()
        put(self.directory / "preflight.json", {"status": "CPU_preflight_passed"})
        put(self.args.old_source_root / d.OLD_CONTRACT, {})
        put(self.args.new_source_root / d.CONTRACT, {})
        put(self.args.new_source_root / "contracts/v6_task_campaign_terrain_worker_v27.json", {})
        self.args.runtime_script.write_text("# synthetic runtime; never executed")
        boundary = d.closed_boundary(self.args, self.native)
        admission = {"test": "adapter output, no fake runtime qualification"}
        control_path = self.args.run_root / "train/native_speed_boundary_controller.lock"
        account_path = self.claim.with_suffix(".lock")
        def held(path):
            with path.open("a") as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def admit(*args):
            held(control_path)
            held(account_path)
            return admission
        with patch.object(d, "native_admission", side_effect=admit):
            prepared = d.prepare(self.args, self.directory, self.native, self.claim)
        self.assertEqual(prepared["status"], "boundary_prepared_not_launched")
        self.assertEqual(d.read(self.args.run_root / "source_revision_030.json")["previous_job_sha256"],
                         boundary["job_pin"]["sha256"])
        joint = ModuleType("wheeled_tasks.chassis.integrated_production")
        joint.authenticate_receipt = lambda *args: None
        def submit(command, **kwargs):
            held(control_path)
            with account_path.open("a") as available:
                fcntl.flock(available, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue((self.directory / "launch.intent.json").exists())
            self.assertIn("--resume-existing", command)
            self.assertIn("--source-revision-receipt", command)
            self.assertNotIn("train_chassis.py", command)
            return SimpleNamespace(pid=12345)
        with patch.dict(sys.modules, {"wheeled_tasks.chassis.integrated_production": joint}), \
                patch.object(d.subprocess, "Popen", side_effect=submit) as popen:
            launched = d.launch(self.args, self.directory, self.native, self.claim)
            self.assertEqual(launched["verification_status"], "first_terrain_actor_pending")
            with self.assertRaisesRegex(ValueError, "Prior"):
                d.launch(self.args, self.directory, self.native, self.claim)
            self.assertEqual(popen.call_count, 1)

    def test_matching_outer_archive_sha_cannot_hide_wrong_inner_artifact(self):
        _, checkpoint = self.closed_fixture()
        root = self.args.run_root
        delivery = d.read(root / "delivery.json")
        archive = root / "delivery.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name in delivery["files"]:
                if name == str(checkpoint.relative_to(root)):
                    data = b"wrong archived checkpoint"
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    output.addfile(info, io.BytesIO(data))
                else:
                    output.add(root / name, arcname=name)
        delivery.update(archive_sha256=d.sha(archive), archive_bytes=archive.stat().st_size)
        put(root / "delivery.json", delivery)
        with self.assertRaisesRegex(ValueError, "archive bytes"):
            d.closed_boundary(self.args, self.native)


if __name__ == "__main__":
    unittest.main()
