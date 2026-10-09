"""Small CPU guard/CLI-order tests; no real training or boundary is executed."""
from argparse import Namespace
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source031_once_queue_test", DIRECTORY / "coordinate_source031.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
NATIVE_UTILITY = Path("/home/yukikaze/Documents/workspace/robot_rl/.worktrees/audit-logger-timing-20261009/scripts/queue_task_campaign_continuation.py")
NATIVE_PROCESS_UTILITY = NATIVE_UTILITY.with_name("queue_task_campaign_boundary_continuation.py")
NATIVE_HELPER = DIRECTORY.parent / "source031_deploy/deploy_source031.py"
helper_spec = importlib.util.spec_from_file_location("queue_test_real_deploy_helper", NATIVE_HELPER)
real_helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(real_helper)


class Fixture:
    def __init__(self, stack):
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.old = self.root / "source_serial_dispatch029_20261009"
        self.new = self.root / "source_serial_dispatch031_20261009"
        (self.new / "scripts").mkdir(parents=True)
        self.old.mkdir()
        self.deploy = self.root / "source031_deploy"
        self.deploy.mkdir()
        self.controller = self.new / queue.CONTROLLER
        self.controller.write_text("# synthetic, never executed\n")
        self.utility = self.new / queue.UTILITY
        shutil.copyfile(NATIVE_UTILITY, self.utility)
        self.process_utility = self.new / queue.PROCESS_UTILITY
        shutil.copyfile(NATIVE_PROCESS_UTILITY, self.process_utility)
        self.publish = queue.native_publish(self.utility)
        self.helper = self.root / "deploy_source031_663efe61.py"
        shutil.copyfile(NATIVE_HELPER, self.helper)
        self.archive = self.root / ("source031-" + queue.COMMIT + "-umask0022.tar.gz")
        self.runtime = self.root / "runtime.sh"
        self.runtime.write_bytes(b"#" * 776)
        self.runtime.chmod(0o644)
        inventory = {queue.CONTROLLER: queue.sha(self.controller), queue.UTILITY: queue.sha(self.utility),
            queue.PROCESS_UTILITY: queue.sha(self.process_utility)}
        names = sorted(real_helper.REQUIRED - inventory.keys())
        names.extend("unused/" + str(n) for n in range(4341 - len(inventory) - len(names)))
        for name in names:
            target = self.new / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"synthetic staged dependency")
            target.chmod(0o644)
            inventory[name] = queue.sha(target)
        with tarfile.open(self.archive, "w:gz", format=tarfile.PAX_FORMAT,
                pax_headers={"comment": queue.COMMIT}) as archive:
            for name in sorted(inventory):
                data = (self.new / name).read_bytes()
                member = tarfile.TarInfo(name)
                member.mode, member.size = 0o644, len(data)
                archive.addfile(member, io.BytesIO(data))
        stage = dict(schema="source031-static-stage-v1", status="staged_only", source_commit=queue.COMMIT,
            old_source_root=str(self.old), new_source_root=str(self.new), dynamic_parent_bound=False,
            PPO_updates=0, qualification=None, helper=queue.pin(self.helper),
            source_archive=queue.pin(self.archive), files=inventory)
        self.stage = self.deploy / "stage.json"
        self.publish(self.stage, stage, exclusive=True)
        self.publish(self.deploy / "preflight.log", dict(status="passed", roles=30), exclusive=True)
        self.publish(self.deploy / "preflight.json", dict(status="CPU_preflight_passed",
            stage=queue.pin(self.stage), log=queue.pin(self.deploy / "preflight.log"), PPO_updates=0,
            qualification=None), exclusive=True)
        for name, value in (("ARCHIVE_SHA", queue.sha(self.archive)), ("HELPER_SHA", queue.sha(self.helper)),
                ("RUNTIME_SHA", queue.sha(self.runtime)), ("STAGE_SHA", queue.sha(self.stage)),
                ("STAGE_FILE_COUNT", len(inventory)), ("PREFLIGHT_SHA", queue.sha(self.deploy / "preflight.json"))):
            stack.enter_context(patch.object(queue, name, value))
        self.args = Namespace(run_root=self.root, runtime_script=self.runtime, apply=True)
        self.cancellation_fixture()
        self.calls = []
        self.delivery = {}
        for key in ("delivery_archive", "delivery_manifest", "archiver"):
            target = self.root / (key + ".fixture")
            target.write_bytes(("authenticated original " + key).encode())
            self.delivery[key] = queue.pin(target)

    def cancellation_fixture(self):
        old_queue = self.root / "source030_queue"
        old_queue.mkdir()
        old_intent = old_queue / "queue.intent.json"
        self.publish(old_intent, dict(schema="source030-once-normal-speed-boundary-queue-v1",
            source_commit="81ad124554c4ab45cc274e2d482d46c76ba125b0",
            coordinator=dict(file=str(self.root / "coordinate_source030_0478bfe3.py"))), exclusive=True)
        self.publish(old_queue / "status.json", dict(status="attention_no_retry", phase="controller"), exclusive=True)
        for lock in (self.root / "source030_queue.lock", self.root / "train/native_speed_boundary_controller.lock"):
            lock.parent.mkdir(exist_ok=True)
            lock.touch()
        directory = self.root / "source031_queue"
        directory.mkdir()
        controller = dict(pid=990001, starttime=1, ppid=990002,
            entry=str(self.root / "source_serial_dispatch030_20261009" / queue.CONTROLLER),
            command=[sys.executable, "-B", str(self.root / "source_serial_dispatch030_20261009" / queue.CONTROLLER)])
        coordinator = dict(pid=990002, starttime=2, ppid=1,
            entry=str(self.root / "coordinate_source030_0478bfe3.py"),
            command=[sys.executable, "-B", str(self.root / "coordinate_source030_0478bfe3.py")])
        self.publish(old_queue / "controller.started.json",
            dict(command=controller["command"], pid=controller["pid"]), exclusive=True)
        self.publish(old_queue / "controller.execution.json",
            dict(command=controller["command"], pid=controller["pid"], exit_code=-15), exclusive=True)
        new_pins = dict(stage=queue.pin(self.stage), preflight=queue.pin(self.deploy / "preflight.json"),
            archive=queue.pin(self.archive), helper=queue.pin(self.helper))
        self.cancel_intent = directory / "cancel_source030.intent.json"
        self.publish(self.cancel_intent, dict(schema=queue.CANCEL_SCHEMA, run_root=str(self.root),
            signal="one_PIDFD_SIGTERM_waiting_controller_only",
            snapshot=dict(old_queue_intent=queue.pin(old_intent), old_controller=controller,
                old_coordinator=coordinator, new_source031=new_pins)), exclusive=True)
        self.cancel_receipt = directory / "cancel_source030.receipt.json"
        self.publish(self.cancel_receipt, dict(schema=queue.CANCEL_SCHEMA,
            status="waiting_source030_cancelled_without_training_signal", run_root=str(self.root),
            intent=queue.pin(self.cancel_intent), old_queue_intent=queue.pin(old_intent),
            old_controller=dict(controller, exited=True), old_coordinator=dict(coordinator, exited=True),
            controller_lock_released=True, old_transition_products_absent=True,
            new_source031=new_pins, old_status=queue.pin(old_queue / "status.json"),
            old_execution=queue.pin(old_queue / "controller.execution.json"), waiting_controller_signals=1,
            worker_signals=0, training_signals=0, PPO_updates=0, refunds=0, qualification=None), exclusive=True)
        self.publish(self.root / "job.json", dict(command=[sys.executable,
            str(self.old / "scripts/run_task_campaign.py")]), exclusive=True)

    def execute(self, command, name, directory, publish):
        self.calls.append((name, command))
        (directory / (name + ".log")).write_text("synthetic subprocess evidence\n")
        if name == "controller":
            publish(self.root / "source031_speed_boundary.json", dict(schema="synthetic-controller-intent"), exclusive=True)
            publish(self.root / "source031_speed_boundary.receipt.json", dict(
                status="native_supervisor_stopped_after_normal_speed", worker_signals=0,
                next_domain_dispatched=False, native_completion_recovery=dict(status="closed_native_completed_speed")), exclusive=True)
        elif name == "prepare":
            publish(self.deploy / "prepare.json", dict(status="boundary_prepared_not_launched", boundary=self.delivery), exclusive=True)
        elif name == "launch":
            publish(self.deploy / "launch.receipt.json", dict(status="launch_submitted",
                verification_status="first_terrain_actor_pending", pid=12345), exclusive=True)
        else:
            raise AssertionError(name)

    def native_style_prepare(self, *, lineage_sha=None):
        """Mocked prepare artifacts follow native pin topology; no full-state claim."""
        entry = self.root / "train/revisions/v26_completed_speed_to_v27/entry.json"
        entry.parent.mkdir(parents=True)
        actual_lineage = queue.sha(self.new / "contracts/v6_integrated_lineage_v27.json")
        self.publish(entry, dict(schema="v26-completed-speed-to-v27-entry-v1",
            new_lineage_manifest_sha256=actual_lineage if lineage_sha is None else lineage_sha), exclusive=True)
        revision = self.root / "source_revision_031.json"
        self.publish(revision, dict(source_commit=queue.COMMIT, new_source_root=str(self.new)), exclusive=True)
        admission = self.deploy / "admission.json"
        self.publish(admission, dict(lineage_manifest=dict(sha256=actual_lineage),
            entry_admission=dict(completed_speed_predecessor_entry=queue.pin(entry))), exclusive=True)
        account, state = self.root / "campaign.account.json", self.root / "train/campaign_state.json"
        for path in (account, state):
            self.publish(path, dict(synthetic_paid_account_preserved=True), exclusive=True)
        prepared = dict(status="boundary_prepared_not_launched", boundary=self.delivery,
            source_revision=queue.pin(revision), admission=queue.pin(admission),
            account=queue.pin(account), state=queue.pin(state))
        self.publish(self.deploy / "prepare.json", prepared)
        return prepared


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.f = Fixture(self.stack)

    def test_default_is_read_only(self):
        self.f.args.apply = False
        result = queue.coordinate(self.f.args, execute=lambda *args: self.fail("must not execute"))
        self.assertEqual(result["status"], "validated_once_queue_not_started")
        self.assertFalse((self.f.root / "source031_queue/queue.intent.json").exists())
        self.assertFalse((self.f.root / "source031_queue.lock").exists())

    def test_once_actual_receipt_then_prepare_retention_launch_order(self):
        def execute(command, name, directory, publish):
            if name == "launch":
                self.assertTrue((directory / "old_delivery.json").is_file())
            self.f.execute(command, name, directory, publish)
        result = queue.coordinate(self.f.args, execute=execute)
        self.assertEqual([name for name, _ in self.f.calls], ["controller", "prepare", "launch"])
        controller = self.f.calls[0][1]
        self.assertEqual(controller[controller.index("--timeout") + 1], "604800")
        self.assertEqual(controller[controller.index("--source-root") + 1], str(self.f.old))
        actual = queue.sha(self.f.root / "source031_speed_boundary.receipt.json")
        for _, command in self.f.calls[1:]:
            self.assertEqual(command[command.index("--controller-receipt-sha256") + 1], actual)
        self.assertIsNone(result["qualification"])
        self.assertEqual(result["status"], "launch_submitted_first_terrain_actor_pending")
        self.assertEqual(queue.read(self.f.root / "source031_queue/queue.intent.json")["retries"], 0)
        with self.assertRaisesRegex(ValueError, "requires inspection"):
            queue.coordinate(self.f.args, execute=execute)
        self.assertEqual(len(self.f.calls), 3)

    def test_controller_error_retained_no_prepare_no_retry(self):
        def fail(command, name, directory, publish):
            self.f.calls.append((name, command))
            (directory / "controller.log").write_text("actual error preserved\n")
            raise RuntimeError("synthetic controller error")
        with self.assertRaisesRegex(RuntimeError, "controller error"):
            queue.coordinate(self.f.args, execute=fail)
        self.assertEqual([name for name, _ in self.f.calls], ["controller"])
        directory = self.f.root / "source031_queue"
        self.assertEqual((directory / "controller.log").read_text(), "actual error preserved\n")
        self.assertEqual(queue.read(directory / "status.json")["status"], "attention_no_retry")
        with self.assertRaisesRegex(ValueError, "Once-only"):
            queue.coordinate(self.f.args, execute=fail)

    def test_unsuccessful_controller_receipt_cannot_prepare(self):
        def wrong(command, name, directory, publish):
            self.f.execute(command, name, directory, publish)
            receipt_file = self.f.root / "source031_speed_boundary.receipt.json"
            receipt = queue.read(receipt_file)
            receipt["next_domain_dispatched"] = True
            publish(receipt_file, receipt)
        with self.assertRaisesRegex(ValueError, "successful actual native closure"):
            queue.coordinate(self.f.args, execute=wrong)
        self.assertEqual([name for name, _ in self.f.calls], ["controller"])

    def test_prepare_failure_never_launches(self):
        def fail(command, name, directory, publish):
            if name == "prepare":
                self.f.calls.append((name, command))
                raise RuntimeError("synthetic prepare rejection")
            self.f.execute(command, name, directory, publish)
        with self.assertRaisesRegex(RuntimeError, "prepare rejection"):
            queue.coordinate(self.f.args, execute=fail)
        self.assertEqual([name for name, _ in self.f.calls], ["controller", "prepare"])
        self.assertFalse((self.f.root / "source031_queue/old_delivery.json").exists())

    def test_helper_mutation_during_seven_day_wait_refuses_prepare(self):
        def mutate(command, name, directory, publish):
            self.f.execute(command, name, directory, publish)
            self.f.helper.write_text("changed while controller waits")
        with self.assertRaisesRegex(ValueError, "helper changed"):
            queue.coordinate(self.f.args, execute=mutate)
        self.assertEqual([name for name, _ in self.f.calls], ["controller"])

    def test_runtime_mutation_after_authenticated_prepare_refuses_launch(self):
        def mutate(directory, prepared, publish):
            result = queue.retain_delivery(directory, prepared, publish)
            self.f.runtime.write_bytes(b"!" * 776)
            return result
        with self.assertRaisesRegex(ValueError, "runtime changed"):
            queue.coordinate(self.f.args, execute=self.f.execute, retain=mutate)
        self.assertEqual([name for name, _ in self.f.calls], ["controller", "prepare"])
        self.assertTrue((self.f.root / "source031_queue/old_delivery.json").exists())

    def test_launch_failure_keeps_intent_and_old_delivery(self):
        def fail(command, name, directory, publish):
            if name == "launch":
                self.f.calls.append((name, command))
                publish(self.f.deploy / "launch.intent.json", dict(status="synthetic-submission-intent"), exclusive=True)
                raise RuntimeError("synthetic launch error")
            self.f.execute(command, name, directory, publish)
        with self.assertRaisesRegex(RuntimeError, "launch error"):
            queue.coordinate(self.f.args, execute=fail)
        directory = self.f.root / "source031_queue"
        self.assertTrue((directory / "old_delivery.json").is_file())
        self.assertEqual(queue.read(directory / "status.json")["phase"], "launch")
        self.assertFalse((directory / "queue.receipt.json").exists())

    def test_preexisting_once_intent_not_overwritten(self):
        directory = self.f.root / "source031_queue"
        directory.mkdir(exist_ok=True)
        old_bytes = b'{"retained":true}\n'
        (directory / "queue.intent.json").write_bytes(old_bytes)
        with self.assertRaisesRegex(ValueError, "Once-only"):
            queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual((directory / "queue.intent.json").read_bytes(), old_bytes)
        self.assertEqual(self.f.calls, [])

    def test_actual_child_flock_prevents_different_queue_invocation(self):
        lock = self.f.root / "source031_queue.lock"
        code = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.read()"
        child = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline(), "held\n")
            with self.assertRaises(BlockingIOError):
                queue.coordinate(self.f.args, execute=self.f.execute)
            self.assertEqual(self.f.calls, [])
            self.assertFalse((self.f.root / "source031_queue/queue.intent.json").exists())
        finally:
            child.stdin.close()
            self.assertEqual(child.wait(timeout=5), 0)
            child.stdout.close()

    def test_delivery_paths_replaced_retained_old_inode_survives(self):
        queue.coordinate(self.f.args, execute=self.f.execute)
        retained = queue.read(self.f.root / "source031_queue/old_delivery.json")
        for item in retained.values():
            original = Path(item["original"]["file"])
            target = Path(item["retained"]["file"])
            old_inode = original.stat().st_ino
            self.assertEqual(target.stat().st_ino, old_inode)
            temporary = original.with_suffix(".new")
            temporary.write_bytes(b"next native finalized delivery")
            temporary.replace(original)
            self.assertEqual(queue.sha(target), item["original"]["sha256"])
            self.assertNotEqual(original.stat().st_ino, old_inode)

    def test_sourced_runtime_0644_not_executable_guard(self):
        queue.inputs(self.f.args)
        self.f.runtime.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "Original sourced runtime identity differs"):
            queue.inputs(self.f.args)

    def test_stage_not_zero_ppo_or_unbound_is_refused(self):
        stage = queue.read(self.f.stage)
        stage["dynamic_parent_bound"] = True
        self.f.publish(self.f.stage, stage)
        with patch.object(queue, "STAGE_SHA", queue.sha(self.f.stage)):
            with self.assertRaisesRegex(ValueError, "frozen unbound"):
                queue.inputs(self.f.args)

    def test_noncontroller_dependency_changed_refused_before_any_execute(self):
        dependency = self.f.new / "src/wheeled_tasks/chassis/integrated_completed_speed_revision.py"
        dependency.write_text("changed transitive dependency")
        with self.assertRaisesRegex(ValueError, "Staged source differs before controller execution"):
            queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual(self.f.calls, [])
        self.assertFalse((self.f.root / "source031_queue/queue.intent.json").exists())

    def test_noncontroller_permission_changed_refused_before_any_execute(self):
        dependency = self.f.new / "src/wheeled_tasks/chassis/integrated_completed_speed_revision.py"
        dependency.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "Staged source permission differs"):
            queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual(self.f.calls, [])

    def test_subprocess_keyboard_interrupt_never_kills_waiting_controller(self):
        directory = self.f.root / "source031_queue"
        directory.mkdir(exist_ok=True)
        class Child:
            pid = 123
            def wait(self):
                raise KeyboardInterrupt("synthetic cancellation")
            def kill(self):
                raise AssertionError("must not kill child")
            def terminate(self):
                raise AssertionError("must not terminate child")
        calls = []
        def popen(command, **kwargs):
            calls.append(kwargs)
            self.assertNotIn("timeout", kwargs)
            return Child()
        with self.assertRaises(KeyboardInterrupt):
            queue.run_command(["synthetic-controller"], "controller", directory, self.f.publish, popen=popen)
        self.assertEqual(calls[0]["env"]["OMP_NUM_THREADS"], "1")
        self.assertTrue((directory / "controller.started.json").exists())
        self.assertTrue((directory / "controller.log").exists())
        self.assertFalse((directory / "controller.execution.json").exists())

    def test_unfrozen_stage_or_inventory_count_refuses_queue(self):
        for field in ("ARCHIVE_SHA", "STAGE_SHA", "STAGE_FILE_COUNT", "PREFLIGHT_SHA"):
            with self.subTest(field=field), patch.object(queue, field, None):
                with self.assertRaisesRegex(ValueError, "not yet frozen"):
                    queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual(self.f.calls, [])

    def test_supersedes_receipt_required_before_any_controller(self):
        self.f.cancel_receipt.unlink()
        with self.assertRaisesRegex(ValueError, "existing non-symlink"):
            queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual(self.f.calls, [])

    def test_supersedes_nonzero_signals_PPO_or_refunds_and_unproven_exit_refused(self):
        original = queue.read(self.f.cancel_receipt)
        for field in ("worker_signals", "training_signals", "PPO_updates", "refunds",
                      "controller_lock_released", "old_transition_products_absent"):
            value = dict(original)
            value[field] = False if field.endswith(("released", "absent")) else 1
            self.f.publish(self.f.cancel_receipt, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cancellation receipt"):
                queue.superseded_waiter(self.f.root, self.f.new)
        value = dict(original, old_controller=dict(original["old_controller"], exited=False))
        self.f.publish(self.f.cancel_receipt, value)
        with self.assertRaisesRegex(ValueError, "identity or exit"):
            queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_status_must_be_retired_waiting_phase(self):
        path = self.f.root / "source030_queue/status.json"
        for status in (dict(status="waiting_for_normal_speed_boundary", phase="controller"),
                       dict(status="attention_no_retry", phase="prepare")):
            self.f.publish(path, status)
            with self.assertRaisesRegex(ValueError, "original waiting phase"):
                queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_receipt_cannot_hide_current_old_transition_or_claim(self):
        for name in ("source_revision_030.json", "source030_speed_boundary.json",
                     "source030_speed_boundary.receipt.json", "source030_deploy/prepare.intent.json",
                     "source030_deploy/admission.json", "source030_deploy/launch.intent.json",
                     "source030_queue/prepare.started.json", "source030_queue/queue.receipt.json",
                     "train/revisions/source030_deploy_admission"):
            path = self.f.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("old transition evidence")
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "cannot supersede"):
                    queue.superseded_waiter(self.f.root, self.f.new)
            finally:
                path.unlink()

    def test_supersedes_current_job_cannot_already_use_old_source030(self):
        self.f.publish(self.f.root / "job.json", dict(command=[sys.executable,
            str(self.f.root / "source_serial_dispatch030_20261009/scripts/run_task_campaign.py")]))
        with self.assertRaisesRegex(ValueError, "already adopted"):
            queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_same_process_starttime_still_live_refused_and_PID_reuse_allowed(self):
        receipt = queue.read(self.f.cancel_receipt)
        reader, actual_entry = queue.native_process_reader(self.f.new)
        identity = receipt["old_controller"]
        def current(pid):
            if pid == identity["pid"]:
                return {k: v for k, v in identity.items() if k != "exited"}
            return reader(pid)
        with patch.object(queue, "native_process_reader", return_value=(current, actual_entry)):
            with self.assertRaisesRegex(ValueError, "still alive"):
                queue.superseded_waiter(self.f.root, self.f.new)
        def reused(pid):
            if pid == identity["pid"]:
                return dict(identity, starttime=identity["starttime"] + 1)
            return reader(pid)
        with patch.object(queue, "native_process_reader", return_value=(reused, actual_entry)):
            queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_actual_controller_or_old_queue_flock_refused(self):
        import fcntl
        for name in ("train/native_speed_boundary_controller.lock", "source030_queue.lock"):
            with (self.f.root / name).open("r") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.subTest(name=name), self.assertRaises(BlockingIOError):
                    queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_new_native_entry_in_old_source_tree_refused(self):
        reader, actual_entry = queue.native_process_reader(self.f.new)
        def current(pid):
            if pid == os.getpid():
                return dict(pid=pid, starttime=1, entry=str(self.f.root /
                    "source_serial_dispatch030_20261009/scripts/train_chassis.py"))
            return reader(pid)
        with patch.object(queue, "native_process_reader", return_value=(current, actual_entry)):
            with self.assertRaisesRegex(ValueError, "native entry"):
                queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_late_old_entry_or_admission_claim_refuses_prepare(self):
        for name in ("source030_speed_boundary.json", "train/revisions/source030_deploy_admission"):
            with self.subTest(name=name):
                admitted = queue.inputs(self.f.args)
                path = self.f.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("late old adoption evidence")
                try:
                    with self.assertRaisesRegex(ValueError, "cannot supersede"):
                        queue.unchanged(admitted)
                finally:
                    path.unlink()

    def test_supersedes_generic_native_entry_or_paid_claim_refused_before_own_prepare(self):
        path = self.f.root / "train/revisions/v26_completed_speed_to_v27"
        path.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "earlier native v27 entry or claim"):
            queue.superseded_waiter(self.f.root, self.f.new)
        self.f.publish(path / "entry.json", dict(new_lineage_manifest_sha256="f" * 64), exclusive=True)
        self.f.publish(self.f.root / "source_revision_031.json",
            dict(source_commit=queue.COMMIT, new_source_root=str(self.f.new)), exclusive=True)
        with self.assertRaisesRegex(ValueError, "earlier native v27 entry or claim"):
            queue.superseded_waiter(self.f.root, self.f.new)
        self.f.publish(path / "entry.json", dict(new_lineage_manifest_sha256=
            queue.sha(self.f.new / "contracts/v6_integrated_lineage_v27.json")))
        with self.assertRaisesRegex(ValueError, "earlier native v27 entry or claim"):
            queue.superseded_waiter(self.f.root, self.f.new)

    def test_prepare_new031_native_entry_pin_allows_launch(self):
        def execute(command, name, directory, publish):
            self.f.execute(command, name, directory, publish)
            if name == "prepare":
                self.f.native_style_prepare()
        result = queue.coordinate(self.f.args, execute=execute)
        self.assertEqual(result["status"], "launch_submitted_first_terrain_actor_pending")
        self.assertEqual([name for name, _ in self.f.calls], ["controller", "prepare", "launch"])

    def test_prepare_old030_native_entry_pin_refuses_launch(self):
        def execute(command, name, directory, publish):
            self.f.execute(command, name, directory, publish)
            if name == "prepare":
                self.f.native_style_prepare(lineage_sha="f" * 64)
        with self.assertRaisesRegex(ValueError, "earlier native v27 entry or claim"):
            queue.coordinate(self.f.args, execute=execute)
        self.assertEqual([name for name, _ in self.f.calls], ["controller", "prepare"])
        self.assertFalse((self.f.deploy / "launch.receipt.json").exists())

    def test_supersedes_exit_record_or_staged_candidate_binding_cannot_be_forged(self):
        receipt = queue.read(self.f.cancel_receipt)
        value = dict(receipt, new_source031=dict(receipt["new_source031"], stage=dict(file="foreign", sha256="0" * 64)))
        self.f.publish(self.f.cancel_receipt, value)
        with self.assertRaisesRegex(ValueError, "bound to the staged Source031"):
            queue.superseded_waiter(self.f.root, self.f.new)
        self.f.publish(self.f.cancel_receipt, receipt)
        path = self.f.root / "source030_queue/controller.started.json"
        self.f.publish(path, dict(command=receipt["old_controller"]["command"], pid=receipt["old_controller"]["pid"] + 1))
        with self.assertRaisesRegex(ValueError, "original observer start/exit records"):
            queue.superseded_waiter(self.f.root, self.f.new)

    def test_supersedes_receipt_bytes_rechecked_after_wait(self):
        admitted = queue.inputs(self.f.args)
        receipt = queue.read(self.f.cancel_receipt)
        self.f.publish(self.f.cancel_receipt, dict(receipt, unexpected_new_note="changed"))
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            queue.unchanged(admitted)

    def test_actual_cancel_module_result_then_supersedes_cross_module(self):
        """Consume actual cancel() output; only its native/PIDFD services are fake."""
        import pytest
        cancellation_file = Path(__file__).with_name("test_cancel_waiting_source030.py")
        spec = importlib.util.spec_from_file_location("source031_actual_cancel_fixture", cancellation_file)
        cancellation_tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cancellation_tests)
        monkeypatch = pytest.MonkeyPatch()
        self.stack.callback(monkeypatch.undo)
        temporary = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        cancellation = cancellation_tests.Fixture(temporary, monkeypatch)
        self.stack.callback(cancellation.close)
        result = cancellation.execute_apply()
        self.assertEqual(result["status"], "waiting_source030_cancelled_without_training_signal")
        (cancellation.root / "source030_queue.lock").touch()
        cancellation_tests.write(cancellation.root / "job.json",
            dict(command=cancellation.training["supervisor"]["command"]))
        _, actual_entry = queue.native_process_reader(self.f.new)
        with patch.object(queue, "native_process_reader",
                return_value=(cancellation.boundary.process_identity, actual_entry)):
            accepted = queue.superseded_waiter(cancellation.root, cancellation.next)
            self.assertEqual(accepted["receipt"], queue.pin(cancellation.root /
                "source031_queue/cancel_source030.receipt.json"))
            cancellation_tests.write(cancellation.root / "source030_speed_boundary.json",
                dict(late_old_boundary=True))
            with self.assertRaisesRegex(ValueError, "cannot supersede"):
                queue.superseded_waiter(cancellation.root, cancellation.next, expected=accepted)


if __name__ == "__main__":
    unittest.main()
