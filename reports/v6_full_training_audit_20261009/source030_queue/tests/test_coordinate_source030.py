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
spec = importlib.util.spec_from_file_location("source030_once_queue_test", DIRECTORY / "coordinate_source030.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
NATIVE_UTILITY = Path("/home/yukikaze/Documents/workspace/robot_rl/.worktrees/audit-logger-timing-20261009/scripts/queue_task_campaign_continuation.py")
NATIVE_HELPER = DIRECTORY.parent / "source030_deploy/deploy_source030.py"
helper_spec = importlib.util.spec_from_file_location("queue_test_real_deploy_helper", NATIVE_HELPER)
real_helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(real_helper)


class Fixture:
    def __init__(self, stack):
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.old = self.root / "source_serial_dispatch029_20261009"
        self.new = self.root / "source_serial_dispatch030_20261009"
        (self.new / "scripts").mkdir(parents=True)
        self.old.mkdir()
        self.deploy = self.root / "source030_deploy"
        self.deploy.mkdir()
        self.controller = self.new / queue.CONTROLLER
        self.controller.write_text("# synthetic, never executed\n")
        self.utility = self.new / queue.UTILITY
        shutil.copyfile(NATIVE_UTILITY, self.utility)
        self.publish = queue.native_publish(self.utility)
        self.helper = self.root / "deploy_source030_612c66e8.py"
        shutil.copyfile(NATIVE_HELPER, self.helper)
        self.archive = self.root / ("source030-" + queue.COMMIT + "-umask0022.tar.gz")
        self.runtime = self.root / "runtime.sh"
        self.runtime.write_bytes(b"#" * 776)
        self.runtime.chmod(0o644)
        inventory = {queue.CONTROLLER: queue.sha(self.controller), queue.UTILITY: queue.sha(self.utility)}
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
        stage = dict(schema="source030-static-stage-v1", status="staged_only", source_commit=queue.COMMIT,
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
                ("RUNTIME_SHA", queue.sha(self.runtime)), ("STAGE_SHA", queue.sha(self.stage))):
            stack.enter_context(patch.object(queue, name, value))
        self.args = Namespace(run_root=self.root, runtime_script=self.runtime, apply=True)
        self.calls = []
        self.delivery = {}
        for key in ("delivery_archive", "delivery_manifest", "archiver"):
            target = self.root / (key + ".fixture")
            target.write_bytes(("authenticated original " + key).encode())
            self.delivery[key] = queue.pin(target)

    def execute(self, command, name, directory, publish):
        self.calls.append((name, command))
        (directory / (name + ".log")).write_text("synthetic subprocess evidence\n")
        if name == "controller":
            publish(self.root / "source030_speed_boundary.json", dict(schema="synthetic-controller-intent"), exclusive=True)
            publish(self.root / "source030_speed_boundary.receipt.json", dict(
                status="native_supervisor_stopped_after_normal_speed", worker_signals=0,
                next_domain_dispatched=False, native_completion_recovery=dict(status="closed_native_completed_speed")), exclusive=True)
        elif name == "prepare":
            publish(self.deploy / "prepare.json", dict(status="boundary_prepared_not_launched", boundary=self.delivery), exclusive=True)
        elif name == "launch":
            publish(self.deploy / "launch.receipt.json", dict(status="launch_submitted",
                verification_status="first_terrain_actor_pending", pid=12345), exclusive=True)
        else:
            raise AssertionError(name)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.f = Fixture(self.stack)

    def test_default_is_read_only(self):
        self.f.args.apply = False
        result = queue.coordinate(self.f.args, execute=lambda *args: self.fail("must not execute"))
        self.assertEqual(result["status"], "validated_once_queue_not_started")
        self.assertFalse((self.f.root / "source030_queue").exists())
        self.assertFalse((self.f.root / "source030_queue.lock").exists())

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
        actual = queue.sha(self.f.root / "source030_speed_boundary.receipt.json")
        for _, command in self.f.calls[1:]:
            self.assertEqual(command[command.index("--controller-receipt-sha256") + 1], actual)
        self.assertIsNone(result["qualification"])
        self.assertEqual(result["status"], "launch_submitted_first_terrain_actor_pending")
        self.assertEqual(queue.read(self.f.root / "source030_queue/queue.intent.json")["retries"], 0)
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
        directory = self.f.root / "source030_queue"
        self.assertEqual((directory / "controller.log").read_text(), "actual error preserved\n")
        self.assertEqual(queue.read(directory / "status.json")["status"], "attention_no_retry")
        with self.assertRaisesRegex(ValueError, "Once-only"):
            queue.coordinate(self.f.args, execute=fail)

    def test_unsuccessful_controller_receipt_cannot_prepare(self):
        def wrong(command, name, directory, publish):
            self.f.execute(command, name, directory, publish)
            receipt_file = self.f.root / "source030_speed_boundary.receipt.json"
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
        self.assertFalse((self.f.root / "source030_queue/old_delivery.json").exists())

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
        self.assertTrue((self.f.root / "source030_queue/old_delivery.json").exists())

    def test_launch_failure_keeps_intent_and_old_delivery(self):
        def fail(command, name, directory, publish):
            if name == "launch":
                self.f.calls.append((name, command))
                publish(self.f.deploy / "launch.intent.json", dict(status="synthetic-submission-intent"), exclusive=True)
                raise RuntimeError("synthetic launch error")
            self.f.execute(command, name, directory, publish)
        with self.assertRaisesRegex(RuntimeError, "launch error"):
            queue.coordinate(self.f.args, execute=fail)
        directory = self.f.root / "source030_queue"
        self.assertTrue((directory / "old_delivery.json").is_file())
        self.assertEqual(queue.read(directory / "status.json")["phase"], "launch")
        self.assertFalse((directory / "queue.receipt.json").exists())

    def test_preexisting_once_intent_not_overwritten(self):
        directory = self.f.root / "source030_queue"
        directory.mkdir()
        old_bytes = b'{"retained":true}\n'
        (directory / "queue.intent.json").write_bytes(old_bytes)
        with self.assertRaisesRegex(ValueError, "Once-only"):
            queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual((directory / "queue.intent.json").read_bytes(), old_bytes)
        self.assertEqual(self.f.calls, [])

    def test_actual_child_flock_prevents_different_queue_invocation(self):
        lock = self.f.root / "source030_queue.lock"
        code = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.read()"
        child = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline(), "held\n")
            with self.assertRaises(BlockingIOError):
                queue.coordinate(self.f.args, execute=self.f.execute)
            self.assertEqual(self.f.calls, [])
            self.assertFalse((self.f.root / "source030_queue").exists())
        finally:
            child.stdin.close()
            self.assertEqual(child.wait(timeout=5), 0)
            child.stdout.close()

    def test_delivery_paths_replaced_retained_old_inode_survives(self):
        queue.coordinate(self.f.args, execute=self.f.execute)
        retained = queue.read(self.f.root / "source030_queue/old_delivery.json")
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
        self.assertFalse((self.f.root / "source030_queue").exists())

    def test_noncontroller_permission_changed_refused_before_any_execute(self):
        dependency = self.f.new / "src/wheeled_tasks/chassis/integrated_completed_speed_revision.py"
        dependency.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "Staged source permission differs"):
            queue.coordinate(self.f.args, execute=self.f.execute)
        self.assertEqual(self.f.calls, [])

    def test_subprocess_keyboard_interrupt_never_kills_waiting_controller(self):
        directory = self.f.root / "source030_queue"
        directory.mkdir()
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


if __name__ == "__main__":
    unittest.main()
