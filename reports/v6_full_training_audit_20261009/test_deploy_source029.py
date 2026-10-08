"""CPU-only guards for the independent source029 deployment helper."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

path = Path(__file__).with_name("deploy_source029.py")
spec = importlib.util.spec_from_file_location("source029_deploy_test_target", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
COMMIT = "a" * 40
REQUIRED = [str(module.CONTRACT), "scripts/run_task_campaign.py", "scripts/chassis_remote_job.py",
            "scripts/repair_known_integrated_origin.py"]


class DeployGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "source.tar.gz"
        self.destination = self.root / "source029"

    def tearDown(self):
        self.temp.cleanup()

    def pack(self, *, comment=COMMIT, extras=(), include=True):
        with tarfile.open(self.archive, "w:gz", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": comment}) as output:
            if include:
                for name in REQUIRED:
                    content = (name + "\n").encode()
                    row = tarfile.TarInfo(name)
                    row.size = len(content)
                    row.mode = 0o644
                    output.addfile(row, io.BytesIO(content))
            for row, content in extras:
                output.addfile(row, io.BytesIO(content) if content is not None else None)

    def rejected(self, **kwargs):
        self.pack(**kwargs)
        with self.assertRaises(ValueError):
            module.extract_new(self.archive, self.destination, COMMIT)
        self.assertFalse(self.destination.exists())

    def test_safe_archive_preserves_every_byte(self):
        self.pack()
        self.assertEqual(module.extract_new(self.archive, self.destination, COMMIT), 4)
        for name in REQUIRED:
            self.assertEqual((self.destination / name).read_bytes(), (name + "\n").encode())

    def test_commit_mismatch_rejected_before_source_publication(self):
        self.rejected(comment="b" * 40)

    def test_traversal_rejected(self):
        row = tarfile.TarInfo("../foreign")
        row.size = 1
        self.rejected(extras=[(row, b"x")])
        self.assertFalse((self.root / "foreign").exists())

    def test_absolute_path_rejected(self):
        row = tarfile.TarInfo(str(self.root / "foreign"))
        row.size = 1
        self.rejected(extras=[(row, b"x")])
        self.assertFalse((self.root / "foreign").exists())

    def test_symlink_rejected(self):
        row = tarfile.TarInfo("scripts/link")
        row.type, row.linkname = tarfile.SYMTYPE, "/tmp/foreign"
        self.rejected(extras=[(row, None)])

    def test_hardlink_rejected(self):
        row = tarfile.TarInfo("scripts/link")
        row.type, row.linkname = tarfile.LNKTYPE, REQUIRED[0]
        self.rejected(extras=[(row, None)])

    def test_duplicate_regular_file_rejected(self):
        row = tarfile.TarInfo(REQUIRED[0])
        row.size = 1
        self.rejected(extras=[(row, b"x")])

    def test_required_transaction_script_must_exist(self):
        row = tarfile.TarInfo("README.md")
        row.size = 1
        self.rejected(include=False, extras=[(row, b"x")])

    def test_existing_frozen_source_never_overwritten(self):
        self.destination.mkdir()
        marker = self.destination / "original"
        marker.write_bytes(b"keep")
        self.pack()
        with self.assertRaises(ValueError):
            module.extract_new(self.archive, self.destination, COMMIT)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_durable_intent_never_overwrites_prior_submission(self):
        intent = self.root / "launch.intent.json"
        module.durable(intent, {"pid": 123, "qualification": None})
        raw = intent.read_bytes()
        with self.assertRaises(FileExistsError):
            module.durable(intent, {"pid": 456})
        self.assertEqual(intent.read_bytes(), raw)
        self.assertEqual(json.loads(raw)["pid"], 123)

    def test_stage_helper_hash_drift_rejected_before_owner_or_source_import(self):
        directory = self.root / "source029_deploy"
        directory.mkdir()
        args = SimpleNamespace(run_root=self.root, source_commit=COMMIT,
                               source_archive=self.archive, new_source_root=self.destination)
        module.durable(self.root / "source_revision_029.json", dict(source_commit=COMMIT,
            source_archive=str(self.archive), new_source_root=str(self.destination)))
        module.durable(directory / "stage.json", dict(source_commit=COMMIT, helper_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "published source029 identity"):
            module.validate_staging(args, directory)


if __name__ == "__main__":
    unittest.main()
