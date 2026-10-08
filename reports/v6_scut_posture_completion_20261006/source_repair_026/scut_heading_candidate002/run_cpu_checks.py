"""Run only this candidate's CPU tests and retain immutable input identity."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
label = sys.argv[1] if len(sys.argv) == 2 else "cpu001"
if not re.fullmatch(r"cpu\d{3}", label):
    raise SystemExit("CPU receipt label must be cpu followed by three digits")
receipt_path = HERE / (label + ".receipt.json")
if receipt_path.exists():
    raise SystemExit("Refuse to overwrite prior CPU receipt")
manifest = json.loads((HERE / "inputs001.json").read_text())
paths = [HERE / name for name in
         ("heading_candidate.py", "test_heading_candidate.py", "inputs001.json", "README.md", "run_cpu_checks.py")]
paths.extend(Path(ref["file"]) for ref in manifest["inputs"].values())


def identity():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


before = identity()
command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_heading_candidate.py"]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE="1")
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock = time.monotonic()
result = subprocess.run(command, cwd=HERE, env=environment, text=True, capture_output=True)
elapsed = time.monotonic() - clock
(HERE / (label + ".stdout")).write_text(result.stdout)
(HERE / (label + ".stderr")).write_text(result.stderr)
after = identity()
counts = re.search(r"(\d+) passed", result.stdout)
receipt = {"schema": "scut-heading-cpu-receipt-v1", "started_utc": started,
    "command": command, "cwd": str(HERE), "exit_code": result.returncode,
    "wall_seconds": elapsed, "passed": int(counts.group(1)) if counts else None,
    "before_sha256": before, "after_sha256": after, "inputs_unchanged": before == after,
    "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
    "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    "native_started": False, "model_loaded": False, "GPU_started": False,
    "remote_actions": False, "PPO_updates": 0, "production_wiring": False,
    "runtime_ready": False, "qualification": None}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": str(receipt_path), "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                  "exit_code": result.returncode, "wall_seconds": elapsed,
                  "passed": receipt["passed"], "inputs_unchanged": before == after}, indent=2))
sys.exit(result.returncode if before == after else 2)
