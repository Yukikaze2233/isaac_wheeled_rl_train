"""Round4 scratch identity; reuse transport and artifact primitives only."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(REPO / "scripts/round3")]
from round3.common import CODE_DIRECTORY, ControlSSH, add_ssh_arguments, file_record
from wheeled_algo.v40_job import JobError, atomic_bytes, json_bytes, strict_json

CONTRACT = "contracts/own_v40_round4_full.json"
EXPERIMENTS = Path("/home/kaiser/robot-rl-sim60/experiments")
SOURCE_OPTIONS = {"--resume", "--finetune", "--warm-start", "--stage-transfer", "--source-checkpoint-sha256"}
AUTHORIZATION = "docs/evidence/round4_serial_authorization_20260915.json"
LEGACY_ASSET_SHA = "df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886"
PHYSICS_MODEL = "legacy_seven_body_equivalent_serial_research"


def legacy_identity(contract, read_bytes):
    """Bind the new authorization and legacy physics files, never a repaired-asset receipt."""
    authorization_raw = read_bytes(AUTHORIZATION)
    authorization = strict_json(authorization_raw)
    if (authorization.get("physics_model") != PHYSICS_MODEL
            or authorization.get("explicit_user_authorization") is not True
            or authorization.get("repaired_dynamics_used") is not False
            or authorization.get("physics_asset_manifest_sha256") != LEGACY_ASSET_SHA
            or authorization.get("initial_not_before") != "2026-09-15T00:00:00+08:00"
            or not authorization.get("user_authorization_text")):
        raise JobError("missing or inconsistent explicit legacy-serial authorization")
    asset = contract["asset"]
    if (asset.get("urdf"), asset.get("mjcf"), asset.get("manifest")) != ("robot.urdf", "inspection.xml", "research_manifest.json"):
        raise JobError("legacy physics must use the audited URDF/MJCF selectors")
    directory = Path(asset["directory"])
    manifest_path = directory / asset["manifest"]
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        raise JobError("physics asset must remain inside the committed source tree")
    raw = read_bytes(str(manifest_path))
    if hashlib.sha256(raw).hexdigest() != LEGACY_ASSET_SHA:
        raise JobError("physics asset is not the authorized legacy research manifest")
    manifest = strict_json(raw)
    for name, digest in manifest["files_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise JobError("legacy asset member escapes its directory")
        if hashlib.sha256(read_bytes(str(directory / relative))).hexdigest() != digest:
            raise JobError("legacy physics asset file changed: " + name)
    return {"physics_model": PHYSICS_MODEL, "physics_asset_manifest_sha256": LEGACY_ASSET_SHA,
            "repaired_dynamics_used": False, "explicit_user_authorization": True,
            "new_fifteen_body_model_role": "preview_only_not_training_physics",
            "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
            "initial_not_before": authorization["initial_not_before"]}


def verify_snapshot(root):
    root = Path(root).absolute()
    snapshot = strict_json((root / "snapshot.json").read_bytes())
    if (snapshot.get("initialization") != "scratch" or snapshot.get("parent") is not None
            or snapshot.get("contract") != CONTRACT or snapshot.get("code_directory_name") != CODE_DIRECTORY):
        raise JobError("not a Round4 scratch snapshot")
    for name, expected in snapshot["files"].items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2
                or relative.parts[0] != CODE_DIRECTORY or not path.resolve().is_relative_to(root.resolve())):
            raise JobError("snapshot member escapes code root")
        if file_record(path) != expected:
            raise JobError("snapshot member changed: " + name)
    repo = root / CODE_DIRECTORY
    identity = legacy_identity(strict_json((repo / CONTRACT).read_bytes()), lambda name: (repo / name).read_bytes())
    if any(snapshot.get(key) != value for key, value in identity.items()):
        raise JobError("snapshot legacy physics identity differs from its authorization")
    return snapshot


def verify_scratch(manifest, plan):
    if manifest.get("asset_manifest_sha256") != LEGACY_ASSET_SHA:
        raise JobError("actual training physics asset is not the authorized legacy model")
    if any(manifest.get(key) != value for key, value in plan["identity"].items()):
        raise JobError("Round4 run contract/stage/seed differs from plan")
    if manifest.get("num_envs") != 1024:
        raise JobError("Round4 manifest must record actual num_envs=1024")
    provenance = manifest.get("source_provenance", {})
    if (manifest.get("training_profile") != "round4_full" or provenance.get("mode") != "scratch"
            or provenance != plan["expected_initialization"] or manifest.get("initialization") != provenance
            or manifest.get("resume_provenance") is not None):
        raise JobError("Round4 manifest must explicitly record scratch source_provenance")
    if any(provenance.get(key) is not None for key in
           ("parent_checkpoint_sha256", "parent_contract_sha256", "source_filename", "parent")):
        raise JobError("scratch run unexpectedly records a parent")
    if manifest.get("domain_randomization_report", {}).get("passed") is not True:
        raise JobError("Round4 startup material physical readback did not pass")
    curriculum = manifest.get("training_curriculum", {})
    if (curriculum.get("timebase") != "successful_ppo_updates" or not isinstance(curriculum.get("state"), dict)
            or type(curriculum.get("completed_updates")) is not int or curriculum["completed_updates"] < 0):
        raise JobError("Round4 requires actual successful-update curriculum state")
    if any(arg.split("=", 1)[0] in SOURCE_OPTIONS for arg in plan["command"]):
        raise JobError("scratch command contains a checkpoint-loading option")


def resume_parent_updates(plan):
    """One paused scratch run followed by its remaining updates, not stage transfer."""
    from wheeled_algo.v40_job import validate_completion
    if plan.get("initialization") != "resume":
        if plan.get("initialization") != "scratch" or plan.get("parent") is not None or plan.get("updates") != 30000:
            raise JobError("not a scratch or explicitly bound resume plan")
        return 0
    parent = plan["parent"]
    receipt = validate_completion(parent["completion"])
    prior = parent["completed_updates"]
    if (receipt["status"] != "stopped" or receipt["export_status"] != "verified"
            or receipt["requested_iterations"] != 30000 or type(prior) is not int
            or not 0 < prior < 30000 or receipt["completed_updates"] != prior
            or plan.get("updates") != 30000-prior
            or parent["contract_sha256"] != plan["contract_sha256"]
            or hashlib.sha256(json_bytes(receipt)).hexdigest() != parent["completion_sha256"]):
        raise JobError("resume parent/count/receipt identity mismatch")
    model = next(item for item in receipt["artifacts"] if item["path"] == "model_final.pt")
    if model["sha256"] != parent["checkpoint_sha256"] or Path(parent["checkpoint"]).name != "model_final.pt":
        raise JobError("resume checkpoint does not match the parent completion")
    return prior


def verify_training_manifest(manifest, plan):
    if plan.get("initialization") != "resume":
        return verify_scratch(manifest, plan)
    prior = resume_parent_updates(plan)
    expected = {"checkpoint_sha256": plan["parent"]["checkpoint_sha256"], "completed_updates": prior,
                "semantics": "optimizer_and_curriculum_progress_not_bitwise_trajectory"}
    if manifest.get("resume_provenance") != expected:
        raise JobError("run does not attest the planned optimizer/curriculum resume")
    argv = list(plan["command"])
    if argv.count("--resume") != 1:
        raise JobError("resume plan requires exactly one explicit --resume")
    index = argv.index("--resume")
    if argv[index+1] != plan["parent"]["checkpoint"]:
        raise JobError("resume command points to another parent")
    del argv[index:index+2]
    original = dict(manifest)
    del original["resume_provenance"]
    # Keep the original scratch gate intact, including rejection of other source modes.
    verify_scratch(original, {**plan, "command": argv})


def training_progress(plan, completion, manifest):
    """The child completion remains invocation-local; cumulative progress is separate."""
    from wheeled_algo.v40_job import validate_completion
    validate_completion(completion)
    prior = resume_parent_updates(plan)
    verify_training_manifest(manifest, plan)
    cumulative = prior + completion["completed_updates"]
    if (completion["requested_iterations"] != plan["updates"] or cumulative > 30000
            or manifest["training_curriculum"]["completed_updates"] != cumulative):
        raise JobError("invocation completion and cumulative curriculum clock disagree")
    return {"prior_completed_updates": prior, "invocation_completed_updates": completion["completed_updates"],
            "cumulative_completed_updates": cumulative,
            "formal_training_target_completed": (cumulative == 30000 and completion["status"] == "completed"
                                                   and completion["export_status"] == "verified")}


def verify_training_run(plan):
    from start_v40_round2 import verify_run
    run = Path(plan["run_dir"])
    checks = verify_run(plan, {"name": "train", "requested_iterations": plan["updates"]})
    progress = training_progress(plan, strict_json((run / "completion.json").read_bytes()),
                                 strict_json((run / "run_manifest.json").read_bytes()))
    return {**checks, **progress}


def system_snapshot(processes=False):
    """Read-only counters and concurrent process inventory; never signal other jobs."""
    result = {}
    for name, argv in (
        ("memory", ["free", "-b"]),
        ("gpu", ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                 "--format=csv"]),
        *(([("processes", ["ps", "-eo", "pid,ppid,etime,rss,args"])]) if processes else []),
    ):
        try:
            reply = subprocess.run(argv, capture_output=True, text=True, timeout=5)
            result[name] = {"exit_code": reply.returncode, "stdout": reply.stdout, "stderr": reply.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


def write_json(path, value):
    """Replace audit state atomically; immutable artifacts use atomic_bytes directly."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)
