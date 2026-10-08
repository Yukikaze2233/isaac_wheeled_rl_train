"""Download one existing immutable source023 seal; never start a simulator."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import time

BASE = Path(__file__).resolve().parent
OUT = BASE / "hold_joint_current_candidate002" / "immutable_update39000"
ROOT = "/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358"
REMOTE = ROOT + "/train/recovery/attempt_009/train/checkpoints/update_00039000"
SOURCE = "6f07ec298e01d2686216787cfba34b394151f1aa"
PINS = {
    "src/wheeled_tasks/chassis/env.py": "380887f8951a2de511264e828f262d362de67c2b77b909d042163eb69e8b7525",
    "src/wheeled_tasks/chassis/recovery.py": "04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6",
    "src/wheeled_tasks/chassis/recovery_training.py": "b475b0166e4cfc0c9a079c06c65bdd20fb744de30d00b92f0965febd5e3d24c5",
    "contracts/v6_task_campaign_recovery_worker_v21.json": "eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2",
}
FILES = ("completion.json", "model.pt", "model.contract.json", "model.checkpoint.json", "progress.json", "contract.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    live_path = BASE / "live020.json"
    live = json.loads(live_path.read_text())
    assert live["progress"]["successful_updates"] + 33620 >= 39000
    assert {row["kind"]: (row["pid"], row["starttime_ticks"]) for row in live["owners"]} == {
        "wrapper": (3662137, 96629148), "supervisor": (3662142, 96629351), "worker": (3662149, 96629601)}
    assert live["completion"] is None
    OUT.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    remote_code = '''from pathlib import Path
import datetime,hashlib,json
root=Path(ROOT); folder=Path(REMOTE)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={name:digest(root/'source_6f07ec298e01'/name) for name in PINS}
assert pins==PINS
files={name:dict(sha256=digest(folder/name),bytes=(folder/name).stat().st_size) for name in FILES}
seal=json.loads((folder/'model.checkpoint.json').read_text())
assert seal['sealed'] is True and seal['learning_updates']==39000
assert seal['checkpoint_sha256']==files['model.pt']['sha256']
assert seal['contract_sha256']==files['model.contract.json']['sha256']==PINS['contracts/v6_task_campaign_recovery_worker_v21.json']
assert seal['task_actor_update']==11795 and seal['learning_lineage_updates']==25253
assert seal['stage_actor_update']==25163 and seal['actor_updates_consumed']==38830
print(json.dumps(dict(read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),folder=str(folder),source_commit=SOURCE,source_pins=pins,files=files,seal=seal,completion=json.loads((folder/'completion.json').read_text()),progress=json.loads((folder/'progress.json').read_text()))))
'''
    remote_code = "ROOT=" + repr(ROOT) + "\nREMOTE=" + repr(REMOTE) + "\nSOURCE=" + repr(SOURCE) + "\nPINS=" + repr(PINS) + "\nFILES=" + repr(FILES) + "\n" + remote_code
    command = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "kaiser@192.168.64.234",
               "exec /home/kaiser/robot-rl-sim60/env/bin/python -B -c " + shlex.quote(remote_code)]
    tick = time.monotonic()
    selected = subprocess.run(command, capture_output=True, timeout=40)
    (OUT / "selection.stdout").write_bytes(selected.stdout)
    (OUT / "selection.stderr").write_bytes(selected.stderr)
    write("selection.execution.json", dict(started_at_utc=started,exit_code=selected.returncode,wall_seconds=time.monotonic()-tick,
        stdout_sha256=sha(OUT/"selection.stdout"),stderr_sha256=sha(OUT/"selection.stderr"),remote_code_sha256=hashlib.sha256(remote_code.encode()).hexdigest()))
    selected.check_returncode()
    evidence = json.loads(selected.stdout)
    write("selection.json", evidence)
    command = ["scp", "-P", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"] + [
        "kaiser@192.168.64.234:" + REMOTE + "/" + name for name in FILES] + [str(OUT)]
    tick = time.monotonic()
    download = subprocess.run(command, capture_output=True, timeout=60)
    (OUT / "download.stdout").write_bytes(download.stdout)
    (OUT / "download.stderr").write_bytes(download.stderr)
    write("download.execution.json", dict(exit_code=download.returncode,wall_seconds=time.monotonic()-tick,
        stdout_sha256=sha(OUT/"download.stdout"),stderr_sha256=sha(OUT/"download.stderr"),command=command))
    download.check_returncode()
    for name, record in evidence["files"].items():
        assert sha(OUT/name) == record["sha256"] and (OUT/name).stat().st_size == record["bytes"]
    import torch
    saved = torch.load(OUT/"model.pt", map_location="cpu", weights_only=True)
    model = saved["model_state_dict"]
    actor = [key for key in model if key.startswith("actor.") or key == "std"]
    critic = [key for key in model if key.startswith("critic.")]
    assert len(actor) == 9 and len(critic) == 8 and model["std"].shape == (6,)
    optimizer = saved["optimizer_state_dict"]
    assert len(optimizer["state"]) == 17
    tensors = []
    def visit(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and torch.isfinite(value).all()
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(model); visit(optimizer)
    infos = saved["infos"]
    assert infos["contract_sha256"] == evidence["seal"]["contract_sha256"]
    for key in ("stage_actor_update", "task_actor_update", "learning_lineage_updates", "actor_updates_consumed"):
        assert infos[key] == evidence["seal"][key]
    assert infos["successful_updates_total"] == 39000
    result = dict(schema="source023-passive-diagnostic-immutable-seal39000-v1",status="downloaded_SHA_and_CPU_full_state_verified",
        started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        readiness_live=dict(file=str(live_path),sha256=sha(live_path)),source_commit=SOURCE,source_pins=PINS,files=evidence["files"],
        seal=evidence["seal"],actor_tensors=len(actor),critic_tensors=len(critic),Adam_parameter_states=len(optimizer["state"]),
        finite_CPU_learning_tensors=len(tensors),remote_writes=0,PPO_updates=0,PPO_signals=0,CUDA_actions=0,native_execution=False,
        qualification=None,scope="Fixed immutable REC11795 source023 checkpoint for a future passive same-seal diagnostic. Not claimed live latest, best, deployment qualified, or native-admitted; no baseline trajectory fabricated.")
    write("verification.json", result)
    print(json.dumps(dict(status=result["status"],checkpoint_sha256=evidence["seal"]["checkpoint_sha256"],REC=11795,consumption_at_seal=39000,report=str(OUT/"verification.json"))))


if __name__ == "__main__":
    main()
