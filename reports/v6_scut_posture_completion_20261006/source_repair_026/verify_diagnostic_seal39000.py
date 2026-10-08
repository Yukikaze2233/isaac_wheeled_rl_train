"""CPU validation of downloaded native RSL-RL actor/critic separate states."""
from pathlib import Path
import datetime
import hashlib
import json
import time

import torch

BASE = Path(__file__).resolve().parent
OUT = BASE / "hold_joint_current_candidate002" / "immutable_update39000"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tick = time.monotonic()
    evidence = json.loads((OUT / "selection.json").read_text())
    for name, record in evidence["files"].items():
        assert sha(OUT / name) == record["sha256"]
        assert (OUT / name).stat().st_size == record["bytes"]
    before = {name: sha(OUT / name) for name in evidence["files"]}
    assert all(json.loads((OUT / name).read_text())["exit_code"] == 0
               for name in ("selection.execution.json", "download.execution.json"))
    seal = json.loads((OUT / "model.checkpoint.json").read_text())
    assert seal == evidence["seal"] and seal["sealed"] and seal["learning_updates"] == 39000
    saved = torch.load(OUT / "model.pt", map_location="cpu", weights_only=True)
    actor = saved["actor_state_dict"]
    critic = saved["critic_state_dict"]
    optimizer = saved["optimizer_state_dict"]
    expected = {f"mlp.{layer}.{suffix}" for layer in (0, 2, 4, 6) for suffix in ("weight", "bias")}
    assert set(actor) == expected | {"distribution.std_param"}
    assert set(critic) == expected
    assert actor["distribution.std_param"].shape == (6,)
    assert actor["mlp.0.weight"].shape == (256, 35) and actor["mlp.6.weight"].shape == (6, 64)
    assert critic["mlp.0.weight"].shape == (256, 81) and critic["mlp.6.weight"].shape == (1, 64)
    assert len(optimizer["state"]) == 17
    assert sum(len(group["params"]) for group in optimizer["param_groups"]) == 17
    tensors = []
    def visit(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and torch.isfinite(value).all()
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(actor); visit(critic); visit(optimizer)
    infos = saved["infos"]
    assert infos["contract_sha256"] == seal["contract_sha256"] == before["model.contract.json"]
    for key in ("stage_actor_update", "task_actor_update", "learning_lineage_updates", "actor_updates_consumed"):
        assert infos[key] == seal[key]
    assert infos["successful_updates_total"] == 39000 and infos["critic_warmup_remaining"] == 0
    contract = json.loads((OUT / "model.contract.json").read_text())
    assert contract["actor_dim"] == 35 and contract["critic_dim"] == 81 and contract["action_dim"] == 6
    assert contract["policy_dt"] == .02 and contract["physics_dt"] == .005
    assert contract["position_iterations"] == 64 and contract["velocity_iterations"] == 32
    assert contract["closure_gap_termination_m"] == .003
    assert {name: sha(OUT / name) for name in evidence["files"]} == before
    result = dict(schema="source023-passive-diagnostic-immutable-seal39000-v2",status="downloaded_SHA_and_native_split_state_CPU_verified",
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verifier_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-tick,
        source_commit=evidence["source_commit"],source_pins=evidence["source_pins"],files=evidence["files"],seal=seal,
        actor_tensors=len(actor),critic_tensors=len(critic),Adam_parameter_states=len(optimizer["state"]),finite_CPU_learning_tensors=len(tensors),
        source_selection=dict(live_file=str(BASE/"live020.json"),live_sha256=sha(BASE/"live020.json"),scope="Existing immutable39000 at live020; not claimed later live latest"),
        inputs_unchanged=True,remote_writes=0,SSH_calls=0,PPO_updates=0,PPO_signals=0,CUDA_actions=0,native_execution=False,qualification=None,
        limitations=["Original old model_state_dict CPU parser rejected; download/SHA evidence retained, only this separate CPU verifier repairs format handling.",
            "Full native admission, original observer callback readiness and measured trajectory remain separate prerequisites.",
            "No baseline trajectory fabricated, no deterministic behavior, best-model or hardware qualification claim."])
    with (OUT / "verification.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],checkpoint_sha256=seal["checkpoint_sha256"],REC=seal["task_actor_update"],report=str(OUT/"verification.json"))))


if __name__ == "__main__":
    main()
