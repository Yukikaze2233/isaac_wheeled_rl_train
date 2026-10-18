"""Synthetic CPU Adam lifecycle fixture, never a robot or PPO training job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from wheeled_algo.training_control import TrainingControl, atomic_json
from wheeled_tasks.chassis.training_budget import BudgetOwner, digest_file


class Meter:
    owner_id = "cli_command_cpu_fixture"

    def read_total_gpu_seconds(self):
        return 0.


def open_book(args):
    review = json.loads(Path(args.budget_review_file).read_text())
    identity = {key: value for key, value in review["attempts"][args.budget_attempt_id].items() if key != "role"}
    identity["requested_updates"] = args.updates
    return BudgetOwner.authenticate(args.budget_review_file, args.budget_review_sha256,
        args.budget_ledger, args.budget_attempt_id, identity, resume_checkpoint=args.resume,
        gpu_meter=Meter(), allow_test_meter=True, now=lambda: datetime.now(timezone.utc))


def main():
    parser = argparse.ArgumentParser()
    for name in ("run-dir", "contract", "resume", "budget-review-file", "budget-review-sha256",
                 "budget-ledger", "budget-attempt-id", "rng-mode"):
        parser.add_argument("--" + name)
    parser.add_argument("--updates", type=int)
    args = parser.parse_args()
    run = Path(args.run_dir)
    run.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(321)
    actor, critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=.001)
    loaded = torch.load(args.resume, weights_only=True, map_location="cpu")
    actor.load_state_dict(loaded["actor"])
    critic.load_state_dict(loaded["critic"])
    optimizer.load_state_dict(loaded["optimizer"])
    torch.set_rng_state(loaded["rng"])
    update = loaded["update"]
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "update": update},
               run / "loaded.pt")
    book = open_book(args)
    control = TrainingControl(run)

    def save(count):
        path = run / f"model_{count}.pt"
        if path.exists():
            book.verify_resume(path)
            return path
        receipt = book.prepare_checkpoint(path)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "update": count,
            "infos": {"training_budget": receipt}}, path)
        atomic_json(path.with_suffix(".checkpoint.json"), {"sealed": True,
            "checkpoint_sha256": digest_file(path), "contract_sha256": digest_file(args.contract),
            "learning_updates": count, "training_budget": receipt})
        book.register_checkpoint(path, receipt)
        return path

    class Stop(Exception):
        pass

    def stop():
        raise Stop()

    status = "completed"
    try:
        for _ in range(args.updates):
            token = book.reserve_update(phase="actor")
            x = torch.randn(8, 3)
            loss = actor(x).square().mean() + critic(x).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update += 1
            book.complete_update(token, actor_optimized=True)
            print(f"CPU_FIXTURE_UPDATE {update}", flush=True)
            control.boundary(update, checkpoint=save, flush=lambda: None,
                             check_limits=book.check_runtime, stop=stop)
            time.sleep(.05)
    except Stop:
        status = "stopped"
    finally:
        path = save(update)
        book.close()
        atomic_json(run / "completion.json", {"status": status, "checkpoint": str(path)})
        control.finish(status)


if __name__ == "__main__":
    main()
