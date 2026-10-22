"""Assessment boundaries and authenticated task decisions."""
import hashlib
import json
from pathlib import Path
import uuid
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_confirmed_bins(previous, decision, allowed):
    confirmed = decision.get("confirmed_jump_bins_m", previous)
    if (not isinstance(confirmed, list) or any(type(x) not in (int, float) or x not in allowed for x in confirmed)
            or not set(previous).issubset(confirmed)):
        raise ValueError("Task assessment cannot withdraw or invent confirmed jump bins")
    return sorted(set(confirmed))


def task_assessment_due(config, task_update):
    """An explicitly zero interval disables requests as well as waiting."""
    task = config.get("task_campaign", {})
    interval = task.get("evaluation_interval", 0)
    return bool(interval) and task_update > 0 and task_update % interval == 0


def next_formal_assessment_update(config, restored_learning_updates):
    """Choose one strictly future learning boundary, never a paid-budget clock."""
    if config.get("formal_v7_flat") is not True:
        return None
    settings = config.get("formal_stage_policy", {})
    mode = settings.get("assessment_mode", "pause_for_assessment")
    if type(mode) is not str or mode not in ("pause_for_assessment", "record_and_continue"):
        raise ValueError("Formal assessment_mode requires pause_for_assessment or record_and_continue")
    first = settings.get("first_assessment_update")
    interval = settings.get("assessment_interval_updates")
    if any(type(value) is not int or value <= 0 for value in (first, interval)):
        raise ValueError("Formal assessment requires positive first_assessment_update and assessment_interval_updates")
    if type(restored_learning_updates) is not int or restored_learning_updates < 0:
        raise ValueError("Formal assessment requires the restored nonnegative learning clock")
    return first if restored_learning_updates < first else first + (
        (restored_learning_updates - first) // interval + 1) * interval


def wait_for_task_assessment(directory, checkpoint, task_update, consumed, check_budget,
                             *, sleep=time.sleep):
    """Yield this GPU without discarding the live training scene or its episodes."""
    request = {"schema": "task-assessment-request-v1", "token": uuid.uuid4().hex,
        "task_actor_update": task_update, "consumed_updates": consumed,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint)}
    temporary = directory / "assessment_request.tmp"
    temporary.write_text(json.dumps(request, indent=2) + "\n")
    temporary.replace(directory / "assessment_request.json")
    response = directory / "assessment_decision.json"
    while True:
        check_budget()
        if response.exists():
            decision = json.loads(response.read_text())
            if decision.get("token") == request["token"]:
                if (decision.get("checkpoint_sha256") != request["checkpoint_sha256"]
                        or decision.get("decision") not in ("continue", "domain_qualified_stop")):
                    raise ValueError("Task assessment decision does not match its immutable checkpoint")
                evaluations = decision.get("evaluations", [])
                if (len(evaluations) != 2
                        or {e["seed"] for e in evaluations} != {290929, 290930}
                        or any(digest(Path(e["file"])) != e["sha256"] for e in evaluations)):
                    raise ValueError("Task assessment requires two authenticated fixed evaluations")
                if (decision["decision"] == "domain_qualified_stop"
                        and decision.get("domain_qualified") is not True):
                    raise ValueError("Early task stop requires a qualified domain")
                return decision
        sleep(1.)
