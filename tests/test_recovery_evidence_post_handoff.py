"""Join native handoff history to committed actor samples and source failures."""
import ast
from pathlib import Path

import pytest
import torch

from test_recovery_evidence import journal, physical_row, total
from wheeled_tasks.chassis.recovery_evidence import EVENTS, FIELDS, SCHEMA


def row(book, interval, *, episode=0, profile=0, ready=False, handoff=False,
        followup=False, learning=True, done=False, terminated=False, timeout=False):
    value = physical_row(book, 0, episode=episode, interval=interval,
                         learning=torch.tensor([learning]))
    value["episode_profile_ids"][:] = profile
    value["episode_lesson_ids"][:] = 2 if profile else 0
    value["done"][:] = done
    value["terminated"][:] = terminated
    value["timeouts"][:] = timeout
    value["reset_mask"][:] = done
    value["returned_episode_ids"][:] = episode + int(done)
    value["events"] = {name: torch.tensor([flag]) for name, flag in zip(
        EVENTS, (ready, handoff, False, followup))}
    value["occurrence_ids"] = {name: torch.tensor([interval if flag else -1])
        for name, flag in zip(EVENTS, (ready, handoff, False, followup))}
    return value


def rollout(book, rows, *, finish="actor", population=None, reasons=None):
    book.begin_rollout(actor_enabled=finish != "critic")
    for step, value in enumerate(rows):
        book.observe_transition(step_index=step, payload=value,
            source_contact_or_attitude=None if reasons is None else reasons[step])
    if finish == "discard":
        book.discard_rollout()
        return
    learning = torch.stack([value["policy_learning_mask"] for value in rows])
    book.bind_loss_population(final_learning_mask=learning,
        optimized_population_mask=learning if population is None else population)
    if finish == "fail":
        book.fail_update()
    else:
        book.finish_update(actor_optimized=finish == "actor")


def test_masked_handoff_crosses_rollouts_then_credits_open_episode_once():
    book = journal(steps=2, count=1)
    rollout(book, [row(book, 1, ready=True, learning=False),
                   row(book, 2, handoff=True, learning=False)])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 0
    rollout(book, [row(book, 3), row(book, 4, followup=True)],
            reasons=[torch.tensor([False]), torch.tensor([False])])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 1
    assert total(book, "closed_episodes") == 0  # First loss can precede episode end.
    assert total(book, "actor_direct_loss_rl_entered") == 0
    rollout(book, [row(book, 5), row(book, 6, done=True, terminated=True)],
            reasons=[torch.tensor([False]), torch.tensor([True])])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 1
    assert total(book, "post_handoff_source_contact_or_attitude_termination_episodes") == 1
    tags = book.metrics()
    prefix = "/RecoveryEvidence/profile/upright/"
    for name in ("script_ready", "legacy_followup_complete", "episodes_with_post_handoff_actor_loss"):
        assert tags[prefix + name] == 1
    assert tags[prefix + "source_contact_or_attitude_measured_intervals"] == 4
    assert tags[prefix + "post_handoff_source_contact_or_attitude_termination_episodes"] == 1
    state = book.checkpoint_state()
    assert state["schema"] == SCHEMA == "v7-recovery-evidence-state-v2"
    assert state["field_names"] == list(FIELDS)
    assert "open episodes included" in state["post_handoff_actor_episode_scope"]


def test_multiple_native_episodes_in_one_rollout_keep_old_profiles_and_no_tail_credit():
    book = journal(steps=6, count=1)
    frames = [row(book, 1, handoff=True, learning=False),
              row(book, 2, done=True, terminated=True),
              row(book, 3, episode=1, profile=1, handoff=True, learning=False),
              row(book, 4, episode=1, profile=1, done=True, terminated=True),
              row(book, 5, episode=2, handoff=True, learning=False),
              row(book, 6, episode=2)]
    population = torch.tensor([[False], [True], [False], [True], [False], [False]])
    rollout(book, frames, population=population, reasons=[torch.tensor([True])] * 6)
    assert total(book, "episodes_with_post_handoff_actor_loss") == 2
    assert total(book, "post_handoff_source_contact_or_attitude_termination_episodes") == 2
    assert book.metrics()["/RecoveryEvidence/profile/upright/episodes_with_post_handoff_actor_loss"] == 1
    assert book.metrics()["/RecoveryEvidence/profile/fallen/episodes_with_post_handoff_actor_loss"] == 1
    # Exclusion from the actual minibatch is not evidence of actor learning.
    rollout(book, [row(book, index, episode=2) for index in range(7, 13)])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 3


@pytest.mark.parametrize("finish", ["fail", "critic", "discard"])
def test_non_actor_commit_keeps_physical_joint_measurement_without_losing_later_credit(finish):
    book = journal(steps=2, count=1)
    frames = [row(book, 1, handoff=True, learning=False), row(book, 2)]
    if finish == "discard":
        # Partial unpaid collection still preserves a masked handoff.
        rollout(book, frames[:1], finish=finish, reasons=[torch.tensor([False])])
        next_interval = 2
    else:
        rollout(book, frames, finish=finish, reasons=[torch.tensor([False])] * 2)
        next_interval = 3
    assert total(book, "episodes_with_post_handoff_actor_loss") == 0
    assert int(book.last_credited_actor_episode_ids[0]) == -1
    rollout(book, [row(book, next_interval),
                   row(book, next_interval + 1, done=True, terminated=True)],
            finish="fail", reasons=[torch.tensor([False]), torch.tensor([True])])
    assert total(book, "post_handoff_source_contact_or_attitude_termination_episodes") == 1
    assert total(book, "episodes_with_post_handoff_actor_loss") == 0
    before = book.counts.clone()
    book.fail_update()
    book.discard_rollout()
    assert torch.equal(book.counts, before)
    rollout(book, [row(book, next_interval + 2, episode=1, profile=1, handoff=True, learning=False),
                   row(book, next_interval + 3, episode=1, profile=1)])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 1
    assert book.metrics()["/RecoveryEvidence/profile/fallen/episodes_with_post_handoff_actor_loss"] == 1


def test_optional_source_witness_is_cloned_and_unavailable_is_not_observed_zero():
    book = journal(steps=1, count=1)
    # No diagnostics, despite a post-handoff termination: do not infer a cause.
    rollout(book, [row(book, 1, handoff=True, done=True, terminated=True)])
    tags = book.metrics()
    assert tags["/RecoveryEvidence/profile/upright/source_contact_or_attitude_measured_intervals"] == 0
    assert total(book, "post_handoff_source_contact_or_attitude_termination_episodes") == 0
    assert "zero measured intervals means unavailable" in book.checkpoint_state()["source_contact_or_attitude_scope"]
    # The source witness is nonexclusive: another producer FSM reason does not
    # erase a concurrently observed contact/attitude cause or invent priority.
    value = row(book, 2, episode=1, profile=1, handoff=True, done=True, terminated=True)
    value["failure_reasons"]["recovery_actor_failure"][:] = True
    witness = torch.tensor([True])
    book.begin_rollout(actor_enabled=True)
    book.observe_transition(step_index=0, payload=value, source_contact_or_attitude=witness)
    witness[:] = False
    mask = value["policy_learning_mask"].unsqueeze(0)
    book.bind_loss_population(final_learning_mask=mask, optimized_population_mask=mask)
    book.finish_update(actor_optimized=True)
    assert total(book, "source_contact_or_attitude_measured_intervals") == 1
    assert total(book, "post_handoff_source_contact_or_attitude_termination_episodes") == 1
    assert total(book, "recovery_actor_failure") == 1
    # A timeout and a pre-handoff source failure do not become post-handoff failures.
    rollout(book, [row(book, 3, episode=2, done=True, timeout=True, handoff=True)],
            reasons=[torch.tensor([True])])
    rollout(book, [row(book, 4, episode=3, done=True, terminated=True)],
            reasons=[torch.tensor([True])])
    assert total(book, "post_handoff_source_contact_or_attitude_termination_episodes") == 1


def test_inference_collection_then_normal_update_uses_owned_episode_state():
    book = journal(steps=2, count=1)
    with torch.inference_mode():
        rollout(book, [row(book, 1, handoff=True, learning=False)], finish="discard")
    assert total(book, "episodes_with_post_handoff_actor_loss") == 0
    rollout(book, [row(book, 2), row(book, 3)])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 1
    with torch.inference_mode():
        rollout(book, [row(book, 4)], finish="discard")
    rollout(book, [row(book, 5), row(book, 6)])
    assert total(book, "episodes_with_post_handoff_actor_loss") == 1


def test_entry_passes_existing_readonly_reason_to_both_actual_observers():
    path = Path(__file__).resolve().parents[1] / "scripts/train_chassis.py"
    tree = ast.parse(path.read_text())
    observers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "observe_transition"
                 and isinstance(node.func.value, ast.Name)
                 and node.func.value.id in ("journal", "recovery_evidence")]
    assert len(observers) == 2
    for call in observers:
        expression = next(kw.value for kw in call.keywords if kw.arg == "source_contact_or_attitude")
        witness = torch.tensor([True])
        extras = {"diagnostics": {"reasons": {"source_attitude_or_contact": witness}}}
        namespace = {"transition": (None, None, None, extras), "data": extras["diagnostics"]}
        code = compile(ast.Expression(expression), str(path), "eval")
        assert eval(code, namespace) is witness
        extras["diagnostics"]["reasons"].clear()
        assert eval(code, namespace) is None
