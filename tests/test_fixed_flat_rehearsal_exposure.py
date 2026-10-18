"""CPU direct-loss exposure arithmetic and the actual entry commit boundary."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_algo.recovery_rollout import RecoveryRolloutMask

ENTRY = Path(__file__).resolve().parents[1] / "scripts/train_chassis.py"


def entry_parts():
    tree = ast.parse(ENTRY.read_text())
    names = {"fixed_flat_rehearsal_assignment", "fixed_flat_rehearsal_counts", "timed_call"}
    scope = {}
    exec(compile(ast.Module(body=[n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(ENTRY), "exec"), scope)
    update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update")
    return scope, update


def assignment(scope):
    # CPU fixture only, not a chosen training quota. Extra spec fields are owned
    # by the environment and must not alter these fixed native-order labels.
    spec = {"case_ids": ("forward", "reverse"), "case_index": (-1, 0, 1, 0),
            "settings": {"fixture": True}}
    before = deepcopy(spec)
    indices, receipt = scope["fixed_flat_rehearsal_assignment"](spec, 4, "cpu")
    assert spec == before
    return indices, receipt


def actual_population():
    mask = RecoveryRolloutMask.__new__(RecoveryRolloutMask)
    mask._validate_storage = lambda: None
    mask._returns_ready = True
    mask.shape = (3, 4)
    mask.storage = SimpleNamespace(step=3)
    mask.algorithm = SimpleNamespace(num_mini_batches=4)
    mask.learning = torch.tensor([[True, True, True, False],
                                  [True, False, True, True],
                                  [True, True, True, True]])
    # Ten eligible frames, eight enter the actual native minibatch population.
    return mask.loss_population_mask()


def test_counts_use_actual_loss_population_not_lane_quota_or_epochs():
    scope, _ = entry_parts()
    indices, receipt = assignment(scope)
    mask = actual_population()
    saved_mask, saved_indices = mask.clone(), indices.clone()
    result = scope["fixed_flat_rehearsal_counts"](mask, indices, 3)
    assert result.dtype == torch.int64 and result.tolist() == [3, 3, 2]
    assert receipt["nominal_lane_fractions"] == {"rough": .25, "forward": .5, "reverse": .25}
    assert result.sum() == mask.sum() == 8
    assert torch.equal(mask, saved_mask) and torch.equal(indices, saved_indices)
    assert scope["fixed_flat_rehearsal_counts"](torch.zeros_like(mask), indices, 3).tolist() == [0, 0, 0]
    assert assignment(scope)[1]["worker_direct_loss_transitions"] == {"rough": 0, "forward": 0, "reverse": 0}


@pytest.mark.parametrize("fails", [False, True])
def test_real_entry_counts_commit_only_after_optimizer_returns(fails):
    scope, update = entry_parts()
    indices, receipt = assignment(scope)
    report = {"fixed_flat_rehearsal": receipt, "parent_updates": 5000, "successful_updates": 1}
    original_report = deepcopy(report)
    mask = actual_population()
    logged = []
    def optimizer():
        if fails:
            raise RuntimeError("failed optimizer fixture")
        mask.zero_()  # Native storage reuse cannot alter the prepared aggregate.
        return {"value": 1.}
    # Execute the production prepare/native-call/commit nodes themselves.
    wanted = []
    for node in update.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
                "rehearsal_counts", "result") for t in node.targets):
            wanted.append(node)
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "rehearsal_counts is not None":
            wanted.append(node)
    assert len(wanted) == 3
    scope.update(report=report, loss_mask=mask, rehearsal_indices=indices, actor_enabled=True,
        original_update=optimizer, timing=None, a=(), kw={},
        runner=SimpleNamespace(logger=SimpleNamespace(writer=object())),
        log_scalar_batch=lambda writer, values, step: logged.append((dict(values), step)))
    code = compile(ast.Module(body=wanted, type_ignores=[]), str(ENTRY), "exec")
    if fails:
        with pytest.raises(RuntimeError, match="failed optimizer"):
            exec(code, scope)
        assert report == original_report and not logged
    else:
        exec(code, scope)
        assert receipt["successful_actor_updates"] == 1
        assert receipt["worker_direct_loss_transitions"] == receipt["last_update_direct_loss_transitions"] == {
            "rough": 3, "forward": 3, "reverse": 2}
        assert logged[0][0]["FixedFlatRehearsal/direct_loss/forward/fraction"] == 3 / 8
        assert logged[0][0]["FixedFlatRehearsal/nominal_lane_fraction/forward"] == .5
        assert logged[0][1] == 5000
        # A successful empty second population cannot carry old counts forward.
        exec(code, scope)
        assert receipt["successful_actor_updates"] == 2
        assert receipt["last_update_direct_loss_transitions"] == {"rough": 0, "forward": 0, "reverse": 0}
        assert receipt["worker_direct_loss_transitions"] == {"rough": 3, "forward": 3, "reverse": 2}
        assert not any("nominal" in tag for tag in logged[1][0])


def test_disabled_and_critic_only_do_not_reduce_or_publish():
    scope, update = entry_parts()
    node = next(n for n in update.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "rehearsal_counts" for t in n.targets))
    def forbidden(*args):
        raise AssertionError("disabled rehearsal attempted reduction")
    scope.update(fixed_flat_rehearsal_counts=forbidden, rehearsal_indices=None,
                 actor_enabled=True, report={}, loss_mask=None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ENTRY), "exec"), scope)
    assert scope["rehearsal_counts"] is None
    scope.update(rehearsal_indices=torch.tensor([0]), actor_enabled=False)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ENTRY), "exec"), scope)
    assert scope["rehearsal_counts"] is None
