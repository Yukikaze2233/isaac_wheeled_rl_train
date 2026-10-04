"""Failure paths for the CPU checkpoint audit; no simulator or PPO updates."""
import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location("audit_exploration_cpu", Path(__file__).with_name("audit_exploration_cpu.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_saved_tensor_rejected(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        audit.tensor_tree_audit({"optimizer": {0: {"exp_avg": torch.tensor([bad])}}})


def test_nested_actor_adam_tensor_count_and_integer_step():
    result = audit.tensor_tree_audit({"actor": [torch.ones(6)], "adam": {0: {"step": torch.tensor(3), "exp_avg": torch.zeros(6)}}})
    assert result == {"finite": True, "tensor_count": 3, "elements": 13}


def test_exact_tree_rejects_parameter_change_and_missing_key():
    expected = {"std": torch.ones(6), "state": [{"step": 20}]}
    assert audit.equal_tree(expected, expected)
    assert not audit.equal_tree({"std": torch.zeros(6), "state": [{"step": 20}]}, expected)
    assert not audit.equal_tree({"std": torch.ones(6)}, expected)


def test_native_gaussian_is_six_learned_scalars_and_mean_is_std_independent():
    distribution = audit.GaussianDistribution(6, init_std=.2)
    assert distribution.std_type == "scalar"
    assert distribution.std_param.shape == (6,)
    assert distribution.std_param.requires_grad
    mean = torch.tensor([[1., 2., 3., 4., 5., 6.]])
    assert torch.equal(distribution.deterministic_output(mean), mean)
    with torch.no_grad():
        distribution.std_param.fill_(.8)
    assert torch.equal(distribution.deterministic_output(mean), mean)


def test_native_clamp_is_reported_separately_from_raw_parameter():
    distribution = audit.GaussianDistribution(6, init_std=.2)
    with torch.no_grad():
        distribution.std_param[0] = -1.
    distribution.update(torch.zeros(1, 6))
    assert distribution.std_param[0] == -1.
    assert distribution.std[0, 0].item() == pytest.approx(1e-6)


def test_pinned_sha_rejects_unknown_checkpoint_without_loading(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Pinned PT hash changed"):
        audit.summarize("bad", str(pt), "unused", "unused", "unused", "0" * 64)
