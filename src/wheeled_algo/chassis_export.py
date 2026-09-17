"""Export the stock RSL5 deterministic actor with an explicit new-task identity."""
import hashlib
import json


def export_actor(actor, directory, identity, observations):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    model = actor.as_onnx(verbose=False).cpu().eval()
    sample = observations.detach().cpu().float()[:8].clone()
    if sample.shape[1] != model.input_size:
        raise ValueError("Export input does not match actor architecture")
    path = directory / "policy.onnx"
    torch.onnx.export(model, sample[:1], str(path), input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17,
        dynamo=False, external_data=False)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.
    for value in (torch.zeros_like(sample[:1]), sample[:1], sample):
        with torch.no_grad():
            expected = model(value).numpy()
        actual = session.run(["actions"], {"obs": value.numpy()})[0]
        if actual.shape != (len(value), 6) or not np.isfinite(actual).all():
            raise RuntimeError("Invalid ONNX output")
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        worst = max(worst, float(np.max(np.abs(actual - expected))))
    result = {**identity, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "input": ["batch", model.input_size], "output": ["batch", 6],
              "normalization": "contract-scaled observations; no running normalizer",
              "verified": True, "max_abs_error": worst}
    (directory / "policy.onnx.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
