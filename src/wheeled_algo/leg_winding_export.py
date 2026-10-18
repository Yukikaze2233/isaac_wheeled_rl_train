"""Bind the V7 paired decoder to both explicit and automatic actor exports."""
import hashlib
import json


def bind_paired_winding(path, version="paired_common_v1"):
    import onnx

    if version not in ("paired_common_v1", "paired_branch107_v1"):
        raise ValueError("Unknown paired leg target version")
    model = onnx.load(str(path))
    if any(item.key == "leg_winding_version" for item in model.metadata_props):
        raise ValueError("Exported model already declares a leg winding version")
    item = model.metadata_props.add()
    item.key, item.value = "leg_winding_version", version
    if version == "paired_branch107_v1":
        from wheeled_tasks.chassis.v5_control import V5Control
        if any(item.key == "leg_target_model_spec_sha256" for item in model.metadata_props):
            raise ValueError("Exported model already declares a leg target geometry")
        item = model.metadata_props.add()
        item.key = "leg_target_model_spec_sha256"
        item.value = V5Control.BRANCH107_MODEL_SPEC_SHA256
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def export_actor(actor, directory, identity, observations):
    """Reuse the stock graph exporter; its historical source stays unchanged."""
    from .chassis_export import export_actor as export_stock_actor

    version = identity.get("leg_winding_version", "paired_common_v1")
    if version not in ("paired_common_v1", "paired_branch107_v1"):
        raise ValueError("Unknown paired leg target version")
    identity = {**identity, "leg_winding_version": version}
    if version == "paired_branch107_v1":
        from wheeled_tasks.chassis.v5_control import V5Control
        identity["leg_target_model_spec_sha256"] = V5Control.BRANCH107_MODEL_SPEC_SHA256
    result = export_stock_actor(actor, directory, identity, observations)
    path = directory / "policy.onnx"
    bind_paired_winding(path, version)
    result["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "policy.onnx.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
