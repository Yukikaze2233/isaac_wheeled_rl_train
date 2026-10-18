"""CPU preparation, actor sampling and paid-state adapters for flat evaluation.

flat baseline command/reward ownership lives in the existing domain modules. This
module creates no simulator and grants no native or mechanical admission.
Evaluation overlays are separate artifacts, never rewritten training contracts.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

from .flat_policy_evaluation import (
    FlatEpisodeLedger, _canonical_payment_names, validate_protocol, verify_checkpoint_identity,
)
from .flat_episode_selection import validate_flat_episode_selection
from .scut_flat_reward import WEIGHTS
from .scut_reward_selection import SCUT_FLAT_PROFILE, SCUT_ROUGH_PROFILES
from .training_profile import (profile_source_files, restore_profile_clocks,
                               validate_profile_selection, verify_profile_evaluation, profile_identity)


VERSION = "flat-evaluation-preparation-v1"
OVERLAY_FIELDS = frozenset(("num_envs", "lane_case_ids", "record_diagnostics"))

# One failure-investigation patch, not a caller-supplied source allowlist.
NUMERICAL_CAPTURE = {
    "schema": "rough016-numerical-capture-v1",
    "checkpoint_sha256": "46ef77778e4c66bb500d48b2efa58bd6751d76504ae3ac33e1fbf23dc7e54f8e",
    "contract_sha256": "355de2d462da72306952f931cb7db92efa81a2a00fdcd7424e3690df7c0c541a",
    "env_file": "src/wheeled_tasks/chassis/env.py",
    "old_env_sha256": "cc1feba45bc268c76ff4676b4e2a0633f10ccf9320bc3ab663b5beaada73d86c",
    "new_env_sha256": "6302164642d565ad2c20ccff8a9e0a47e1e2b5774839b312591e59ff2e21646d",
    "helper_file": "src/wheeled_tasks/chassis/numerical_failure_capture.py",
    "helper_sha256": "bbd52b43d5fd435a595409ceed969d9537afeacf72403522e114ad07623fe0bc",
}


def numerical_capture_compatibility(*, checkpoint_sha256, contract_sha256, protocol,
                                    overlay, infos, runtime):
    """Permit only the pinned pre-reset capture patch on the actual 4701 parent."""
    pin = NUMERICAL_CAPTURE
    if (checkpoint_sha256 != pin["checkpoint_sha256"] or contract_sha256 != pin["contract_sha256"]
            or infos.get("stage") != "terrain" or infos.get("global_actor_update") != 4701
            or infos.get("stage_actor_update") != 201):
        raise ValueError("Numerical capture requires the pinned original rough contract and 4701 checkpoint")
    expected = dict(schema="flat-policy-evaluation-v1", mode="source_distribution",
        actor_mode="sampled", episodes_per_lane=3, environment_seed=290929, actor_seed=290929,
        policy_dt_s=.02, horizon_seconds=20., center_reference="root_link_origin_world",
        assessment="measurement_only")
    if (protocol != expected or set(overlay) != OVERLAY_FIELDS or overlay["num_envs"] != 4096
            or overlay["lane_case_ids"] != ["source_distribution"] * 4096
            or overlay["record_diagnostics"] is not True):
        raise ValueError("Numerical capture requires sampled source distribution, 4096 lanes, and three 20s episodes")
    saved = infos.get("profile_source_sha256", {})
    if (saved.get(pin["env_file"]) != pin["old_env_sha256"]
            or runtime.get(pin["env_file"]) != pin["new_env_sha256"]
            or pin["helper_file"] in saved
            or runtime.get(pin["helper_file"]) != pin["helper_sha256"]):
        raise ValueError("Numerical capture source differs from the exact reviewed env/helper patch")
    comparison = dict(runtime)
    comparison[pin["env_file"]] = pin["old_env_sha256"]
    return comparison, {**pin, "scope": "pre-reset first numerical-event capture only; unchanged dynamics and thresholds",
        "exact_failed_4729_replay": False, "training_release": False}


def formal_entry_compatibility(contract, saved, runtime):
    """One reviewed entry-role change; physical/observation sources stay exact.

    The foundation007 preflight admitted only train/4096. Its replacement also
    admits source_flat_evaluation/1..4096, retaining every physical check. Both
    complete file hashes are pinned here; this is not a general source bypass.
    Actual runtime hashes remain in the receipt and evaluation budget identity.
    """
    name = "src/wheeled_tasks/chassis/preflight.py"
    old = "f58edb651ec5c0046880501028487ea0b4ba7075c49edfc91648b27fd66eedb9"
    new = "91d946939265d53486fcda684236d77c4ff940190f7247da6bd8b5ff7b1e9bbb"
    if (contract.get("formal_v7_flat") is True and isinstance(saved, Mapping)
            and saved.get(name) == old and runtime.get(name) == new):
        compatible = dict(runtime)
        compatible[name] = old
        return compatible, {"file": name, "checkpoint_source_sha256": old,
            "evaluation_source_sha256": new,
            "scope": "entry_role_and_lane_count_admission_only; no physical/control/reset changes"}
    return runtime, None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(data, label):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value in {label}: {value}")
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class BoundFile:
    path: Path
    data: bytes

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        return cls(path, path.read_bytes())

    @property
    def digest(self):
        return sha(self.data)

    def receipt(self):
        return {"path": str(self.path), "sha256": self.digest, "size_bytes": len(self.data)}


def _relative(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Source paths must be nonempty workspace-relative paths")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Source symlink escapes the workspace")
    return path


def runtime_source_files(contract):
    """Captured shared project dependencies; not a full SDK dependency closure."""
    result = profile_source_files(contract) + [
        "src/wheeled_tasks/chassis/flat_policy_evaluation.py",
        "src/wheeled_tasks/chassis/spin_center_measurement.py",
        "src/wheeled_tasks/chassis/flat_evaluation_runtime.py",
        "src/wheeled_tasks/chassis/flat_evaluation_setup.py",
        "src/wheeled_tasks/chassis/small_rough_evaluation.py",
        "src/wheeled_tasks/chassis/temporal_flat_evaluation.py",
        "src/wheeled_tasks/chassis/temporal_flat_rehearsal.py",
        "src/wheeled_tasks/chassis/scripted_recovery_evaluation.py",
        "src/wheeled_tasks/chassis/flat_evaluation_loop.py",
        "src/wheeled_tasks/chassis/usd_dependency_closure.py",
        "scripts/evaluate_flat_policy.py", contract["control_math_source"]]
    for key, names in (("signal_delay", ("signal_delay", "delay_buffer")),
                       ("feedback_transport", ("feedback_transport", "feedback_transport_config", "pc_communication", "delay_buffer"))):
        if key in contract:
            result.extend(f"src/wheeled_tasks/chassis/{name}.py" for name in names)
    for key in ("command_transport", "usb_transport", "dynamics_randomization", "contact_domain"):
        if contract.get(key, {}).get("enabled"):
            name = "dynamics" if key == "dynamics_randomization" else key
            result.append(f"src/wheeled_tasks/chassis/{name}.py")
    if contract.get("actuator_response", {}).get("enabled"):
        evidence = contract.get("actuator_response_evidence")
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("file"), str):
            raise ValueError("Evaluation requires the declared actuator response evidence; a candidate is not evidence")
        result.append(evidence["file"])
    if contract.get("reward_profile") in SCUT_ROUGH_PROFILES:
        result.append("src/wheeled_tasks/chassis/rough_evaluation.py")
    if contract.get("recovery_training", {}).get("source_flat_handoff"):
        result.append("src/wheeled_tasks/chassis/evaluation.py")
    return list(dict.fromkeys(result))


def validate_overlay(protocol, overlay):
    from .small_rough_evaluation import SURFACE, validate_overlay as validate_small_rough
    from .temporal_flat_evaluation import selected as temporal_selected
    from .scripted_recovery_evaluation import selected as recovery_selected
    if (not isinstance(overlay, Mapping)
            or set(overlay) not in (OVERLAY_FIELDS, OVERLAY_FIELDS | {"evaluation_surface"},
                                   OVERLAY_FIELDS | {"contact_generation_distance"},
                                   OVERLAY_FIELDS | {"evaluation_surface", "contact_generation_distance"})):
        raise ValueError("Evaluation overlay requires num_envs, lane_case_ids, record_diagnostics, and optional evaluation_surface")
    if "evaluation_surface" in overlay and overlay["evaluation_surface"] not in ("plane_regression_v1", SURFACE):
        raise ValueError("Unknown explicit regression surface")
    if overlay.get("evaluation_surface") == SURFACE:
        validate_small_rough(protocol, overlay)
    elif "contact_generation_distance" in overlay:
        from .contact_generation_distance import SUPPORTED_VERSIONS
        if (not (temporal_selected(protocol) or recovery_selected(protocol))
                or "evaluation_surface" in overlay
                or overlay["contact_generation_distance"] not in SUPPORTED_VERSIONS):
            raise ValueError("Contact candidate overlay requires an explicit retained-skill regression")
    count = overlay["num_envs"]
    if type(count) is not int or count <= 0 or overlay["record_diagnostics"] is not True:
        raise ValueError("Evaluation requires a positive lane count and pre-reset diagnostics")
    lanes = overlay["lane_case_ids"]
    if not isinstance(lanes, (list, tuple)) or len(lanes) != count:
        raise ValueError("Evaluation lane assignment must match num_envs")
    from .scripted_recovery_evaluation import selected, scene_groups
    if selected(protocol):
        if "evaluation_surface" in overlay:
            raise ValueError("Scripted recovery evaluation supports only its existing flat surface")
        scene_groups(protocol, lanes)
    # Reuse the actual ledger's exact case-coverage validation.
    FlatEpisodeLedger(protocol, lanes, list(WEIGHTS))
    result = {"num_envs": count, "lane_case_ids": list(lanes), "record_diagnostics": True}
    if "evaluation_surface" in overlay:
        result["evaluation_surface"] = overlay["evaluation_surface"]
    if "contact_generation_distance" in overlay:
        result["contact_generation_distance"] = overlay["contact_generation_distance"]
    return result


class BoundEvaluationInputs:
    """Capture input bytes once, then authenticate and deserialize those bytes.

    Rechecking paths detects ordinary parallel edits; only an externally frozen
    execution tree can prevent all races. Receipts do not claim native execution
    or that arbitrary files sharing a module name were actually imported.
    """

    @classmethod
    def capture(cls, *, root, checkpoint, contract, protocol, manifest, control, overlay,
                numerical_capture=False, numerical_capture_directory=None, diagnostics_directory=None):
        obj = cls()
        if type(numerical_capture) is not bool or (numerical_capture_directory is not None) != numerical_capture:
            raise ValueError("Numerical capture requires its explicit selector and output directory together")
        obj.numerical_capture = numerical_capture
        obj.numerical_capture_directory = (str(Path(numerical_capture_directory).resolve())
                                           if numerical_capture else None)
        obj.diagnostics_directory = (str(Path(diagnostics_directory).resolve())
                                     if diagnostics_directory is not None else None)
        obj.root = Path(root).resolve(strict=True)
        captured = {}
        def read_once(path):
            path = Path(path).resolve(strict=True)
            if path not in captured:
                captured[path] = BoundFile.read(path)
            return captured[path]
        obj.inputs = {name: read_once(path) for name, path in (
            ("checkpoint", checkpoint), ("contract", contract), ("protocol", protocol),
            ("manifest", manifest), ("control", control), ("overlay", overlay))}
        parsed = {name: strict_json(item.data, name) for name, item in obj.inputs.items() if name != "checkpoint"}
        obj.contract, obj.manifest, obj.control = parsed["contract"], parsed["manifest"], parsed["control"]
        obj.protocol = validate_protocol(parsed["protocol"])
        obj.overlay = validate_overlay(obj.protocol, parsed["overlay"])
        if not validate_profile_selection(obj.contract) or validate_flat_episode_selection(obj.contract) is None:
            raise ValueError("Independent flat evaluation requires the authenticated source episode and 78D profile")
        recovery = obj.contract.get("recovery_training", {}).get("source_flat_handoff")
        if bool(recovery) != obj.is_scripted_recovery and not (obj.is_small_rough_regression or obj.is_temporal_flat):
            raise ValueError("Scripted recovery checkpoint requires its explicit recovery evaluation protocol")
        if obj.is_scripted_recovery:
            from .scripted_recovery_evaluation import validate_binding
            validate_binding(obj.protocol, obj.contract, obj.overlay["lane_case_ids"], obj.root)
        if obj.is_plane_regression and not obj.is_rough:
            raise ValueError("Plane regression overlay requires an authenticated rough checkpoint")
        if obj.is_rough and not obj.is_plane_regression and obj.protocol["mode"] != "source_distribution":
            raise ValueError("Rough evaluation currently preserves the source distribution; fixed routes are not implemented")
        if obj.inputs["manifest"].digest != obj.contract.get("asset_manifest_sha256"):
            raise ValueError("Bound asset manifest differs from the training contract")
        if obj.inputs["control"].digest != obj.contract.get("control_math_sha256"):
            raise ValueError("Bound control differs from the training contract")
        names = runtime_source_files(obj.contract)
        if obj.is_small_rough_regression or obj.is_temporal_flat or obj.is_scripted_recovery:
            names.extend(runtime_source_files(obj.evaluation_config()))
            names.append("src/wheeled_tasks/chassis/small_rough_evaluation.py")
        if "heading_hold" in obj.protocol:
            names.append("src/wheeled_tasks/chassis/heading_hold_evaluation.py")
        if obj.numerical_capture:
            names.append(NUMERICAL_CAPTURE["helper_file"])
        obj.sources = {name: read_once(_relative(obj.root, name)) for name in names}
        if obj.sources[obj.contract["control_math_source"]].data != obj.inputs["control"].data:
            raise ValueError("Runtime control bytes differ from the explicitly bound control")
        # Evidence dependencies are parsed from the same bound evidence bytes.
        evidence = obj.contract.get("actuator_response_evidence")
        if obj.contract.get("actuator_response", {}).get("enabled"):
            content = strict_json(obj.sources[evidence["file"]].data, "actuator response evidence")
            for name, digest in content.get("files_sha256", {}).items():
                if name not in obj.sources:
                    obj.sources[name] = read_once(_relative(obj.root, name))
                item = obj.sources[name]
                if item.digest != digest:
                    raise ValueError(f"Actuator response evidence dependency mismatch: {name}")
        obj._checkpoint = None
        obj.authentication = None
        return obj

    def load_checkpoint(self):
        if self._checkpoint is None:
            import torch
            try:
                value = torch.load(io.BytesIO(self.inputs["checkpoint"].data), map_location="cpu", weights_only=True)
            except Exception as error:
                raise ValueError("Checkpoint rejected by restricted CPU weights_only loader; no unrestricted pickle fallback") from error
            if not isinstance(value, dict) or not isinstance(value.get("infos"), Mapping) or not isinstance(value.get("actor_state_dict"), Mapping):
                raise ValueError("Checkpoint requires infos and actor_state_dict mappings")
            if not value["actor_state_dict"] or any(not isinstance(n, str) or not isinstance(v, torch.Tensor)
                    or v.device.type != "cpu" or v.dtype != torch.float32 or not bool(torch.isfinite(v).all())
                    for n, v in value["actor_state_dict"].items()):
                raise ValueError("Actor checkpoint must contain finite CPU float32 tensors")
            self._checkpoint = value
        return deepcopy(self._checkpoint)

    def authenticate(self):
        for name in ("contract", "manifest", "control", "protocol", "overlay"):
            if canonical(getattr(self, name)) != canonical(strict_json(self.inputs[name].data, name)):
                raise ValueError(f"Parsed bound {name} was mutated after capture")
        checkpoint = self.load_checkpoint()
        infos = checkpoint["infos"]
        hashes = {name: item.digest for name, item in self.sources.items()}
        comparison, entry_adapter = formal_entry_compatibility(self.contract,
            infos.get("profile_source_sha256", {}), hashes)
        capture_adapter = None
        if self.numerical_capture:
            comparison, capture_adapter = numerical_capture_compatibility(
                checkpoint_sha256=self.inputs["checkpoint"].digest,
                contract_sha256=self.inputs["contract"].digest, protocol=self.protocol,
                overlay=self.overlay, infos=infos, runtime=hashes)
        profile = verify_profile_evaluation(self.contract, infos, comparison)
        identity = verify_checkpoint_identity(infos,
            expected_contract_sha256=self.inputs["contract"].digest,
            expected_asset_sha256=self.inputs["manifest"].digest,
            expected_control_sha256=self.inputs["control"].digest,
            checkpoint_sha256=self.inputs["checkpoint"].digest,
            runtime_source_sha256=comparison, required_sources=profile_source_files(self.contract))
        if entry_adapter is not None:
            profile["compatibility_comparison_identity"] = profile["runtime_identity"]
            profile["runtime_identity"] = profile_identity(self.contract, hashes)
            profile["scope"] = "exact_shared_runtime_except_the_pinned_entry_role_adapter"
            identity["source_equality_exception"] = entry_adapter
        if capture_adapter is not None:
            profile["compatibility_comparison_identity"] = profile["runtime_identity"]
            profile["runtime_identity"] = profile_identity(self.contract, hashes)
            profile["scope"] = "exact_shared_runtime_except_pinned_numerical_capture_patch"
            identity["source_equality_exception"] = capture_adapter
        stage = infos.get("stage")
        if stage not in self.contract.get("enabled_stages", []):
            raise ValueError("Checkpoint stage is not enabled by the bound training contract")
        clocks = restore_profile_clocks(self.contract, infos, stage=stage, requested={})
        self.authentication = {"profile": profile, "identity": identity, "stage": stage,
            "entry_role_adapter": entry_adapter,
            "numerical_capture_adapter": capture_adapter,
            "actor_clocks": clocks, "overlay_sha256": sha(canonical(self.overlay)),
            "protocol_sha256": self.inputs["protocol"].digest,
            "training_contract_rewritten": False}
        return deepcopy(self.authentication)

    def build_setup(self):
        from .flat_evaluation_setup import FlatEvaluationSetup
        receipt = self.authenticate()
        setup = FlatEvaluationSetup(self.protocol, tuple(self.overlay["lane_case_ids"]),
            receipt["actor_clocks"]["stage_actor_update"], receipt["actor_clocks"]["global_actor_update"],
            run_role=self.run_role, evaluation_surface=self.overlay.get("evaluation_surface", "training"))
        setup.validate_constructor(self.evaluation_config(), self.overlay["num_envs"],
            self.protocol["environment_seed"], self.run_role)
        return setup

    @property
    def is_scripted_recovery(self):
        from .scripted_recovery_evaluation import selected
        return selected(self.protocol)

    @property
    def is_rough(self):
        return self.is_small_rough_regression or (not self.is_scripted_recovery and not self.is_temporal_flat
            and self.contract.get("reward_profile") in SCUT_ROUGH_PROFILES)

    @property
    def is_temporal_flat(self):
        from .temporal_flat_evaluation import selected
        return selected(self.protocol)

    @property
    def is_small_rough_regression(self):
        from .small_rough_evaluation import SURFACE
        return self.overlay.get("evaluation_surface") == SURFACE

    @property
    def execution_stage(self):
        if self.is_small_rough_regression:
            return "terrain"
        if self.is_temporal_flat or self.is_scripted_recovery and "formal_small_rough_mix" in self.contract:
            return "foundation"
        return self.authentication["stage"]

    @property
    def is_plane_regression(self):
        return self.overlay.get("evaluation_surface") == "plane_regression_v1"

    @property
    def run_role(self):
        if self.is_scripted_recovery:
            return "source_scripted_recovery_evaluation"
        if self.is_plane_regression:
            return "source_rough_plane_regression"
        return "source_rough_evaluation" if self.is_rough else "source_flat_evaluation"

    def evaluation_config(self):
        """Diagnostics and explicit evaluation-only scene allocation; contract bytes stay bound."""
        result = deepcopy(self.contract)
        if self.is_small_rough_regression:
            from .small_rough_evaluation import configuration
            result = configuration(result)
        elif self.is_temporal_flat:
            from .temporal_flat_evaluation import configuration
            result = configuration(result)
        elif self.is_scripted_recovery:
            from .scripted_recovery_evaluation import validate_binding
            result = validate_binding(self.protocol, result, self.overlay["lane_case_ids"], self.root)
        if self.overlay.get("contact_generation_distance") is not None:
            result["formal_contact_generation_distance"] = self.overlay["contact_generation_distance"]
        result["record_diagnostics"] = True
        if self.numerical_capture:
            result["formal_physics_trace_directory"] = self.numerical_capture_directory
        elif self.diagnostics_directory is not None and result.get("formal_physics_trace_directory"):
            # Bind diagnostic relocation before construction and loop checks.
            result["formal_physics_trace_directory"] = self.diagnostics_directory
        return result

    def assert_unchanged(self):
        changed = [str(item.path) for item in (*self.inputs.values(), *self.sources.values())
                   if not item.path.is_file() or sha(item.path.read_bytes()) != item.digest]
        if changed:
            raise ValueError("Bound evaluation input/source changed: " + ", ".join(changed))

    def receipt(self):
        if self.authentication is None:
            raise ValueError("Authenticate the checkpoint before producing a preparation receipt")
        from .small_rough_evaluation import receipt as rough_receipt
        return {"schema": VERSION, "status": "prepared_only", "inputs": {k: v.receipt() for k, v in self.inputs.items()},
            "source_bindings": {k: v.receipt() for k, v in self.sources.items()},
            "authentication": deepcopy(self.authentication), "overlay": deepcopy(self.overlay),
            "effective_configuration_sha256": sha(canonical(self.evaluation_config())),
            "diagnostic_directory_override": ({
                "original": self.contract["formal_physics_trace_directory"],
                "effective": self.evaluation_config()["formal_physics_trace_directory"],
                "physical_configuration_changed": False}
                if self.contract.get("formal_physics_trace_directory")
                and self.evaluation_config().get("formal_physics_trace_directory")
                    != self.contract["formal_physics_trace_directory"] else None),
            "constructor_settings": {"num_envs": self.overlay["num_envs"],
                "seed": self.protocol["environment_seed"], "run_role": self.run_role,
                "stage_name": self.execution_stage},
            **({"small_rough_regression": rough_receipt(self.contract, self.evaluation_config())}
               if self.is_small_rough_regression else {}),
            "protocol": deepcopy(self.protocol), "native_execution_allowed": False,
            "native_qualification": False, "training_release": False,
            "binding_scope": "captured project source bytes; not complete SDK closure or native execution proof"}


def _stock_actor(state):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models.mlp_model import MLPModel
    # Initialization is discarded by strict load, and consumes no caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        actor = MLPModel(TensorDict({"policy": torch.zeros(1, 35)}, batch_size=[1]),
            {"actor": ["policy"]}, "actor", 6, hidden_dims=[256, 128, 64], activation="elu",
            obs_normalization=False, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"})
    if state is not None:
        actor.load_state_dict(state, strict=True)
    return actor.eval()


def rng_digest(generator):
    return sha(bytes(generator.get_state().cpu().tolist()))


def actor_state_digest(state):
    """Hash named tensor bytes, not incidental pickle serialization."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        data = value.numpy().tobytes()
        header = canonical({"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "bytes": len(data)})
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(data)
    return digest.hexdigest()


class EvaluationActor:
    """Frozen stock Gaussian inference with a private sampling generator.

    The explicit mean+std*epsilon has the stock Gaussian distribution, but does
    not claim bitwise replay of global-RNG Normal.sample or another device.
    Mean and sampled runs share an initial seed/state only; later resets and
    RNG consumption can diverge when their trajectories diverge.
    """

    @classmethod
    def from_prepared(cls, inputs, *, device="cpu"):
        inputs.authenticate()
        actor = _stock_actor(inputs.load_checkpoint()["actor_state_dict"]).to(device)
        value = cls(actor, mode=inputs.protocol["actor_mode"], seed=inputs.protocol["actor_seed"])
        value.captured_checkpoint_sha256 = inputs.inputs["checkpoint"].digest
        return value

    def __init__(self, actor, *, mode, seed):
        import inspect
        import torch
        from rsl_rl.models.mlp_model import MLPModel
        from rsl_rl.modules.distribution import GaussianDistribution
        if (type(actor) is not MLPModel or type(actor.distribution) is not GaussianDistribution
                or actor.obs_dim != 35 or actor.distribution.output_dim != 6
                or actor.obs_normalization or actor.is_recurrent or actor.distribution.std_type != "scalar"):
            raise ValueError("Evaluation supports only the frozen stock scalar Gaussian actor35/action6")
        if mode not in ("mean", "sampled") or type(seed) is not int or seed < 0:
            raise ValueError("Actor mode and nonnegative integer actor seed must be explicit")
        if any(not bool(torch.isfinite(v).all()) for v in actor.state_dict().values()):
            raise ValueError("Actor parameters must be finite")
        if not bool((actor.distribution.std_param.detach() > 0.).all()):
            raise ValueError("Stock Gaussian standard deviation parameters must be strictly positive")
        self.actor, self.mode = actor.eval().requires_grad_(False), mode
        self.device = next(actor.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.seed, self.steps = seed, 0
        self.initial_rng_sha256 = rng_digest(self.generator)
        self.sdk_sources = {name: BoundFile.read(inspect.getfile(cls)) for name, cls in (
            ("rsl_rl/models/mlp_model.py", MLPModel),
            ("rsl_rl/modules/distribution.py", GaussianDistribution))}
        self.last = None
        self.captured_checkpoint_sha256 = None
        self.loaded_state_sha256 = actor_state_digest(self.actor.state_dict())

    def state_digest(self):
        return actor_state_digest(self.actor.state_dict())

    def step(self, policy_observation):
        import torch
        from tensordict import TensorDict
        if (not isinstance(policy_observation, torch.Tensor) or policy_observation.ndim != 2
                or policy_observation.shape[1] != 35 or policy_observation.dtype != torch.float32
                or policy_observation.shape[0] == 0
                or policy_observation.device != self.device or not bool(torch.isfinite(policy_observation).all())):
            raise ValueError("Actor input must be finite float32 [N,35] on its device")
        with torch.no_grad():
            obs = TensorDict({"policy": policy_observation}, batch_size=[len(policy_observation)])
            output = self.actor.mlp(self.actor.get_latent(obs))
            self.actor.distribution.update(output)
            mean, std = self.actor.distribution.params
            if not bool(torch.isfinite(mean).all() & torch.isfinite(std).all() & (std > 0.).all()):
                raise ValueError("Stock Gaussian distribution requires finite mean and positive finite std")
            if self.mode == "mean":
                action = self.actor.distribution.deterministic_output(output)
            else:
                epsilon = torch.randn(mean.shape, dtype=mean.dtype, device=self.device, generator=self.generator)
                action = mean + std * epsilon
            if action.shape != (len(policy_observation), 6) or not bool(torch.isfinite(action).all()):
                raise ValueError("Nonfinite or incorrectly shaped raw actor output")
            self.last = {"mean": mean.clone(), "std": std.clone(), "raw_action": action.clone()}
        self.steps += 1
        return action

    def receipt(self):
        return {"actor_mode": self.mode, "actor_seed": self.seed, "calls": self.steps,
            "captured_checkpoint_sha256": self.captured_checkpoint_sha256,
            "loaded_actor_state_sha256": self.loaded_state_sha256,
            "current_actor_state_sha256": self.state_digest(),
            "generator_device": str(self.device), "initial_rng_sha256": self.initial_rng_sha256,
            "current_rng_sha256": rng_digest(self.generator), "distribution": "stock_Gaussian_scalar_std",
            "action_scope": "raw Gaussian output; controller owns clipping and scaling",
            "sdk_source_bindings": {name: item.receipt() for name, item in self.sdk_sources.items()},
            "sdk_binding_scope": "two inference modules only; not complete transitive SDK closure",
            "pairing_scope": "same initial seed only; later trajectory and RNG consumption may diverge"}


class PreResetLedgerAdapter:
    """Map real production diagnostics into the ledger before opening newborns."""

    def __init__(self, protocol, lane_case_ids, *, reward_profile=SCUT_FLAT_PROFILE):
        self.ledger = FlatEpisodeLedger(protocol, lane_case_ids,
            _canonical_payment_names(reward_profile), reward_profile=reward_profile)

    def _trial(self):
        trial = copy(self.ledger)
        trial.active = deepcopy(self.ledger.active)
        trial.records = self.ledger.records[:]
        trial.attempts, trial.last_ids = self.ledger.attempts[:], self.ledger.last_ids[:]
        return trial

    @staticmethod
    def _values(value, count, tail, name, *, kind="float"):
        import torch
        wanted = (count, *tail)
        if not isinstance(value, torch.Tensor) or value.shape != wanted:
            raise ValueError(f"{name} must have tensor shape {wanted}")
        if (kind == "bool" and value.dtype != torch.bool or kind == "int" and value.dtype != torch.int64
                or kind == "float" and not value.is_floating_point()):
            raise ValueError(f"{name} has the wrong dtype")
        return value.detach().cpu().tolist()

    def begin_initial(self, state, *, root_position_world, yaw_rad, reset_metadata):
        trial = self._trial()
        self._begin(trial, list(range(len(trial.active))), state, root_position_world, yaw_rad, reset_metadata)
        self.ledger = trial

    def _begin(self, ledger, lanes, state, positions, yaws, metadata):
        n = len(ledger.active)
        ids = self._values(state["episode_id"], n, (), "returned episode_id", kind="int")
        ticks = self._values(state["episode_tick"], n, (), "returned episode_tick", kind="int")
        pos = self._values(positions, n, (3,), "returned root position")
        yaw = self._values(yaws, n, (), "returned yaw")
        if not isinstance(metadata, (list, tuple)) or len(metadata) != n:
            raise ValueError("Reset metadata must contain one explicit record per lane")
        for lane in lanes:
            if ticks[lane] != 0:
                raise ValueError("A newly opened episode must have returned tick zero")
            ledger.begin_episode(lane, ids[lane], root_link_position_world=pos[lane],
                yaw_rad=yaw[lane], reset_metadata=metadata[lane])

    def observe_paid(self, extras, reward, done, *, returned_root_position_world=None,
                     returned_yaw_rad=None, reset_metadata=None):
        import torch
        if any(active is None and self.ledger.attempts[lane] < self.ledger.protocol["episodes_per_lane"]
               for lane, active in enumerate(self.ledger.active)):
            raise ValueError("Open every requested episode before consuming its paid transition")
        d, returned = extras["diagnostics"], extras["returned_episode_state"]
        n = len(self.ledger.active)
        mapping = {"episode_id": ("episode_id", (), "int"), "episode_tick": ("episode_tick", (), "int"),
            "reward": ("reward", (), "float"), "done": ("done", (), "bool"),
            "terminated": ("terminated", (), "bool"), "timeout": ("timeout", (), "bool"),
            "boundary": ("boundary", (), "bool"), "learning_mask": ("policy_learning_mask", (), "bool"),
            "bootstrap_mask": ("policy_bootstrap_mask", (), "bool"),
            "measurement_valid": ("measurement_valid_mask", (), "bool"),
            "command_override_active": ("command_override_active", (), "bool"),
            "effective_command": ("effective_command", (3,), "float"), "base_request": ("base_request", (3,), "float"),
            "root_link_position_world": ("root_link_position_world", (3,), "float"),
            "root_com_velocity_body": ("root_COM_velocity_body", (3,), "float"),
            "angular_velocity_body": ("angular_velocity_body", (3,), "float"), "yaw_rad": ("root_yaw_rad", (), "float")}
        # Preserve the same paid pre-reset posture, including terminal frames.
        # Older evaluator diagnostics remain valid and report posture as absent.
        if "gravity" in d:
            mapping["projected_gravity_body"] = ("gravity", (3,), "float")
        if "root_COM_velocity_world" in d:
            mapping["root_com_velocity_world"] = ("root_COM_velocity_world", (3,), "float")
        values = {out: self._values(d[name], n, tail, name, kind=kind) for out, (name, tail, kind) in mapping.items()}
        self._values(reward, n, (), "returned reward")
        self._values(done, n, (), "returned done", kind="bool")
        if not torch.equal(reward, d["reward"]) or not torch.equal(done, d["done"]):
            raise ValueError("Returned reward/done differ from paid pre-reset diagnostics")
        if not isinstance(d["payments"], Mapping) or set(d["payments"]) != set(self.ledger.payment_names):
            raise ValueError(f"Paid diagnostics require exactly {len(self.ledger.payment_names)} source payment columns")
        payments = {name: self._values(value, n, (), name) for name, value in d["payments"].items()}
        if not isinstance(d["reasons"], Mapping) or not d["reasons"]:
            raise ValueError("Paid diagnostics require explicit termination reasons")
        reasons = {name: self._values(value, n, (), name, kind="bool") for name, value in d["reasons"].items()}
        reset = self._values(returned["reset_mask"], n, (), "returned reset_mask", kind="bool")
        ids = self._values(returned["episode_id"], n, (), "returned episode_id", kind="int")
        ticks = self._values(returned["episode_tick"], n, (), "returned episode_tick", kind="int")
        trial, completed, newborn = self._trial(), [], []
        for lane, active in enumerate(self.ledger.active):
            if active is None:
                continue  # Quota-complete lanes do not create extra attempts.
            is_done = values["done"][lane]
            if reset[lane] != is_done or ids[lane] != values["episode_id"][lane] + int(is_done) or ticks[lane] != (0 if is_done else values["episode_tick"][lane]):
                raise ValueError("Returned episode identity/reset boundary differs from the paid transition")
            frame = {key: value[lane] for key, value in values.items()}
            frame.update(payments={key: value[lane] for key, value in payments.items()},
                         termination_reasons=[key for key, value in reasons.items() if value[lane]])
            record = trial.observe(lane, frame)
            if record is not None:
                completed.append(record)
                if trial.attempts[lane] < trial.protocol["episodes_per_lane"]:
                    newborn.append(lane)
        if newborn:
            self._begin(trial, newborn, returned, returned_root_position_world, returned_yaw_rad, reset_metadata)
        self.ledger = trial
        return completed
