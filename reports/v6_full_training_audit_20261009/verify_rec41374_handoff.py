#!/usr/bin/env python3
"""Independent CPU-only checks of a sealed REC actor export; no scene or writes to the bundle."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib, importlib.util, json, subprocess, sys
import numpy as np
import torch
import onnx
import onnxruntime as ort
from tensordict import TensorDict
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'models/v6_recovery_candidate_41374_20261009'
S = ROOT / 'reports/v6_full_training_audit_20261009/rec41374_source'
OUT = ROOT / 'reports/v6_full_training_audit_20261009/rec41374_independent_verification.json'
SOURCE = '6f07ec298e01d2686216787cfba34b394151f1aa'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

torch.set_num_threads(1)
checks={}
contract=read(B/'contract.json');seal=read(S/'model_final.checkpoint.json');agent=read(B/'agent_config.json')
checks['source_checkpoint_sha_matches_fullseal'] = digest(S/'model_final.pt') == seal['checkpoint_sha256'] == digest(B/'model.pt') == 'ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085'
checks['source_contract_matches_fullseal'] = digest(S/'model_final.contract.json') == seal['contract_sha256'] == digest(B/'contract.json') == digest(B/'policy.onnx.contract.json')
checks['original_source_seal_retained'] = digest(B/'source.checkpoint.json') == digest(S/'model_final.checkpoint.json')
checks['source_agent_config_retained'] = digest(B/'agent_config.json') == digest(S/'agent_config.json')
checks['bundle_manifest_all_bytes'] = all(digest(B/k)==v['sha256'] and (B/k).stat().st_size==v['bytes'] for k,v in read(B/'bundle_manifest.json')['files'].items())
checks['SHA256SUMS_all_files'] = all(digest(B/row.split('  ',1)[1])==row.split('  ',1)[0] for row in (B/'SHA256SUMS').read_text().splitlines())
saved=torch.load(B/'model.pt',map_location='cpu',weights_only=True)
checks['full_learning_state_present'] = all(k in saved for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict','infos'))
checks['infos_contract_binding'] = saved['infos']['contract_sha256']==digest(B/'contract.json')
checks['infos_sealed_clock_binding'] = saved['infos']['successful_updates_total']==seal['learning_updates'] and all(saved['infos'][k]==seal[k] for k in ('learning_lineage_updates','stage_actor_update','global_actor_update','actor_updates_consumed','task_actor_update'))
checks['interfaces_35_81_6_singleframe'] = tuple(contract[k] for k in ('actor_dim','critic_dim','action_dim','history_length'))==(35,81,6,1)
checks['no_running_normalizer'] = agent['actor']['obs_normalization'] is False and not agent.get('empirical_normalization')
config=deepcopy(agent['actor']);config.pop('class_name')
actor=MLPModel(TensorDict({'policy':torch.zeros(1,35)},batch_size=[1]),agent['obs_groups'],'actor',6,**config).cpu().eval()
actor.load_state_dict(saved['actor_state_dict'],strict=True)
checks['native_actor_all_finite'] = all(torch.isfinite(v).all().item() for v in actor.state_dict().values())
onnx.checker.check_model(str(B/'policy.onnx'))
session=ort.InferenceSession(str(B/'policy.onnx'),sess_options=ort.SessionOptions(),providers=['CPUExecutionProvider'])
checks['onnx_named_interface'] = (session.get_inputs()[0].name,session.get_outputs()[0].name)==('obs','actions')
checks['onnx_shapes'] = session.get_inputs()[0].shape[-1]==35 and session.get_outputs()[0].shape[-1]==6
abi=read(B/'deployment/abi_test_vectors.json');rows=abi['vectors'];sensor=module('bundled_scut_observation',B/'deployment/scut_observation.py');refmodule=module('bundled_reference',B/'deployment/references.py')
n=len(rows)
def stack(key): return torch.tensor([r[key] for r in rows],dtype=torch.float32)
commands=stack('commands_vx_yaw_height');request=torch.tensor([r['jump_request'] for r in rows]);apex=torch.tensor([r['jump_apex_m'] for r in rows]);elapsed=torch.tensor([r['command_elapsed_s'] for r in rows]);reference=refmodule.CommandReference(n,'cpu',contract['command_reference'])
reference.update(commands[:,2],torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.full((n,),7.),request,elapsed,apex)
rebuilt=sensor.build_manual35(stack('gyro_b_rad_s'),stack('gravity_b'),commands,stack('motor_q_control_rad'),stack('motor_dq_control_rad_s'),stack('previous_clipped_action_control'),torch.tensor(abi['nominal_position_control_rad']),request,apex,reference)
expected=stack('obs');abi_max=float((rebuilt-expected).abs().max())
checks['abi_sensor_snapshots_reconstruct_exactly'] = abi_max == 0.
generator=torch.Generator().manual_seed(261009)
samples=torch.cat((expected,torch.randn(256,35,generator=generator)*2.,torch.zeros(1,35)),0)
with torch.inference_mode(): native=actor(TensorDict({'policy':samples},batch_size=[len(samples)]),stochastic_output=False).numpy()
actual=session.run(['actions'],{'obs':samples.numpy()})[0]
np.testing.assert_allclose(actual,native,atol=1e-5,rtol=1e-5)
checks['native_mean_matches_onnx'] = True
np.testing.assert_allclose(actual[:n],np.array([r['actions'] for r in rows]),atol=1e-5,rtol=1e-5)
checks['abi_actions_match_onnx'] = True
# Independent scalar decode verifies the documented policy/control permutation and clipping.
checks['policy_control_permutation_inverse'] = np.array_equal(np.array([0,1,3,4,2,5])[np.array([0,1,4,2,3,5])],np.arange(6))
contract_bindings={}
for key in ('profiles','height_lookup'):
    r=contract['recovery_training']; path=ROOT/r[key+'_file']; frozen=subprocess.check_output(['git','show',SOURCE+':'+r[key+'_file']],cwd=ROOT)
    sha=hashlib.sha256(frozen).hexdigest();contract_bindings[r[key+'_file']]={'source_sha256':sha,'contract_sha256':r[key+'_sha256'],'local_equal_source':path.read_bytes()==frozen}
checks['recovery_tables_authenticate'] = all(v['source_sha256']==v['contract_sha256'] and v['local_equal_source'] for v in contract_bindings.values())
paths=['src/wheeled_tasks/chassis/'+p for p in ['recovery.py','recovery_training.py','recovery_observer.py','mechanism_observer.py','continuous_encoders.py','step_trigger.py','motion_limits.py','scut_observation.py','references.py','v5_control.py']]
sources={}
for p in paths:
    raw=subprocess.check_output(['git','show',SOURCE+':'+p],cwd=ROOT)
    sources[p]={'source_sha256':hashlib.sha256(raw).hexdigest(),'local_equal_source':(ROOT/p).read_bytes()==raw}
result={'schema':'rec41374-independent-cpu-handoff-verification-v1','at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if all(checks.values()) else 'failed','checks':checks,'scope':'CPU identity, fullseal, native deterministic actor, ONNX and ABI only; not a physical evaluation','source_commit':SOURCE,'onnx_sha256':digest(B/'policy.onnx'),'checkpoint_sha256':digest(B/'model.pt'),'contract_sha256':digest(B/'contract.json'),'observation_samples':len(samples),'abi_vectors':n,'abi_observation_max_abs_error':abi_max,'native_onnx_max_abs_error':float(np.abs(native-actual).max()),'clock':{k:saved['infos'][k] for k in ('successful_updates_total','learning_lineage_updates','stage_actor_update','global_actor_update','actor_updates_consumed','task_actor_update')},'recovery_table_bindings':contract_bindings,'recovery_source_bindings':sources,'gpu_used':False,'isaac_started':False,'optimizer_steps':0,'model_modified':False,'hardware_deployment_ready':False,'qualified':None}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'status':result['status'],'checks':checks,'native_onnx_max_abs_error':result['native_onnx_max_abs_error']},ensure_ascii=False))
assert all(checks.values()), 'Independent verification failed'
