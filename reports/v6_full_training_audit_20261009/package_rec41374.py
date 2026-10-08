from pathlib import Path
import json,subprocess,hashlib,shutil,datetime,tarfile
root=Path(__file__).resolve().parents[2]
reports=root/'reports/v6_full_training_audit_20261009'
out=root/'models/v6_recovery_priority_handoff_20261009'
assert not out.exists();out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source='6f07ec298e01d2686216787cfba34b394151f1aa'
verify=json.loads((reports/'rec41374_independent_verification.json').read_text())
shutil.copytree(root/'models/v6_recovery_candidate_41374_20261009',out/'model')
shutil.copyfile(root/'docs/V6_RECOVERY_PRIORITY_MODEL_HANDOFF_20261009.md',out/'handoff.md')
paths=list(verify['recovery_source_bindings'])+['src/wheeled_tasks/chassis/training_schedule.py','src/wheeled_tasks/chassis/root_kinematics.py','contracts/v6_recovery_profiles_v2.json','contracts/v6_height_lookup_v1.json','contracts/v6_wheel_response_control_200hz_v1.json']
for name in paths:
 raw=subprocess.check_output(['git','show',source+':'+name],cwd=root)
 if name in verify['recovery_source_bindings']:assert hashlib.sha256(raw).hexdigest()==verify['recovery_source_bindings'][name]['source_sha256']
 p=out/'recovery_reference'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
contract=json.loads((out/'model/contract.json').read_text())
for key in ['profiles','height_lookup']:
 p=out/'recovery_reference'/contract['recovery_training'][key+'_file'];assert sha(p)==contract['recovery_training'][key+'_sha256']
for name in ['rec41374_download.json','rec41374_independent_verification.json','model_selection_audit.md']:
 shutil.copyfile(reports/name,out/name)
(out/'README.md').write_text('# 恢复优先候选 REC14169 / 41374\n\n主模型 `model/policy.onnx`；部署对接见 `handoff.md`。\n\n这是最新封存恢复候选，不是已经验证的恢复+高速综合最强模型。恢复前段由配套FSM完成，策略负责接管；普通课程仅 |vx|≤1m/s、|yaw|≤1rad/s。高速/实车能力尚未核验。\n\n`recovery_reference/` 是固定训练源码与profile的移植参考，仿真专属真值代码不可直接作为实车传感器。数值/来源证据随包提供。\n\n在此目录运行 `sha256sum -c SHA256SUMS` 核对完整交付。\n')
manifest=dict(schema='recovery-priority-handoff-v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source,onnx_sha256=sha(out/'model/policy.onnx'),qualified=None,hardware_deployment_ready=False,scope='sealed_recovery_candidate_with_frozen_FSM_reference_not_joint_highspeed_qualification',files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
(out/'handoff_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'SHA256SUMS'))
r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=out,capture_output=True,text=True);assert r.returncode==0,r.stderr
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
result=dict(package=str(out),archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,verified_file_count=len(r.stdout.splitlines()),onnx_sha256=manifest['onnx_sha256'],qualified=None)
(reports/'rec41374_delivery_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
