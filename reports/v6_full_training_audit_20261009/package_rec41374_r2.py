from pathlib import Path
import json,subprocess,hashlib,shutil,datetime,tarfile
root=Path(__file__).resolve().parents[2]
reports=root/'reports/v6_full_training_audit_20261009'
out=root/'models/v6_recovery_priority_handoff_20261009_r2'
assert not out.exists();out.mkdir()
original=root/'models/v6_recovery_priority_handoff_20261009.tar.gz'
original_digest=hashlib.sha256(original.read_bytes()).hexdigest()
assert original_digest=='c7122f8310dcd3b22ef6f7c94ecb1db7c86f67f91d11cbf12fa7cd99420053fe'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source='6f07ec298e01d2686216787cfba34b394151f1aa'
verify=json.loads((reports/'rec41374_independent_verification.json').read_text())
export_root=root/'models/v6_recovery_candidate_41374_20261009'
omitted=['model.pt','completion.json']
shutil.copytree(export_root,out/'model',ignore=shutil.ignore_patterns(*omitted))
original_export_manifest=json.loads((export_root/'bundle_manifest.json').read_text())
subset_manifest=dict(schema='chassis-candidate-bundle-v1',projection='inference_handoff_subset',source_bundle_manifest_sha256=sha(export_root/'bundle_manifest.json'),omitted_source_files={n:original_export_manifest['files'][n] for n in omitted},files={n:v for n,v in original_export_manifest['files'].items() if n not in omitted})
(out/'model/bundle_manifest.json').write_text(json.dumps(subset_manifest,ensure_ascii=False,indent=2)+'\n')
(out/'model/SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(out/'model'))+'\n' for p in sorted((out/'model').rglob('*')) if p.is_file() and p!=out/'model/SHA256SUMS'))
assert subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=out/'model',capture_output=True).returncode==0

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
for name in ['native_comparison_addendum.json','native_comparison_addendum.md']:
 shutil.copyfile(reports/'model_compare'/name,out/name)
old=root/'models/v6_recovery_candidate_38000_20261008'
new=root/'models/v6_recovery_candidate_41374_20261009'
old_interface=json.loads((old/'deployment/interface.json').read_text())
new_interface=json.loads((new/'deployment/interface.json').read_text())
old_interface.pop('identity');new_interface.pop('identity')
assert old_interface==new_interface
assert (old/'contract.json').read_bytes()==(new/'contract.json').read_bytes()
assert (old/'deployment/v6_wheel_response_control_200hz_v1.json').read_bytes()==(new/'deployment/v6_wheel_response_control_200hz_v1.json').read_bytes()
comparison=dict(schema='REC-38000-41374-common-interface-v1',old_model=str(old/'policy.onnx'),old_model_sha256=sha(old/'policy.onnx'),new_model_sha256=sha(new/'policy.onnx'),interface_except_identity_equal=True,contract_byte_equal=True,control_contract_byte_equal=True,old_onnx_included=False)
(out/'common_interface_verification.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
(out/'README.md').write_text('# 最新恢复候选 REC14169 / 41374：r2交付\n\n本包主模型 `model/policy.onnx` 是 REC41374；完整对接见 `handoff.md`，原生两场景回放增补见 `native_comparison_addendum.md/.json`。\n\n最新不等于最好：两候选的本次左侧90°自起链均7.46s完成，旧REC38000制动明显更稳。按恢复制动优先回放旧38000，其绝对路径在handoff.md；本包不含旧38000模型，不可将旧模型结论记到本包模型。最新高速旋转曝光不足、其它倒地/terrain/普通爬阶/主动跳阶未测，未授实车资格。\n\n`recovery_reference/` 为原REC固定训练源码与profile的移植参考；仿真真值不作为实车传感器。普通爬阶和主动跳阶是两个独立课程与结果，不共享成功信用。\n\n在此目录运行 `sha256sum -c SHA256SUMS` 核对所有文件。原r1包与原模型字节保持不变。\n')

manifest=dict(schema='recovery-priority-handoff-v2',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source,onnx_sha256=sha(out/'model/policy.onnx'),qualified=None,hardware_deployment_ready=False,scope='sealed_REC41374_with_frozen_FSM_reference_and_two_scene_comparison_no_hardware_qualification',comparison_replay_preference='REC38000_for_recovery_and_braking_limited_observation',prior_archive_sha256=original_digest,files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
(out/'handoff_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'SHA256SUMS'))
r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=out,capture_output=True,text=True);assert r.returncode==0,r.stderr
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
result=dict(package=str(out),archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,verified_file_count=len(r.stdout.splitlines()),onnx_sha256=manifest['onnx_sha256'],qualified=None)
assert sha(original)==original_digest
assert sha(out/'model/policy.onnx')=='63574953c97aad374fa1554c53486cd540a3019857af72e0c06cc010214299bf'
assert not list(out.rglob('*.pt'))
result['original_archive_unchanged']=True
result['original_archive_sha256']=original_digest
(reports/'rec41374_delivery_receipt_r2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
