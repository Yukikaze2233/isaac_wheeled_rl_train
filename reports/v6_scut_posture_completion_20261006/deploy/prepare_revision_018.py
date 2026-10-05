#!/usr/bin/env python3
"""Authenticate direct v15→v18 against four latest own-domain full seals."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

OUT = Path(__file__).resolve().parent
BASE = "/home/kaiser/robot-rl-sim60"
RUN = BASE + "/experiments/v6-task-campaign-20261001T110045Z-282358"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", "2222", "kaiser@192.168.64.234"]
SOURCE_PLAN_SHA = "3d4ceb76c212fb47b3380d980b26afda19d0203785f556c5aa2fb44b5b353e61"
DOMAINS = ("recovery", "speed", "terrain", "jump", "stepjump")
PARENTS = dict(recovery=2294, speed=5063, terrain=2186, jump=5552)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_observed(startup):
    receipt = startup["runtime_preflight"]
    assert receipt["status"] == "runtime_configuration_passed"
    return receipt["observed"]


def load_evidence(native_path=None):
    """Require every admission receipt to bind the same final frozen archive."""
    paths = dict(source_candidate=OUT / "source_candidate_018.json",
        clean_archive_tests=OUT.parent / "verification/clean_archive_tests.json",
        remote_clean_30_roles=OUT / "remote_clean_30_roles_018.json",
        native_population=Path(native_path) if native_path is not None else OUT.parent / "verification/native_population_018.json")
    meta, tests, preflight, native = (read(paths[name]) for name in paths)
    for item in (tests, preflight, native):
        assert item["status"] == "passed"
        assert item["source_commit"] == meta["source_commit"]
        assert item["archive_sha256"] == meta["archive_sha256"]
    assert tests["exitcode"] == preflight["exitcode"] == 0 and preflight["role_count"] == 30
    assert sha(meta["archive_local"]) == meta["archive_sha256"]
    with tarfile.open(meta["archive_local"], "r:gz") as archive:
        assert archive.pax_headers.get("comment") == meta["source_commit"]
        def archived(name):
            return json.loads(archive.extractfile(name).read())
        plan = archived("contracts/v6_task_campaign_v18.json")
        assert hashlib.sha256(archive.extractfile("contracts/v6_task_campaign_v18.json").read()).hexdigest() == meta["plan_sha256"]
        assert hashlib.sha256(archive.extractfile("contracts/v6_task_campaign_migration_v18.json").read()).hexdigest() == meta["manifest_sha256"]
        pins = {row["name"]: row["sha256"] for row in plan["domains"]}
        assert set(native["domains"]) == set(DOMAINS)
        # Validate the source's actual observed runtime schema, including
        # geometry/population/caps and independent timers; booleans alone fail.
        sys.path.insert(0, str(Path(meta["clean_root"]) / "src"))
        sys.path.insert(0, str(Path(meta["clean_root"]) / "scripts"))
        from wheeled_tasks.chassis.preflight import validate_runtime_state
        from probe_scut_curriculum import source_snapshot
        from recovery_prefix_support import BASELINE_PT_SHA, BASELINE_CONTRACT_SHA
        for domain in DOMAINS:
            probe = native["domains"][domain]
            config = archived(next(row["contract"] for row in plan["domains"] if row["name"] == domain))
            binding = config["design_preflight"]
            design = archived(binding["file"])
            role = design["stages"][binding["stage"]]["roles"]["train"]
            rule = role["invocation"]["seed"]
            seed = rule if type(rule) is int else rule["root_seed"] + probe["block_index"]
            assert probe["status"] == "engineering_probe_completed" and probe["domain"] == domain
            assert probe["schema"] == "scut-curriculum-native-probe-v18" and probe["run_role"] == "train"
            assert probe["num_envs"] == role["invocation"]["num_envs"] == 4096
            assert probe["seed"] == plan["seed"] == seed and probe["block_index"] == 0
            assert probe["target_contract_sha256"] == pins[domain] == meta["worker_hashes"][domain]
            assert probe["source_files_unchanged"] is probe["target_contract_unchanged"] is True
            assert probe["source_files_changed"] == []
            assert probe["forced_engineering_actor_update"] == config["ordinary_posture_curriculum"]["origin_actor_update"]
            source_files = source_snapshot(config)
            assert probe["source_files_sha256"] == source_files, "Probe source snapshot belongs to a different candidate"
            for name, digest in source_files.items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            checkpoint = probe["checkpoint"]
            assert checkpoint["checkpoint_sha256"] == BASELINE_PT_SHA
            assert checkpoint["checkpoint_contract_sha256"] == BASELINE_CONTRACT_SHA
            assert checkpoint["seal_and_infos_authenticated"] is checkpoint["actor_only_deterministic_inference"] is True
            assert probe["formal_updates_consumed"] == probe["optimizer_steps"] == 0 and probe["qualification"] is None
            assert probe["policy_steps"] == 600 and probe["physics_steps"] == 2400
            observed = runtime_observed(probe["startup"])
            expected = deepcopy(role["runtime"])
            if expected.get("seed") == {"source": "effective_seed"}:
                expected["seed"] = seed
            validate_runtime_state(expected, observed)
            assert observed["seed"] == seed and observed["modules"]["ordinary_posture_curriculum"] is True
            proof = probe["ordinary_posture_native_proof"]
            assert proof["actual_native_velocity_delta_observed"] is proof["actual_root_rotation_observed"] is True
            assert proof["initial"]["native_root_quaternion_readback_passed"] is proof["initial"]["geometry_bound"] is True
            assert proof["initial"]["birth_selected_count"] > 0 and proof["initial"]["birth_abs_angle_max_rad"] > 0
            assert proof["final"]["push_events"] > 0 and proof["final"]["push_velocity_delta_max_m_s"] > 0
            groups = probe["startup"]["scene_group_counts"]
            assert sum(groups.values()) == 4096 and min(groups.values()) > 0
            timing = probe["control_timing"]
            assert timing["pd_updates"] == timing["feedback_refreshes"] == 4
            assert timing["target_decodes"] == 1 and (timing["physics_dt"], timing["policy_dt"]) == (.005, .02)
    return dict(source=meta, tests=tests, preflight=preflight, native=native,
        evidence_sha256={name: sha(path) for name, path in paths.items()})


REMOTE = r'''
import hashlib,json,pathlib,sys,tempfile
from copy import deepcopy
from datetime import datetime,timezone
r=pathlib.Path(RUN);src=pathlib.Path(META['source_remote'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(args):
 for value in args[1:]:
  if value in (b'-c',b'-m'):return None
  if value.endswith(b'.py'):return pathlib.Path(value.decode())
 return None
job=read(r/'job.json');assert not pathlib.Path('/proc/'+str(job['pid'])+'/cmdline').exists(),'Previous owner still live'
oldcmd=job['command'];old=pathlib.Path(oldcmd[oldcmd.index('--contract')+1]).parent.parent
assert old.name=='source_8f05f19c8dba'
assert pathlib.Path(oldcmd[oldcmd.index('--contract')+1])==old/'contracts/v6_task_campaign_v15.json'
assert sha(old/'contracts/v6_task_campaign_v15.json')==SOURCE_PLAN_SHA
assert not (r/'source_revision_017.json').exists() and not (r/'scut_curriculum_launch_017.json').exists(),'Unlaunched v17 must not become deployed predecessor'
owned=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=p.read_bytes().split(b'\0');script=entry(argv)
  if script and script.name in ('run_task_campaign.py','train_chassis.py','evaluate_chassis.py','chassis_remote_job.py') and any(x==str(r).encode() or x==str(r/'train').encode() or x.startswith(str(r/'train').encode()+b'/') for x in argv):owned.append(str(p))
 except (OSError,PermissionError):pass
assert not owned,owned
state=read(r/'train/campaign_state.json');account=read(r/'train/consumption.json')
assert state['contract_sha256']==account['contract_sha256']==SOURCE_PLAN_SHA
assert state['consumed_updates']==account['consumed_updates']==29205
assert state['actor_updates_consumed']==account['actor_updates_consumed']==29075
assert account['rollback_refunds_consumed_updates'] is False
sys.path.insert(0,str(src/'src'))
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count,checkpoint_contract_path
from wheeled_tasks.chassis.ordinary_posture_revision import reopen_completed_domains,ORIGINS,revised_plan,revised_manifest,validate_frozen_workers,source_binding
from wheeled_tasks.chassis.task_campaign import validate_task_resume,validate_plan_migration,account_updates
source=read(old/'contracts/v6_task_campaign_v15.json');target=read(src/'contracts/v6_task_campaign_v18.json')
assert target==revised_plan(src) and read(src/'contracts/v6_task_campaign_migration_v18.json')==revised_manifest(src)
assert sha(src/'contracts/v6_task_campaign_v18.json')==META['plan_sha256'] and sha(src/'contracts/v6_task_campaign_migration_v18.json')==META['manifest_sha256']
assert len(validate_frozen_workers(src))==5
assert account_updates(source,account['attempts'])=={k:account[k] for k in account_updates(source,account['attempts'])}
validate_plan_migration(source,target,r/'train')
parents={}
for domain,actor in PARENTS.items():
 paths=[]
 for p in (r/'train'/domain).glob('attempt_*/train/**/*.checkpoint.json'):
  if any(x.endswith('.pending') for x in p.parts):continue
  seal=read(p);cp=p.with_suffix('').with_suffix('.pt')
  assert type(seal.get('learning_updates')) is int and seal['learning_updates']>=0
  paths.append((seal['learning_updates'],cp.name=='model_final.pt',cp,seal))
 assert paths,domain+' has no sealed state'
 _,_,parent,seal=max(paths,key=lambda x:x[:2])
 assert checkpoint_update_count(parent)==seal['learning_updates']<=account['consumed_updates']
 assert seal['task_campaign']['domain']==domain and seal['sealed'] is True
 assert parent.resolve().is_relative_to(r/'train'/domain)
 contract=checkpoint_contract_path(parent);own_file,own_sha=source_binding(domain,src)
 assert sha(contract)==own_sha and read(contract)==read(src/own_file)
 assert sha(parent)==seal['checkpoint_sha256'] and seal['contract_sha256']==own_sha
 worker=next(row for row in target['domains'] if row['name']==domain)
 validate_task_resume(read(contract),read(src/worker['contract']))
 assert type(seal['task_actor_update']) is int and seal['task_actor_update']==actor
 if domain in ORIGINS:
  prior=state['domains'][domain]
  assert prior['status'] in ('domain_schedule_finished_unassessed','domain_review_finished_unassessed')
  assert prior['domain_qualified'] is None and pathlib.Path(prior['candidate_checkpoint']).resolve()==parent.resolve()
  assert prior['candidate_checkpoint_sha256']==sha(parent)
 parents[domain]=dict(path=str(parent),sha256=sha(parent),seal=seal,contract=str(contract),contract_sha256=sha(contract))
replayed=deepcopy(state);records=deepcopy(account['attempts'])
reopening=reopen_completed_domains(replayed,records,source,target,r/'train')
assert records==account['attempts'] and account_updates(target,records)==account_updates(source,records)
assert all(replayed[k]==state[k] for k in ('consumed_updates','actor_updates_consumed','domain_consumed_updates','evaluations'))
assert all(replayed['domains'].get(k)==state['domains'].get(k) for k in ('flat','jump','stepjump'))
assert all(replayed['domains'][name]['completion_history'][-1]==dict(plan_revision='v15',record=state['domains'][name]) for name in ORIGINS)
once=deepcopy(replayed);assert reopen_completed_domains(replayed,records,source,target,r/'train')==reopening and replayed==once
before=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),job=job,campaign_state=state,consumption=account,
 actual_parents=parents,authenticated_reopening=reopening,source_commit=META['source_commit'],
 evidence_sha256=EVIDENCE_SHA,scope='four_latest_own_domain_full_seals_three_atomic_reopenings_no_rollback_no_refund_unassessed')
proof=r/'scut_posture_predecessor_018.json'
if proof.exists():
 prior=read(proof);assert all(prior[k]==before[k] for k in before if k!='observed_at_utc')
else:
 with proof.open('x') as f:json.dump(before,f,indent=2);f.write(chr(10))
command=list(oldcmd);command[command.index(str(old/'scripts/run_task_campaign.py'))]=str(src/'scripts/run_task_campaign.py')
command[command.index('--contract')+1]=str(src/'contracts/v6_task_campaign_v18.json')
receipt=dict(schema='task-plan-revision-v18',previous_job_sha256=sha(r/'job.json'),command=command,
 source_commit=META['source_commit'],old_source_root=str(old),new_source_root=str(src),
 new_contract_relative_path='contracts/v6_task_campaign_v18.json',old_contract_sha256=SOURCE_PLAN_SHA,
 new_contract_sha256=sha(src/'contracts/v6_task_campaign_v18.json'),source_archive=META['archive_remote'],source_archive_sha256=META['archive_sha256'])
sys.path.insert(0,str(src/'scripts'))
from chassis_remote_job import validate_source_revision
with tempfile.NamedTemporaryFile(mode='w',dir=r,prefix='source_revision_018_validation_',suffix='.pending',delete=False) as f:
 json.dump(receipt,f);pending=pathlib.Path(f.name)
try:validate_source_revision(r,command,job,pending)
finally:pending.unlink()
path=r/'source_revision_018.json'
if path.exists():assert read(path)==receipt
else:
 with path.open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
print(json.dumps(dict(status='passed',revision=receipt,predecessor=read(proof))))
'''


def remote_code(evidence):
    values = dict(RUN=RUN, META=evidence["source"], SOURCE_PLAN_SHA=SOURCE_PLAN_SHA,
                  PARENTS=PARENTS, EVIDENCE_SHA=evidence["evidence_sha256"])
    return "\n".join(name + "=" + repr(value) for name, value in values.items()) + "\n" + REMOTE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-remote-code", action="store_true", help="Print authenticated script; do not connect")
    args = parser.parse_args()
    evidence = load_evidence();code = remote_code(evidence)
    if args.emit_remote_code:
        print(code);return
    command = "source " + shlex.quote(BASE + "/bin/sim60-runtime.sh") + " && export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && " + shlex.join([BASE + "/env/bin/python", "-B", "-"])
    result = subprocess.run(SSH + [shlex.join(["bash", "-lc", command])], input=code,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr[-3000:]
    proof = json.loads(result.stdout)
    for name, value in (("source_revision_018.json", proof["revision"]), ("predecessor_state_018.json", proof["predecessor"])):
        (OUT / name).write_text(json.dumps(value, indent=2) + "\n")
    proof.update(verified_at_utc=datetime.now(timezone.utc).isoformat(), status="frozen_source_and_four_own_sealed_parents_authenticated_not_started")
    (OUT / "predeploy_018.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(status=proof["status"], parents=proof["predecessor"]["actual_parents"], consumed=proof["predecessor"]["consumption"]["consumed_updates"])))


if __name__ == "__main__":
    main()
