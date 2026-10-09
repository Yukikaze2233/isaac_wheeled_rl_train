"""Real cancellation flow with synthetic file/proc dependencies and no signals.

The fixture is also imported by the source031 coordinator integration test.
Archive/PIDFD/native-admission services are synthetic; safety/event/JSON/locks
flow through the actual helper, not a fabricated same-shaped receipt.
"""
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

FILE = Path(__file__).resolve().parents[1] / "cancel_waiting_source030.py"
SPEC = importlib.util.spec_from_file_location("cancel_waiting_source030_test_subject", FILE)
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def publish(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")


class Fixture:
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "experiment"
        self.root.mkdir()
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(subject, "ROOT", self.root)
        monkeypatch.setattr(subject, "NEW_FILE_COUNT", 3)
        self.old, self.new, self.next = [self.root / ("source_serial_dispatch" + v + "_20261009") for v in ("029", "030", "031")]
        for folder in (self.old, self.new, self.next, self.root / "source030_queue", self.root / "source031_deploy", self.root / "train"):
            folder.mkdir(parents=True)
        plan_file = self.old / "contracts/v6_task_campaign_v26.json"
        write(plan_file, {"synthetic": True})
        self.claim = self.root / "budget/v6-task-domains-20261001.account.json"
        write(self.claim, dict(contract_sha256=subject.pin(plan_file)["sha256"], consumed_updates=42000, actor_updates_consumed=41820))
        write(self.root / "train/campaign_state.json", dict(contract_sha256=subject.pin(plan_file)["sha256"]))
        self.coordinator_file = self.root / "coordinate_source030_0478bfe3.py"
        self.coordinator_file.write_text("# synthetic coordinator, never executed\n")
        monkeypatch.setattr(subject, "OLD_QUEUE_SHA", subject.pin(self.coordinator_file)["sha256"])
        archive = self.root / ("source031-" + subject.NEW_COMMIT + "-umask0022.tar.gz")
        archive.write_text("synthetic archive service input, never extracted\n")
        monkeypatch.setattr(subject, "NEW_ARCHIVE_SHA", subject.pin(archive)["sha256"])
        helper = self.root / "deploy_source031_663efe61.py"
        helper.write_text("import json\ndef archive_files(archive, commit, *, verify_tree):\n    return json.loads((verify_tree.parent/'source031_deploy/stage.json').read_text())['files']\n")
        monkeypatch.setattr(subject, "NEW_HELPER_SHA", subject.pin(helper)["sha256"])
        inventory = {}
        for name in ("scripts/example.py", "contracts/example.json", "src/example.py"):
            destination = self.next / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("synthetic source identity\n")
            inventory[name] = subject.pin(destination)["sha256"]
        stage = dict(schema="source031-static-stage-v1", status="staged_only", source_commit=subject.NEW_COMMIT,
            source_archive=subject.pin(archive), helper=subject.pin(helper), files=inventory,
            old_source_root=str(self.old), new_source_root=str(self.next), dynamic_parent_bound=False, PPO_updates=0, qualification=None)
        write(self.root / "source031_deploy/stage.json", stage)
        write(self.root / "source031_deploy/preflight.log", dict(status="passed", roles=30))
        write(self.root / "source031_deploy/preflight.json", dict(status="CPU_preflight_passed", stage=subject.pin(self.root / "source031_deploy/stage.json"),
            log=subject.pin(self.root / "source031_deploy/preflight.log"), PPO_updates=0, qualification=None))
        self.repin_new()
        self.admitted = dict(root=self.root, old=self.old, new=self.new)
        for key, relative in (('controller','scripts/hold_completed_speed_boundary.py'),('publisher','scripts/queue_task_campaign_continuation.py')):
            file = self.new / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# synthetic native dependency, never executed\n")
            self.admitted[key] = subject.pin(file)
        self.admitted['stage'] = subject.pin(self.root / "source031_deploy/stage.json")
        self.identities = {}
        for role, (pid, start) in subject.PINS.items():
            if role == 'controller':
                command=[sys.executable,'-B',self.admitted['controller']['file'],'--source-root',str(self.old),'--run-root',str(self.root/'train'),
                    '--intent',str(self.root/'source030_speed_boundary.json'),'--apply','--timeout','604800']
                ppid=subject.PINS['coordinator'][0]
            elif role == 'coordinator':
                command=[sys.executable,'-B',str(self.coordinator_file),'--run-root',str(self.root),'--apply'];ppid=1
            else:
                script={'wrapper':'chassis_remote_job.py','supervisor':'run_task_campaign.py','worker':'train_chassis.py'}[role]
                command=[sys.executable,'-B',str(self.old/'scripts'/script),'--run-root',str(self.root)]
                ppid=1 if role=='wrapper' else subject.PINS['wrapper' if role=='supervisor' else 'supervisor'][0]
            self.identities[pid]=dict(pid=pid,starttime=start,ppid=ppid,entry=command[2],command=command)
        self.training={role:deepcopy(self.identities[subject.PINS[role][0]]) for role in ('wrapper','supervisor','worker')}
        old_intent=dict(schema='source030-once-normal-speed-boundary-queue-v1',source_commit='81ad124554c4ab45cc274e2d482d46c76ba125b0',
            inputs={k:v for k,v in self.admitted.items() if k not in ('root','old','new')},coordinator=subject.pin(self.coordinator_file),
            controller_wait_seconds=604800,retries=0)
        write(self.root/'source030_queue/queue.intent.json',old_intent)
        monkeypatch.setattr(subject,'OLD_INTENT_SHA',subject.pin(self.root/'source030_queue/queue.intent.json')['sha256'])
        self.waiting_status=dict(status='waiting_for_normal_speed_boundary',phase='controller',intent=subject.pin(self.root/'source030_queue/queue.intent.json'),qualification=None)
        write(self.root/'source030_queue/status.json',self.waiting_status)
        write(self.root/'source030_queue/controller.started.json',dict(command=self.identities[subject.PINS['controller'][0]]['command'],pid=subject.PINS['controller'][0]))
        self.directory=self.root/'train/speed/attempt_003/train'
        self.directory.mkdir(parents=True)
        self.set_actor(600)
        self.native_receipt=dict(selection=dict(fresh_domain_actor_updates=0),allowance=dict(requested_actor_updates=14897))
        self.signals=[];self.opens=[];self.on_open=None;self.on_signal=None
        self.controller_lock=self.root/'train/native_speed_boundary_controller.lock'
        self.controller_lock.write_text('')
        self.lock_fd=self.controller_lock.open('r')
        fcntl.flock(self.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        monkeypatch.setattr(subject,'lock_owner',lambda path:[subject.PINS['controller'][0]] if subject.PINS['controller'][0] in self.identities else [])
        self.queue=SimpleNamespace(SCHEMA=old_intent['schema'],COMMIT=old_intent['source_commit'],unchanged=self.unchanged,native_publish=lambda path:publish)
        self.boundary=SimpleNamespace(process_identity=lambda pid:deepcopy(self.identities.get(pid)),_state=lambda pid:'R',admission=self.admission)
        self.modules=(self.queue,self.admitted,self.boundary,self.open_fd,self.signal_fd)
        self.args=SimpleNamespace(run_root=self.root,new_stage_sha256=subject.NEW_STAGE_SHA,new_preflight_sha256=subject.NEW_PREFLIGHT_SHA,apply=False)

    def repin_new(self):
        self.monkeypatch.setattr(subject,'NEW_STAGE_SHA',subject.pin(self.root/'source031_deploy/stage.json')['sha256'])
        self.monkeypatch.setattr(subject,'NEW_PREFLIGHT_SHA',subject.pin(self.root/'source031_deploy/preflight.json')['sha256'])

    def set_actor(self, actor):
        write(self.directory/'progress.json',dict(task_actor_update=actor))

    def unchanged(self, admitted):
        for item in (v for v in admitted.values() if isinstance(v,dict) and 'file' in v):
            if subject.pin(item['file'])!=item:raise ValueError('Queued input changed')

    def admission(self, old, train):
        assert old==self.old and train==self.root/'train'
        return dict(**deepcopy(self.training),receipt=deepcopy(self.native_receipt),directory=self.directory,
                    plan=dict(budget_source_stage=str(self.claim.parent)))

    def open_fd(self, pid):
        self.opens.append(pid)
        if self.on_open:self.on_open()
        return os.open(self.controller_lock,os.O_RDONLY)

    def signal_fd(self, fd):
        assert fd>=0
        self.signals.append(subject.PINS['controller'][0])
        if self.on_signal:return self.on_signal()
        controller=self.identities.pop(subject.PINS['controller'][0])
        self.identities.pop(subject.PINS['coordinator'][0])
        self.lock_fd.close();self.lock_fd=None
        write(self.root/'source030_queue/status.json',dict(status='attention_no_retry',phase='controller',error='controller exited -15',qualification=None))
        write(self.root/'source030_queue/controller.execution.json',dict(pid=controller['pid'],command=controller['command'],exit_code=-15))

    def execute_apply(self):
        self.args.apply=True
        return subject.cancel(self.args,modules=self.modules,sleep=lambda seconds:None)

    def close(self):
        if self.lock_fd is not None:self.lock_fd.close();self.lock_fd=None


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    f=Fixture(tmp_path,monkeypatch)
    yield f
    f.close()


def test_default_is_read_only_and_opens_no_pidfd(fixture):
    before={p:p.read_bytes() for p in fixture.root.rglob('*') if p.is_file()}
    result=subject.cancel(fixture.args,modules=fixture.modules)
    assert result['status']=='validated_waiting_source030_not_cancelled'
    assert not fixture.opens and not fixture.signals
    assert {p:p.read_bytes() for p in fixture.root.rglob('*') if p.is_file()}==before


def test_actual_cancel_flow_publishes_once_and_never_signals_training(fixture):
    before=(fixture.root/'source030_queue/queue.intent.json').read_bytes()
    result=fixture.execute_apply()
    assert result['status']=='waiting_source030_cancelled_without_training_signal'
    assert fixture.opens==fixture.signals==[4166078]
    assert result['worker_signals']==result['training_signals']==result['PPO_updates']==result['refunds']==0
    assert result['controller_lock_released'] and result['old_transition_products_absent']
    assert all(result['old_'+key]['exited'] for key in ('controller','coordinator'))
    intent=subject.read(fixture.root/'source031_queue/cancel_source030.intent.json')
    assert intent['snapshot']['old_queue_intent']==result['old_queue_intent']
    assert (fixture.root/'source030_queue/queue.intent.json').read_bytes()==before
    with pytest.raises(ValueError,match='prior cancellation'):
        fixture.execute_apply()
    assert fixture.signals==[4166078]


@pytest.mark.parametrize('name',subject.PRODUCTS)
def test_any_prior_transition_product_refuses_before_intent_or_pidfd(fixture,name):
    write(fixture.root/name,{'synthetic':'already produced'})
    with pytest.raises(ValueError,match='transition artifact'):fixture.execute_apply()
    assert not fixture.opens and not fixture.signals
    assert not (fixture.root/'source031_queue/cancel_source030.intent.json').exists()


@pytest.mark.parametrize('role',subject.PINS)
def test_reused_or_missing_owner_refuses_before_signal(fixture,role):
    fixture.identities[subject.PINS[role][0]]['starttime']+=1
    with pytest.raises(ValueError,match='identity/state'):fixture.execute_apply()
    assert not fixture.opens and not fixture.signals


@pytest.mark.parametrize('actor',[13897,14897,-1,True])
def test_last_1000_or_invalid_actor_is_not_waiting_cancellation(fixture,actor):
    fixture.set_actor(actor)
    with pytest.raises(ValueError,match='1000 actor'):fixture.execute_apply()
    assert not fixture.opens and not fixture.signals


@pytest.mark.parametrize('fault',['phase','new_preflight','new_source','old_intent','migrated_claim','controller_parent','lock_owner'])
def test_real_input_faults_refuse_without_any_signal(fixture,fault,monkeypatch):
    if fault=='phase':write(fixture.root/'source030_queue/status.json',dict(status='normal_boundary_closed_preparing',phase='prepare'))
    elif fault=='new_preflight':write(fixture.root/'source031_deploy/preflight.log',dict(status='passed',roles=29))
    elif fault=='new_source':(fixture.next/'scripts/example.py').write_text('changed code')
    elif fault=='old_intent':write(fixture.root/'source030_queue/queue.intent.json',{'changed':True})
    elif fault=='migrated_claim':write(fixture.claim,dict(contract_sha256='not-old-v26'))
    elif fault=='controller_parent':fixture.identities[4166078]['ppid']=1
    else:monkeypatch.setattr(subject,'lock_owner',lambda path:[123])
    with pytest.raises(ValueError):fixture.execute_apply()
    assert not fixture.opens and not fixture.signals


@pytest.mark.parametrize('fault',['pid_reuse','transition','phase','owner','new_source','account','intent'])
def test_after_pidfd_open_race_keeps_intent_but_sends_nothing(fixture,fault):
    def change():
        if fault=='pid_reuse':fixture.identities[4166078]['starttime']+=1
        elif fault=='transition':write(fixture.root/subject.PRODUCTS[0],{'hold':'has begun'})
        elif fault=='phase':write(fixture.root/'source030_queue/status.json',dict(status='prepared_old_delivery_retained_launching',phase='launch'))
        elif fault=='owner':fixture.identities[4141576]['starttime']+=1
        elif fault=='new_source':(fixture.next/'scripts/example.py').write_text('changed after open')
        elif fault=='intent':write(fixture.root/'source031_queue/cancel_source030.intent.json',{'changed_after_open':True})
        else:write(fixture.claim,dict(contract_sha256='new-v27'))
    fixture.on_open=change
    with pytest.raises(ValueError):fixture.execute_apply()
    assert fixture.opens==[4166078] and not fixture.signals
    assert subject.read(fixture.root/'source031_queue/cancel_source030.receipt.json')['status']=='attention_no_retry'


def test_timeout_never_kills_parent_or_retries(fixture):
    fixture.on_signal=lambda:None
    clock=iter([0.,61.])
    fixture.args.apply=True
    with pytest.raises(TimeoutError):subject.cancel(fixture.args,modules=fixture.modules,now=lambda:next(clock),sleep=lambda seconds:None)
    assert fixture.signals==[4166078]
    assert subject.read(fixture.root/'source031_queue/cancel_source030.receipt.json')['status']=='attention_no_retry'
