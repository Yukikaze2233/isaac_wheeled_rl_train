"""Focused CPU checks for one isolated native FSM transition. No Isaac/PPO."""
import importlib.util,math
from pathlib import Path
import pytest,torch
BASE=Path(__file__).resolve().parent
FROZEN=Path('/tmp/scut-recovery-frozen-021-qv1ukmbr/src/wheeled_tasks/chassis/recovery.py')
CANDIDATE=BASE/'candidate/src/wheeled_tasks/chassis/recovery.py'

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=load('_capture_original',FROZEN);NEW=load('_capture_candidate',CANDIDATE)

def make(mod,config=None,count=1):
 nominal=torch.tensor([-.3207,.0949,0.,.3207,-.0949,0.])
 f=mod.RecoveryFSM(count,'cpu',config,nominal,nominal,nominal,nominal,[1.,1.,-1.,-1.]);q=nominal.repeat(count,1);f.reset(torch.arange(count),q,True);return f,q

def tick(f,q,gyro=.1,tilt=0.,support=True,height=.305,clear=True):
 n=len(q);g=q.new_tensor([math.sin(tilt),0.,-math.cos(tilt)]).repeat(n,1);w=q.new_tensor([0.,gyro,0.]).repeat(n,1)
 return f.update(q,torch.zeros_like(q),g,w,q.new_full((n,),height),torch.full((n,),support),torch.full((n,),clear))

def state_equal(a,b):
 for k,v in vars(a).items():
  w=getattr(b,k)
  if isinstance(v,torch.Tensor):assert torch.equal(v,w),k
  elif isinstance(v,dict):
   if any(isinstance(x,torch.Tensor) for x in v.values()):
    for n in v:assert torch.equal(v[n],w[n]),(k,n)
   else:assert v==w,k
  else:assert v==w,k

def test_exact_single_source_change_and_defaults():
 old=FROZEN.read_text();new=CANDIDATE.read_text();needle='        self._enter(capturing & (tilt < math.radians(20.)) & support & (estimated_height > .22), RecoveryPhase.PREPARE)\n'
 change='        self._enter(capturing & (tilt < math.radians(20.)) & support & (estimated_height > .22)\n                    & (gyro.norm(dim=-1) < self.config["handover_gyro_max"]), RecoveryPhase.PREPARE)\n'
 assert old.count(needle)==new.count(change)==1;assert new.replace(change,needle)==old
 assert NEW.RecoveryFSM.DEFAULTS==OLD.RecoveryFSM.DEFAULTS

@pytest.mark.parametrize('gyro',[0.,.1,.74])
def test_low_gyro_existing_entry_identical(gyro):
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=int(OLD.RecoveryPhase.CAPTURE)
 tick(a,q,gyro);tick(b,q,gyro);assert b.phase.item()==int(NEW.RecoveryPhase.PREPARE);state_equal(a,b)

@pytest.mark.parametrize('gyro',[.75,.7501,6.9672311855401565,9.])
def test_high_gyro_keeps_capture_no_ready_release_or_clock_reset(gyro):
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=int(OLD.RecoveryPhase.CAPTURE)
 tick(a,q,gyro);tick(b,q,gyro);assert a.phase.item()==7 and b.phase.item()==6
 assert b.age_ticks.item()==1 and b.phase_ticks.item()==1 and b.ready_ticks.item()==0
 assert not b.motion_released.item() and b.reroute_count.item()==0

@pytest.mark.parametrize('cfg,gyro,expected',[(.5,.4,7),(.5,.5,6),(.9,.8,7),(.9,.9,6)])
def test_existing_config_gyro_threshold_is_used(cfg,gyro,expected):
 f,q=make(NEW,{'handover_gyro_max':cfg});f.phase[:]=6;tick(f,q,gyro);assert f.phase.item()==expected

@pytest.mark.parametrize('case',[{'tilt':math.radians(25)},{'support':False},{'height':.20},{'tilt':math.radians(90),'support':False}])
def test_old_capture_gates_are_not_relaxed(case):
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=6;tick(a,q,.1,**case);tick(b,q,.1,**case);state_equal(a,b);assert b.phase.item()==6

@pytest.mark.parametrize('phase',[0,1,2,3,4,5,7,8,9,10])
def test_other_phase_updates_exact(phase):
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=phase
 tick(a,q,.85);tick(b,q,.85);state_equal(a,b)
 # Later ORBIT/THRUST ticks can enter CAPTURE, where the intended change applies.

def test_only_one_original_unsupported_reroute_and_no_second_grant():
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=6
 for _ in range(30):tick(a,q,0.,math.radians(90),False);tick(b,q,0.,math.radians(90),False);state_equal(a,b)
 assert b.reroute_count.item()==1 and b.route.item()==2
 a.phase[:]=b.phase[:]=6
 for _ in range(40):tick(a,q,0.,math.radians(90),False);tick(b,q,0.,math.radians(90),False);state_equal(a,b)
 assert b.reroute_count.item()==1 and b.phase.item()==6

def test_low_gyro_full_handover_and_original_one_second_release_identical():
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=6;tick(a,q);tick(b,q);state_equal(a,b)
 for _ in range(20):tick(a,q);tick(b,q);state_equal(a,b)
 assert b.phase.item()==8
 for _ in range(40):tick(a,q);tick(b,q);state_equal(a,b)
 assert b.phase.item()==9 and not b.motion_released.item()
 for _ in range(199):tick(a,q);tick(b,q);state_equal(a,b)
 assert b.stable_ticks.item()==199 and not b.motion_released.item()
 tick(a,q);tick(b,q);state_equal(a,b);assert b.stable_ticks.item()==200 and b.motion_released.item()

def test_original_eight_second_failure_is_not_lengthened():
 a,q=make(OLD);b,_=make(NEW);a.phase[:]=b.phase[:]=6
 for _ in range(1599):tick(a,q,6.9);tick(b,q,6.9)
 assert b.phase.item()==6 and b.age_ticks.item()==1599
 tick(a,q,6.9);tick(b,q,6.9)
 assert a.phase.item()==b.phase.item()==10 and a.failure_code.item()==b.failure_code.item()==6
 assert b.age_ticks.item()==1600 and not b.motion_released.item()
