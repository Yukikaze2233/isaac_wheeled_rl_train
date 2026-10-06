"""Targeted CPU native-update equivalence/lifecycle; no scene or PPO."""
import ast,gzip,importlib.util,json,math,types
from copy import deepcopy
from pathlib import Path
import pytest,torch
HERE=Path(__file__).resolve().parent
ROOT=Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
H=load(HERE/'capture_prepare_hook.py','hook_test')
D=load(HERE/'run_capture_prepare_021.py','hook_driver_test')
N=load(ROOT/'src/wheeled_tasks/chassis/recovery.py','hook_native_test')
P=load(HERE.parent/'candidate/src/wheeled_tasks/chassis/recovery.py','hook_patch_test')
M=load(HERE.parent.parent/'mechanical_capture_candidate_021/mechanical_capture.py','hook_mechanical003_test')

def writer(path,value):path.write_text(json.dumps(value,allow_nan=False))
def make(mod,count=4):
 p=torch.tensor([0.,.2,0.,0.,.2,0.]);f=mod.RecoveryFSM(count,'cpu',{'side_speed_rad_s':1.5},p,p,p,p,[1.,1.,-1.,-1.]);q=p.expand(count,-1).clone();f.reset(torch.arange(count),q,True);return f,q

def attach(tmp,f,auth=lambda _:None,hooks=None,write=writer):
 env=types.SimpleNamespace(recovery=types.SimpleNamespace(fsm=f,reference_tick=0))
 h=H.CapturePrepareHook(env,hooks or M.HookSet(),write,tmp,M.plain,auth,lambda:[(i,1) for i in range(min(2,f.count))]);h.apply();return h

def inputs(q,gyro=.1,tilt=0.,support=True,height=.305,clear=True):
 n=len(q);return (q,torch.zeros_like(q),q.new_tensor([math.sin(tilt),0.,-math.cos(tilt)]).expand(n,3),q.new_tensor([0.,gyro,0.]).expand(n,3),q.new_full((n,),height),torch.full((n,),support),torch.full((n,),clear))
def compare(a,b):
 keys=set(vars(a))-{'update','_enter'};assert keys==set(vars(b))-{'update','_enter'}
 for k in keys:
  x,y=getattr(a,k),getattr(b,k)
  if isinstance(x,torch.Tensor):assert torch.equal(x,y),k
  elif isinstance(x,dict) and any(isinstance(v,torch.Tensor) for v in x.values()):
   for n in x:assert torch.equal(x[n],y[n]),(k,n)
  else:assert x==y,k

@pytest.mark.parametrize('gyro',[0.,.2,.7499,.75,.8,6.9672311855401565,9.])
def test_patched_source_all_values_exact(tmp_path,gyro):
 a,q=make(N);b,_=make(P);a.phase[:]=b.phase[:]=6;h=attach(tmp_path,a);rng=torch.get_rng_state().clone()
 for _ in range(5):a.update(*inputs(q,gyro));b.update(*inputs(q,gyro));compare(a,b);h.env.recovery.reference_tick+=1
 assert h.calls_attempted==h.calls_completed==5 and torch.equal(rng,torch.get_rng_state());assert not h.cleanup()
 assert 'update' not in vars(a) and '_enter' not in vars(a)

@pytest.mark.parametrize('phase',range(11))
def test_same_tick_other_phase_changes_equivalent(tmp_path,phase):
 a,q=make(N);b,_=make(P);a.phase[:]=b.phase[:]=phase;h=attach(tmp_path,a)
 for _ in range(4):a.update(*inputs(q,.85));b.update(*inputs(q,.85));compare(a,b);h.env.recovery.reference_tick+=1
 assert not h.cleanup()

def test_mixed_batch_original_new_CAPTURE_does_not_change_this_tick(tmp_path):
 a,q=make(N);b,_=make(P);a.phase[:]=b.phase[:]=torch.tensor([4,6,2,7]);h=attach(tmp_path,a)
 a.update(*inputs(q,.9));b.update(*inputs(q,.9));compare(a,b)
 assert a.phase.tolist()==[6,6,2,7] and h.blocked_lane_transitions==1
 assert not h.cleanup();events=[json.loads(l) for l in gzip.open(tmp_path/'capture_prepare_enter_events.jsonl.gz','rt')]
 assert h.first_interventions=={1:1} and sum(r['all4096_blocked_capture_transitions'] for r in events)==1

def test_reroute_failure_partial_reset_and_no_extra_rng(tmp_path):
 a,q=make(N);b,_=make(P);h=attach(tmp_path,a);rng=torch.get_rng_state().clone()
 a.phase[:]=b.phase[:]=6
 for _ in range(30):a.update(*inputs(q,0.,math.pi/2,False));b.update(*inputs(q,0.,math.pi/2,False));compare(a,b);h.env.recovery.reference_tick+=1
 assert a.reroute_count.tolist()==[1]*4
 a.phase[:]=b.phase[:]=6
 for _ in range(40):a.update(*inputs(q,0.,math.pi/2,False));b.update(*inputs(q,0.,math.pi/2,False));compare(a,b);h.env.recovery.reference_tick+=1
 assert a.reroute_count.tolist()==[1]*4
 ids=torch.tensor([0,2]);a.reset(ids,q,True);b.reset(ids,q,True);compare(a,b)
 a.age_ticks[:]=b.age_ticks[:]=1599;a.phase[:]=b.phase[:]=6
 a.update(*inputs(q,.9));b.update(*inputs(q,.9));compare(a,b);assert a.failure_code.tolist()==[6]*4
 assert torch.equal(rng,torch.get_rng_state()) and not h.cleanup()

def test_low_speed_one_second_clock_matches_patch(tmp_path):
 a,q=make(N);b,_=make(P);a.phase[:]=b.phase[:]=6;h=attach(tmp_path,a)
 for _ in range(260):a.update(*inputs(q));b.update(*inputs(q));compare(a,b);h.env.recovery.reference_tick+=1
 assert a.stable_ticks[0]==199 and not a.motion_released.any()
 a.update(*inputs(q));b.update(*inputs(q));compare(a,b);assert a.motion_released.all();assert not h.cleanup()

@pytest.mark.parametrize('bad',['gyroshape','qshape','nonfinite'])
def test_original_rejection_failure_and_cleanup(tmp_path,bad):
 a,q=make(N);b,_=make(P);a.phase[:]=b.phase[:]=6;h=attach(tmp_path,a);args=list(inputs(q))
 if bad=='gyroshape':args[3]=torch.zeros(2,3)
 if bad=='qshape':args[0]=torch.zeros(2,6)
 if bad=='nonfinite':args[3]=torch.full((4,3),float('nan'))
 if bad!='nonfinite':
  with pytest.raises(ValueError):a.update(*args)
  with pytest.raises(ValueError):b.update(*args)
  assert h.calls_attempted==1 and h.calls_completed==0
 else:a.update(*args);b.update(*args)
 compare(a,b);assert not h.in_update and h.current_gyro is None and not h.cleanup();assert 'update' not in vars(a) and '_enter' not in vars(a)

def test_direct_enter_outside_original_update_is_not_filtered(tmp_path):
 f,q=make(N);f.phase[:]=6;h=attach(tmp_path,f);f._enter(torch.ones(4,dtype=torch.bool),7);assert f.phase.tolist()==[7]*4 and h.calls_attempted==0;assert not h.cleanup()

def test_partial_apply_and_duplicate_owner_rejections(tmp_path):
 f,q=make(N)
 class FailSecond(M.HookSet):
  def install(self,owner,name,callback):
   if len(self.rows)==1:raise RuntimeError('fixture second install failure')
   return super().install(owner,name,callback)
 with pytest.raises(RuntimeError,match='second install'):attach(tmp_path,f,hooks=FailSecond())
 assert 'update' not in vars(f) and '_enter' not in vars(f)
 h=attach(tmp_path,f)
 with pytest.raises(ValueError,match='duplicate'):h.apply()
 with pytest.raises(ValueError,match='local callback'):H.CapturePrepareHook(h.env,M.HookSet(),writer,tmp_path,M.plain,lambda _:None,lambda:[])
 assert not h.cleanup() and not h.cleanup()

def test_foreign_owner_not_overwritten_and_config_not_restored(tmp_path):
 f,q=make(N);h=attach(tmp_path,f);foreign=lambda *args:None;f.update=foreign;f.config['push_speed_rad_s']=3.
 errors=h.cleanup();assert errors and f.update is foreign and f.config['push_speed_rad_s']==3. and '_enter' not in vars(f)

def test_full2400_completion_restores_before_consumer(tmp_path):
 f,q=make(N);h=attach(tmp_path,f);f.phase[:]=9
 for i in range(2400):h.env.recovery.reference_tick=i;f.update(*inputs(q))
 assert not h.finish();r=json.loads((tmp_path/'capture_prepare_hook_preclose.json').read_text());assert r['complete'] and r['hook_callbacks_restored'] and r['native_reference_tick_last']==2399
 assert 'update' not in vars(f) and '_enter' not in vars(f)

@pytest.mark.parametrize('native_value',[6.,.5,float('nan')])
def test_config_cannot_silently_inherit_a_different_baseline(tmp_path,native_value):
 f,q=make(N);f.config['side_speed_rad_s']=native_value
 with pytest.raises(ValueError,match='config binding'):attach(tmp_path,f)

def test_cold_factory_has_no_early_env_isaac_import():
 tree=ast.parse((HERE/'run_capture_prepare_021.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_observer')
 for n in ast.walk(fn):
  if isinstance(n,(ast.Import,ast.ImportFrom)):pytest.fail('Factory adds an import before constructed env')

@pytest.mark.parametrize('when',['auth_init','parent_finish'])
def test_parent_lifecycle_cleanup_precedes_parent_failure_close(tmp_path,when):
 f,q=make(N);env=types.SimpleNamespace(recovery=types.SimpleNamespace(fsm=f,reference_tick=0));order=[]
 class FakeParent:
  def __init__(self,env,output):self.mechanical=types.SimpleNamespace(birth_ids=torch.ones(4,dtype=torch.long))
  def finish(self,output):
   order.append('parent_finish');assert 'update' not in vars(f) and '_enter' not in vars(f)
   if when=='parent_finish':raise RuntimeError('parent late finish')
   return {}
  def _before_close(self):order.append('parent_close');assert 'update' not in vars(f) and '_enter' not in vars(f)
 side=types.SimpleNamespace(make_side_observer=lambda *args:FakeParent)
 cap=types.SimpleNamespace(sha=lambda _:None);sm=types.SimpleNamespace(LANES=(0,1))
 def auth(_):
  if when=='auth_init':raise RuntimeError('auth failure')
 obs=D.make_observer(None,side,cap,M,sm,H,writer,'unused','unused',authenticate=auth)
 if when=='auth_init':
  with pytest.raises(RuntimeError,match='auth failure'):obs(env,tmp_path)
  assert order==['parent_close'];return
 owner=obs(env,tmp_path)
 for i in range(2400):env.recovery.reference_tick=i;f.phase[:]=9;f.update(*inputs(q))
 with pytest.raises(RuntimeError,match='parent late finish'):owner.finish(tmp_path)
 assert order==['parent_finish','parent_close'] and owner.capture_prepare.closed


def fake_pre_side_rows():
 return types.SimpleNamespace(LANES=(0,1),rows=lambda path:path,
  comparable=lambda row:row,compare_bits=lambda a,b:None if a==b else 'different native frame')
def pre_rows(steps=4):
 return [dict(state=dict(lane=l),physical_step=s,value=s) for l in (0,1) for s in range(steps)]
def test_pre_intervention_compares_only_prior_physical_step():
 baseline=pre_rows();actual=deepcopy(baseline)
 next(x for x in actual if x['state']['lane']==0 and x['physical_step']==2)['value']=99
 r=H.pre_intervention_review(fake_pre_side_rows(),baseline,actual,dict(selected_first_birth_first_blocked_update_call={'0':2}))
 assert r['status']=='pre_intervention_equal' and r['lanes'][0]['compared_through_physical_step']==1
 assert r['lanes'][1]['compared_rows']==4

def test_pre_intervention_mismatch_is_preserved_not_admitted():
 baseline=pre_rows();actual=deepcopy(baseline);actual[1]['value']=99
 r=H.pre_intervention_review(fake_pre_side_rows(),baseline,actual,dict(selected_first_birth_first_blocked_update_call={'0':3}))
 assert r['status']=='pre_intervention_rejected' and r['lanes'][0]['first_mismatch']['physical_step']==1

def test_pre_intervention_missing_row_and_duplicate_rejected():
 baseline=pre_rows();actual=deepcopy(baseline);actual.pop(1)
 with pytest.raises(ValueError,match='history incomplete'):
  H.pre_intervention_review(fake_pre_side_rows(),baseline,actual,dict(selected_first_birth_first_blocked_update_call={'0':3}))
 with pytest.raises(ValueError,match='duplicate'):
  H.pre_intervention_review(fake_pre_side_rows(),baseline,baseline+[baseline[0]],dict(selected_first_birth_first_blocked_update_call={}))
