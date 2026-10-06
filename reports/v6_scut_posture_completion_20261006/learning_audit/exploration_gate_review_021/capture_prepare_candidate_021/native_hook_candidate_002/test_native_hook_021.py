"""CPU revision002: actual frozen55 NativeObserver chain, no App/scene/PPO."""
import ast,gzip,hashlib,importlib.util,inspect,json,math,types
from pathlib import Path
import pytest,torch
HERE=Path(__file__).resolve().parent
ROOT=Path('/tmp/scut-recovery-frozen-021-qv1ukmbr')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
H=load(HERE/'capture_prepare_hook.py','hook002_helper');D=load(HERE/'run_capture_prepare_021.py','hook002_driver')
N=load(ROOT/'src/wheeled_tasks/chassis/recovery.py','hook002_native');P=load(HERE.parent/'candidate/src/wheeled_tasks/chassis/recovery.py','hook002_patch')
M=load(HERE.parent.parent/'mechanical_capture_candidate_021/mechanical_capture.py','hook002_mechanical')
BASE=HERE.parents[2]/'probe_prefix_mean_stochastic_019.py';B=load(BASE,'hook002_actual55')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,v):path.write_text(json.dumps(v,allow_nan=False))
def setup(tmp,mod=N,count=4,parent=None):
 q=torch.tensor([0.,.2,0.,0.,.2,0.]);f=mod.RecoveryFSM(count,'cpu',{'side_speed_rad_s':1.5},q,q,q,q,[1.,1.,-1.,-1.]);q=q.expand(count,-1).clone();f.reset(torch.arange(count),q,True)
 r=types.SimpleNamespace(fsm=f,enabled=f.enabled,release_finished=torch.ones(count,dtype=torch.bool),reference_tick=0,
  pure_rl=f.pure_rl,policy_motion_hold=f.motion_hold,begin_policy=lambda:None,torque=lambda q,dq,e:e,
  pose_curriculum=types.SimpleNamespace(level_ids=torch.zeros(count,dtype=torch.long),profile_ids=torch.zeros(count,dtype=torch.long),profile_names=('fixture',)))
 env=types.SimpleNamespace(recovery=r,num_envs=count,device='cpu',scene_groups=['fixture']*count,kinds=['flat']*count)
 o=(parent or B.NativeObserver)(env,tmp)
 return f,q,env,o

def authenticate(f,o):return D.authenticate_inherited_update(f,o,BASE,D.BASE_OBSERVER_SHA,sha)
def attach(tmp,f,env,o,hooks=None):
 h=H.CapturePrepareHook(env,hooks or M.HookSet(),write,tmp,M.plain,lambda f:authenticate(f,o),lambda:[(i,1) for i in range(2)]);h.apply();return h

def inputs(q,gyro=.1,tilt=0.,support=True):
 n=len(q);return (q,torch.zeros_like(q),q.new_tensor([math.sin(tilt),0.,-math.cos(tilt)]).expand(n,3),q.new_tensor([0.,gyro,0.]).expand(n,3),q.new_full((n,),.305),torch.full((n,),support),torch.ones(n,dtype=torch.bool))
def compare(a,b):
 keys=set(vars(a))-{'update','_enter'};assert keys==set(vars(b))-{'update','_enter'}
 for k in keys:
  x,y=getattr(a,k),getattr(b,k)
  if isinstance(x,torch.Tensor):assert torch.equal(x,y),k
  elif isinstance(x,dict) and any(isinstance(v,torch.Tensor) for v in x.values()):
   for n in x:assert torch.equal(x[n],y[n]),(k,n)
  else:assert x==y,k

def close(h,o):
 cb=h.inherited_update;assert not h.cleanup();assert o.recovery.fsm.update is cb and '_enter' not in vars(o.recovery.fsm);o.sparse.close()

@pytest.mark.parametrize('gyro',[.2,.9,6.9672311855401565])
def test_real55_chain_patch_full_state_same_tick_and_gate_counters(tmp_path,gyro):
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();f,q,e,o=setup(a);p,_,pe,po=setup(b,P);h=attach(a,f,e,o)
 f.phase[:]=p.phase[:]=torch.tensor([4,6,2,7]);rng=torch.get_rng_state().clone()
 for i in range(40):
  e.recovery.reference_tick=i;f.update(*inputs(q,gyro));p.update(*inputs(q,gyro));compare(f,p)
  for k in ('gates','handover_ref','release_ref'):assert torch.equal(getattr(o,k),getattr(po,k)),k
 assert h.calls_completed==o.reference_calls==po.reference_calls==40 and torch.equal(rng,torch.get_rng_state())
 close(h,o);po.sparse.close()

def test_real55_chain_1s_release_reroute_and_partial_reset_preserved(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();f,q,e,o=setup(a);p,_,_,po=setup(b,P);h=attach(a,f,e,o)
 f.phase[:]=p.phase[:]=6
 for i in range(261):e.recovery.reference_tick=i;f.update(*inputs(q));p.update(*inputs(q));compare(f,p)
 assert f.motion_released.all() and torch.equal(o.release_ref,po.release_ref) and o.release_ref[0]==261
 f.reset(torch.tensor([0,2]),q,True);p.reset(torch.tensor([0,2]),q,True);compare(f,p)
 f.phase[:]=p.phase[:]=6
 for i in range(50):f.update(*inputs(q,0.,math.pi/2,False));p.update(*inputs(q,0.,math.pi/2,False));compare(f,p)
 assert f.reroute_count.tolist()==[1]*4
 f.phase[:]=p.phase[:]=6;f.age_ticks[:]=p.age_ticks[:]=1599
 f.update(*inputs(q,.9));p.update(*inputs(q,.9));compare(f,p);assert f.failure_code.tolist()==[6]*4
 close(h,o);po.sparse.close()

@pytest.mark.parametrize('fault',['wrong_original','foreign_closure','hash','local_enter'])
def test_inherited_observer_chain_rejects_wrong_identity(tmp_path,fault):
 f,q,e,o=setup(tmp_path)
 if fault=='wrong_original':o.original_update=types.MethodType(lambda *_:None,f)
 if fault=='foreign_closure':
  other=tmp_path/'other';other.mkdir();_,_,_,o2=setup(other);f.update=o2.recovery.fsm.update;o2.sparse.close()
 if fault=='local_enter':f._enter=lambda *_:None
 with pytest.raises(ValueError):D.authenticate_inherited_update(f,o,BASE,'0'*64 if fault=='hash' else D.BASE_OBSERVER_SHA,sha)
 o.sparse.close()

def test_original_rejection_once_cleanup_restores_actual55_chain(tmp_path):
 f,q,e,o=setup(tmp_path);h=attach(tmp_path,f,e,o);args=list(inputs(q));args[3]=torch.zeros(2,3)
 with pytest.raises(ValueError):f.update(*args)
 assert h.calls_attempted==1 and h.calls_completed==o.reference_calls==0 and not h.in_update and h.current_gyro is None
 close(h,o)

def test_partial_apply_and_foreign_callback_not_overwritten(tmp_path):
 f,q,e,o=setup(tmp_path);inherited=f.update
 class FailSecond(M.HookSet):
  def install(self,owner,name,callback):
   if len(self.rows)==1:raise RuntimeError('second install')
   return super().install(owner,name,callback)
 with pytest.raises(RuntimeError):attach(tmp_path,f,e,o,FailSecond())
 assert f.update is inherited and '_enter' not in vars(f)
 h=attach(tmp_path,f,e,o);foreign=lambda *_:None;f.update=foreign
 assert h.cleanup() and f.update is foreign and '_enter' not in vars(f);o.sparse.close()

@pytest.mark.parametrize('when',['auth_init','late_finish'])
def test_actual55_parent_constructed_then_outer_cleanup_before_parent_close(tmp_path,when):
 order=[]
 class ActualParent(B.NativeObserver):
  def __init__(self,env,output):super().__init__(env,output);self.mechanical=types.SimpleNamespace(birth_ids=torch.ones(4,dtype=torch.long))
  def _before_close(self):
   order.append('parent_close');assert isinstance(self.env.recovery.fsm.update,types.FunctionType);self.sparse.close()
  def finish(self,output):
   order.append('parent_finish');assert isinstance(self.env.recovery.fsm.update,types.FunctionType);raise RuntimeError('late parent finish')
 # env preparation only; use the real base class inside the driver superclass chain.
 f,q,e,old=setup(tmp_path);old.sparse.close();del f.update
 def auth(f):
  if when=='auth_init':raise RuntimeError('init auth')
  return authenticate(f,inspect.getclosurevars(f.update).nonlocals['self'])
 side=types.SimpleNamespace(make_side_observer=lambda *_:ActualParent);cap=types.SimpleNamespace(sha=sha);sh=types.SimpleNamespace(LANES=(0,1))
 cls=D.make_observer(None,side,cap,M,sh,H,write,'unused','unused',auth)
 if when=='auth_init':
  with pytest.raises(RuntimeError,match='init auth'):cls(e,tmp_path)
  assert order==['parent_close'];return
 obj=cls(e,tmp_path)
 for i in range(2400):e.recovery.reference_tick=i;f.phase[:]=9;f.update(*inputs(q))
 with pytest.raises(RuntimeError,match='late parent'):obj.finish(tmp_path)
 assert order==['parent_finish','parent_close'] and obj.capture_prepare.closed and obj.reference_calls==2400


def fake_side():return types.SimpleNamespace(LANES=(0,1),rows=lambda path:path,comparable=lambda r:r,compare_bits=lambda a,b:None if a==b else 'different')
def rows():return [dict(state=dict(lane=l),physical_step=s,value=s) for l in (0,1) for s in range(5)]

def test_global_prefix_hard_gate_accepts_later_perlane_rng_divergence():
 b=rows();a=rows();a[1]['value']=100
 r=H.pre_intervention_review(fake_side(),b,a,dict(first_global_intervention_update_call=1,selected_first_birth_first_blocked_update_call={'0':3}))
 assert r['status']=='pre_intervention_equal' and r['lanes'][0]['compared_rows']==1
 desc=r['lanes'][0]['after_global_before_selected_observation'];assert not desc['bitwise_identical'] and desc['first_mismatch']['physical_step']==1 and not desc['admission_gate'] and not desc['single_lane_causal_isolation']

def test_real_shared_env_rng_reset_before_perstep_jitter_can_diverge_without_selected_override():
 a=torch.Generator().manual_seed(7);b=torch.Generator().manual_seed(7);assert torch.equal(torch.rand(4,6,generator=a),torch.rand(4,6,generator=b))
 # Frozen env.reset ids consumes subset draws before next all-lane signal_delay.advance.
 torch.rand(1,6,generator=b)
 assert not torch.equal(torch.rand(4,6,generator=a)[0],torch.rand(4,6,generator=b)[0])

@pytest.mark.parametrize('failure',['early_diff','missing','selected_before_global'])
def test_strict_global_prefix_rejects_before_intervention(tmp_path,failure):
 b=rows();a=rows();hook=dict(first_global_intervention_update_call=3,selected_first_birth_first_blocked_update_call={'0':4})
 if failure=='early_diff':a[1]['value']=99;assert H.pre_intervention_review(fake_side(),b,a,hook)['status']=='pre_intervention_rejected';return
 if failure=='missing':a.pop(1)
 if failure=='selected_before_global':hook['selected_first_birth_first_blocked_update_call']['0']=2
 with pytest.raises(ValueError):H.pre_intervention_review(fake_side(),b,a,hook)

def test_no_global_intervention_retains_complete_common_hard_comparison():
 b=rows();r=H.pre_intervention_review(fake_side(),b,b,dict(first_global_intervention_update_call=None,selected_first_birth_first_blocked_update_call={}))
 assert r['status']=='pre_intervention_equal' and all(x['compared_rows']==5 for x in r['lanes'])

def test_cold_factory_no_early_native_import():
 fn=next(n for n in ast.parse((HERE/'run_capture_prepare_021.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_observer')
 assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(fn))
