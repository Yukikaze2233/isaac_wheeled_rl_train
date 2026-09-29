"""Offline force-balance diagnostic on recorded poses; not a forward replay."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
from identify_pair_dynamics import Replay


def diagnose(run,bundle,side,segments,output,stride=20):
 r=Replay(run,side=side,bundle=bundle,coordinate_sign=-1.,physics_dt=.001)
 m,d=r.model,r.data
 m.dof_armature[:]=0.;m.dof_damping[:]=0.;m.dof_frictionloss[:]=0.
 records=[]
 active={n:v for n,v in r.spec['nominal_joint_pos'].items() if n in r.audit.manifest['control_joint_names']}
 wheel=m.joint(r.letter+'_joint3').dofadr[0]
 prescribed=np.r_[r.vi,wheel]
 free=np.array([i for i in range(m.nv) if i not in prescribed])
 def tangent():
  eq=d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
  J=d.efc_J.reshape(d.nefc,m.nv)[eq]
  T=np.zeros((m.nv,3));T[prescribed,np.arange(3)]=1
  T[free]=np.linalg.lstsq(J[:,free],-J[:,prescribed],rcond=1e-10)[0]
  return T
 for seg in segments:
  ix,q0,*_=r.initial_state(seg)
  qmeas=r.real_q[ix]; dq=savgol_filter(qmeas,31,3,deriv=1,delta=.001,axis=0)
  ddq=savgol_filter(qmeas,51,3,deriv=2,delta=.001,axis=0)
  state=dict(r.spec['nominal_joint_pos'])
  for j in range(m.njnt):state[m.joint(j).name]=q0[m.joint(j).qposadr[0]]
  for k in range(50,len(ix)-50,stride):
   sample=ix[k]; state,gap=r.audit.solve({**active,**dict(zip(r.joints,qmeas[k]))},state)
   mujoco.mj_resetData(m,d)
   for j in range(m.njnt):d.qpos[m.joint(j).qposadr[0]]=state[m.joint(j).name]
   mujoco.mj_forward(m,d);T=tangent();v=np.r_[dq[k],0.];qbase=d.qpos.copy()
   # Differentiate the closure tangent along the measured velocity, not by
   # differentiating quantized passive coordinates.
   ds=1e-5; d.qpos[:]=qbase+ds*(T@v);mujoco.mj_forward(m,d);Tp=tangent()
   d.qpos[:]=qbase-ds*(T@v);mujoco.mj_forward(m,d);Tm=tangent()
   curve=((Tp-Tm)/(2*ds))@v
   d.qpos[:]=qbase;d.qvel[:]=T@v
   compression=r.binding['compression_at_q_zero_m']-d.qpos[r.si]
   d.ctrl[r.su]=np.polynomial.polynomial.polyval(np.clip(compression/r.curve['stroke_m'],0,1),r.curve['monomial_coefficients_n'])
   mujoco.mj_forward(m,d)
   M=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,M,d.qM)
   inertia=T.T@M@T;b=T.T@(M@curve+d.qfrc_bias-d.qfrc_actuator)
   # The wheel is unpowered on the suspended rig; eliminate its free response.
   aw=-(inertia[2,:2]@ddq[k]+b[2])/inertia[2,2]
   required=inertia[:2,:2]@ddq[k]+inertia[:2,2]*aw+b[:2]
   records.append([seg,sample,*qmeas[k],*dq[k],*ddq[k],*required,*r.real_torque[sample],r.knee_inner_deg(d.qpos[r.knee_qi]),gap])
  print(json.dumps({'side':side,'segment':seg,'rows':len(records)}),flush=True)
 rows=np.array(records);output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output/'inverse.npz',rows=rows,columns=['segment','sample','q0','q1','dq0','dq1','ddq0','ddq1','required0','required1','sent0','sent1','beta_deg','gap_m'])
 summaries=[]
 for which,label in [(np.isin(rows[:,0],[9,11,13,15]),'fit'),(~np.isin(rows[:,0],[9,11,13,15]),'heldout')]:
  free=which&(rows[:,12]<104)&(np.max(abs(rows[:,4:6]),axis=1)>.08)
  for axis in range(2):
   velocity=rows[free,4+axis];acc=rows[free,6+axis];Y=rows[free,10+axis]-rows[free,8+axis]
   X=np.c_[acc,velocity,np.tanh(velocity/.05)]
   f=lsq_linear(X,Y,bounds=([0,0,0],[.1,5,10]))
   summaries.append({'role':label,'axis':axis,'n':int(free.sum()),'nominal_force_rmse':float(np.sqrt(np.mean(Y**2))),'fit_residual_rmse':float(np.sqrt(np.mean((Y-X@f.x)**2))),'armature_damping_coulomb':f.x.tolist(),'residual_mean':float(np.mean(Y-X@f.x))})
 (output/'diagnosis.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps(summaries))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--side',required=True);p.add_argument('--segments',nargs='+',type=int,default=[9,11,13,15,37,39,41,43,79,89]);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=20);a=p.parse_args();diagnose(a.run,a.bundle,a.side,a.segments,a.output,a.stride)
