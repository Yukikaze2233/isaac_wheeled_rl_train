"""Export recorded native physical-prefix traces at 50 Hz; no physics or policy updates."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory",type=Path)
args=parser.parse_args()
base=args.directory
report=json.loads((base/"report.json").read_text())
assert report["execution_status"]=="completed" and report["formal_updates_consumed"]==0
assert report["actor_tensors_unchanged"] and report["checkpoint_unchanged"]
source=base/"physical_frames.jsonl"
out=base/"trace_tensorboard"
out.mkdir(exist_ok=False)
writer=SummaryWriter(str(out))
writer.add_text("RecoveryPrefixTrace/Scope","Recorded native 200 Hz physical frames, selecting every fourth frame (50 Hz). Step is actual physical tick; sim_time_s is authoritative. Pose is after physics, ownership/gates are native before torque. Old flat actor, N=4, nominal-delay diagnostic, zero PPO.",0)
seen={};count=0
with source.open() as stream:
    for line in stream:
        row=json.loads(line)
        if row["physical_tick"]%4:continue
        name=row["case"];tick=row["physical_tick"];stem="RecoveryPrefixTrace/"+name+"/"
        assert tick>seen.get(name,0);seen[name]=tick
        x,y,z,w=row["root_pose_world_xyzw"][3:]
        norm=math.sqrt(x*x+y*y+z*z+w*w)
        assert norm>0 and math.isfinite(norm)
        x,y,z,w=(v/norm for v in (x,y,z,w))
        gx,gy,gz=2*(w*y-x*z),-2*(y*z+w*x),-(1-2*(x*x+y*y))
        values={"sim_time_s":row["sim_time_s"],"pose_after_physics/pitch_deg":math.degrees(math.atan2(gx,-gz)),
            "pose_after_physics/roll_deg":math.degrees(math.atan2(-gy,-gz)),
            "pose_after_physics/tilt_deg":math.degrees(math.acos(max(-1,min(1,-gz)))),
            "pose_after_physics/gyro_norm_rad_s":math.sqrt(sum(v*v for v in row["root_angular_velocity_body_rad_s"])),
            "native_before_torque/raw_phase":row["raw_fsm_phase_used"],
            "native_before_torque/ready_dwell_ticks":row["ready_ticks"],
            "native_before_torque/blend_fraction":row["blend_fraction_used"],
            "native_before_torque/actual_RL_ownership":int(row["effective_control_ownership"] in ("rl_hold","rl_unlocked"))}
        values.update({"native_before_torque/gate_"+name:int(v) for name,v in row["gates"].items()})
        for name,value in zip(report["startup"]["active_joint_order"],row["applied_torque6"]):
            values["applied_effort_after_chain/"+name+"_Nm"]=value
        for tag,value in values.items():
            assert math.isfinite(value);writer.add_scalar(stem+tag,value,tick)
        count+=1
writer.flush();writer.close()
receipt={"source_physical_frames_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_report_sha256":hashlib.sha256((base/"report.json").read_bytes()).hexdigest(),
    "source_decimation":4,"source_hz":200,"trace_hz":50,"step_unit":"native_physical_tick_5ms",
    "lane_frames_written":count,"last_physical_tick_by_case":seen,
    "event_files_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
    "scope":"Recorded physical diagnosis, not new physics or PPO or qualification.","local_tensorboard_server_started":False}
(out/"trace_writer_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"trace_frames":count,"event_files":len(receipt["event_files_sha256"]),"qualification":False}))
