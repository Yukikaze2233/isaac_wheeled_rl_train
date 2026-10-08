"""Summarize the exact fourth 100 speed actor updates from existing TB HTTP data."""
from pathlib import Path
import json, math, hashlib
root = Path(__file__).resolve().parent
raw_path = root / "source029_tb400_001.json"
x = json.loads(raw_path.read_text())
steps = set(range(41703, 41803))
def series(section, tag, baseline=False):
    rows = x[section][tag]
    by_step = {int(row[1]): float(row[2]) for row in rows}
    required = steps | ({41702} if baseline else set())
    if not required <= by_step.keys() or not all(math.isfinite(by_step[k]) for k in required):
        raise ValueError(f"Incomplete/nonfinite series {tag}")
    return by_step
result = dict(schema="source029-fourth-100-actor-tensorboard-summary-v1",
    source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    source_commit="e5bd9c1367b875d7f229b9a1930647013d023372",
    actor_updates=[301,400], consumed_TB_steps=[41703,41802],
    method="Exact 100 actor log steps; batch group means weighted by rollout_samples, conditional stop means by stop samples. Counter deltas exclude preceding paid critic rollout.",
    qualification=None, native_evaluations=0, PPO_updates_by_reader=0,
    timing_mean_seconds={}, groups={}, recovery_ended_deltas={}, scope_notes=[
      "Group names are birth/sampling labels, not fixed commands: SCUT can resample both vx and yaw.",
      "Signed mean vx/command can cancel; their difference is not velocity MAE.",
      "Rollout sample weights include script/blend statistics; policy-owned exposure is reported separately.",
      "original_chain/L2 includes non-fallen births; this is not a four-fallen-pose success rate.",
      "This is sampled randomized training, not a deterministic checkpoint evaluation or hardware qualification."])
for tag in x["timing"]:
    if tag.endswith("_seconds"):
        values=series("timing",tag)
        result["timing_mean_seconds"][tag]=sum(values[k] for k in steps)/100
names=sorted({tag.split("/")[1] for tag in x["selected"]})
for name in names:
    prefix=f"LearningAudit/{name}/"
    counts=series("selected",prefix+"rollout_samples")
    count=sum(counts[k] for k in steps)
    if count<=0:raise ValueError("No rollout samples")
    row=dict(rollout_samples=count, update_points=100)
    for metric in ("height_error_m","planar_speed_m_s","command_vx_m_s",
                   "command_yaw_rad_s","velocity/signed_forward_m_s"):
        values=series("selected",prefix+metric)
        row[metric]=sum(values[k]*counts[k] for k in steps)/count
    stop=series("selected",prefix+"settled_stop/samples")
    values=series("selected",prefix+"settled_stop/mean_speed_m_s")
    n=sum(stop[k] for k in steps)
    row["settled_stop_samples"]=n
    row["settled_stop_mean_speed_m_s"]=(sum(values[k]*stop[k] for k in steps)/n if n else None)
    result["groups"][name]=row
for level in ("prefix_stand","prefix_release","original_chain"):
    row={}
    for metric in ("ended","successes","failures","collection_timeouts"):
        values=series("recovery",f"RecoveryOutcomes/{level}/{metric}",True)
        delta=values[41802]-values[41702]
        if delta<0 or delta!=int(delta):raise ValueError("Counter reset or non-integer")
        row[metric]=int(delta)
    if row["ended"]!=row["successes"]+row["failures"]+row["collection_timeouts"]:
        raise ValueError("Ended count differs from outcome decomposition")
    row["success_fraction"]=(row["successes"]/row["ended"] if row["ended"] else None)
    result["recovery_ended_deltas"][level]=row
for tag in x["recovery"]:
    if tag.startswith("RecoveryRollout/"):
        values=series("recovery",tag)
        result.setdefault("rollout_metric_mean",{})[tag]=sum(values[k] for k in steps)/100
out=root/"source029_tb400_001.summary.json"
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(result,indent=2,ensure_ascii=False))
