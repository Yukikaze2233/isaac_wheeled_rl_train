"""CPU-only provenance, command clocks and first-cohort measurement contracts."""
from collections import Counter
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
SPEC = importlib.util.spec_from_file_location("locomotion_exploration_probe", ROOT / "scripts/probe_locomotion_exploration.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
from locomotion_exploration_sampling import CASES, FirstCohortLedger

NAMES = tuple(name for name in CASES for _ in range(4))
PROFILES = dict(zip(CASES, (0., .5, -.5)))


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=CPU Probe Test", "-c", "user.email=cpu-probe@example.invalid", "commit", "-qm", "CPU fixture")
    return git(root, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    names = [*probe.OWNED_SCRIPTS, "src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/v40/core.py",
             "src/wheeled_tasks/agents/v40_ppo_cfg.py", "scripts/train_chassis.py"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU-only frozen source\n")
    revision = commit(root)
    monkeypatch.setattr(probe, "CORE_FILES", ("src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/v40/core.py"))
    monkeypatch.setattr(probe, "CORE_REVISION", revision)
    return root, revision


def test_complete_clean_source_is_authentic(source):
    root, revision = source
    result = probe.authenticate_source(root, revision)
    assert result["source_revision"] == result["base_core_revision"] == revision
    assert result["complete_clean_checkout"] is True
    assert result["actual_source_files_sha256"] == probe.source_files(root)
    assert set(result["core_files_sha256"]) == set(probe.CORE_FILES)


@pytest.mark.parametrize("identity", ["HEAD", "b4ce809", "A" * 40, "0" * 40])
def test_full_actual_head_identity_required(source, identity):
    with pytest.raises(ValueError):
        probe.authenticate_source(source[0], identity)


@pytest.mark.parametrize("name", ["src/wheeled_tasks/chassis/env.py", "unexpected.txt"])
def test_dirty_or_untracked_source_is_rejected(source, name):
    root, revision = source
    (root / name).write_text("unfrozen\n")
    with pytest.raises(ValueError, match="clean"):
        probe.authenticate_source(root, revision)


def test_even_committed_extra_core_change_is_rejected(source):
    root, baseline = source
    (root / "src/wheeled_tasks/chassis/env.py").write_text("# unauthorized physics change\n")
    candidate = commit(root)
    assert candidate != baseline
    with pytest.raises(ValueError, match="differs from b4"):
        probe.authenticate_source(root, candidate)


def test_clean_harness_update_keeps_core_identity(source):
    root, baseline = source
    (root / "scripts/probe_locomotion_exploration.py").write_text("# new bounded observer\n")
    candidate = commit(root)
    result = probe.authenticate_source(root, candidate)
    assert result["source_revision"] == candidate
    assert result["base_core_revision"] == baseline


def test_source_sha_checks_actual_recorded_bytes(source, monkeypatch):
    root, revision = source
    actual = probe.source_files(root)
    actual["scripts/train_chassis.py"] = "0" * 64
    monkeypatch.setattr(probe, "source_files", lambda root: actual)
    with pytest.raises(ValueError, match="Actual source is not from H"):
        probe.authenticate_source(root, revision)


@pytest.mark.parametrize("speed", [0., .5, -.5])
def test_native_trajectory_has_exact_bounded_clock(speed):
    plan = probe.schedule(speed)
    assert len(plan) == 1000
    counts = Counter(phase for phase, _ in plan)
    assert counts["settle"] == 100
    if not speed:
        assert counts == {"settle": 100, "stand": 650, "hold": 250}
    else:
        assert counts == {"settle": 100, "acceleration": 17, "steady": 200, "braking": 17,
                          "stop_warmup": 100, "stop": 500, "hold": 66}
        assert plan[317] == ("braking", 0.)
    assert all(vx == 0. for phase, vx in plan if phase in ("settle", "braking", "stop", "stop_warmup", "stand", "hold"))


def test_other_speed_cannot_expand_the_diagnostic():
    with pytest.raises(ValueError):
        probe.schedule(3.)


def command_env():
    return SimpleNamespace(cfg={"command_slew": {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}},
        commands=torch.tensor([[0., 0., .305]]), command_target=torch.zeros(1, 2), mode=torch.zeros(1, dtype=torch.long),
        history=SimpleNamespace(last_tick=5), updates=0)


def test_driver_writes_both_command_and_target_once_with_native_slew():
    env = command_env()
    env.update_targets = lambda: setattr(env, "updates", env.updates + 1)
    previous = env.commands.clone()
    ramp = []
    for _ in range(17):
        previous = probe.apply_command(env, previous, torch.tensor([[.5, 0., .305]]))
        ramp.append(previous[0, 0].item())
        assert torch.equal(env.commands[:, :2], env.command_target)
        assert env.history.last_tick is None
    assert ramp[0] == pytest.approx(.03) and ramp[-1] == pytest.approx(.5)
    braking = probe.apply_command(env, previous, torch.tensor([[0., 0., .305]]))
    assert braking[0, 0].item() == pytest.approx(.47)
    assert env.updates == 18
    assert torch.equal(braking[:, :2], env.command_target)
    yaw = probe.apply_command(env, torch.tensor([[0., 0., .305]]), torch.tensor([[0., 1., .305]]))
    assert yaw[0, 1].item() == pytest.approx(.08)


def frame(lane=0, phase="stand", tick=1):
    return {"lane": lane, "phase": phase, "physics_tick": tick, "phase_anchor_xy_m": [0., 0.],
        "root_position_local_m": [.003*tick, 0., .305], "root_link_velocity_world_m_s": [3.,4.,0.],
        "root_link_velocity_body_m_s": [.1,0.,0.], "root_angular_velocity_body_rad_s": [0.,0.,-.088],
        "command": [0.,0.,.305], "height_m": .305, "pitch_deg": 2., "roll_deg": -3.}


def test_complete_window_uses_world_speed_and_actual_exposure():
    metrics, ledger = probe.PhaseMetrics(NAMES, PROFILES), FirstCohortLedger(NAMES)
    for tick in range(401, 3001):
        metrics.observe(frame(tick=tick))
    rows = metrics.finish(ledger)
    stand = rows[0]["phases"]["stand"]
    assert stand["complete_exposure"] is True
    assert stand["observed_seconds"] == 13.
    assert stand["completed_means"]["planar_speed_m_s"] == 5.
    assert stand["completed_means"]["vx_mae_m_s"] == pytest.approx(.1)
    assert stand["completed_means"]["yaw_mae_rad_s"] == pytest.approx(.088)
    assert stand["peak_displacement_m"] == pytest.approx(9.)
    windows = ledger.summary(physical_tick=4000)["cases"]["stand_305mm"]["windows"]["stand"]
    assert windows["complete_count"] == 1
    assert windows["mean"] == 5.


def test_partial_window_has_no_complete_mean_or_credit():
    metrics, ledger = probe.PhaseMetrics(NAMES, PROFILES), FirstCohortLedger(NAMES)
    metrics.observe(frame(tick=401))
    rows = metrics.finish(ledger)
    stand = rows[0]["phases"]["stand"]
    assert stand["completed_means"] is None
    assert stand["observed_partial_means"]["planar_speed_m_s"] == 5.
    windows = ledger.summary(physical_tick=4)["cases"]["stand_305mm"]["windows"]["stand"]
    assert windows["complete_count"] == 0 and windows["mean"] is None
    assert rows[4]["phases"]["stop"]["observed_physics_frames"] == 0


def test_missing_physical_frame_is_rejected():
    metrics = probe.PhaseMetrics(NAMES, PROFILES)
    metrics.observe(frame(tick=401))
    with pytest.raises(ValueError, match="lost a physical frame"):
        metrics.observe(frame(tick=403))


def tensor_view(value):
    return SimpleNamespace(torch=value)


def fake_env():
    data = SimpleNamespace(joint_pos=tensor_view(torch.zeros(12,18)), joint_vel=tensor_view(torch.zeros(12,18)),
        root_link_pose_w=tensor_view(torch.tensor([[0.,0.,.305,0.,0.,0.,1.]]*12)),
        root_link_lin_vel_w=tensor_view(torch.tensor([[.1,.2,0.]]*12)),
        root_link_lin_vel_b=tensor_view(torch.tensor([[.3,.4,0.]]*12)),
        root_com_lin_vel_w=tensor_view(torch.tensor([[1.,2.,0.]]*12)),
        root_com_lin_vel_b=tensor_view(torch.tensor([[3.,4.,0.]]*12)),
        root_com_ang_vel_b=tensor_view(torch.tensor([[0.,0.,-.088]]*12)),
        projected_gravity_b=tensor_view(torch.tensor([[0.,0.,-1.]]*12)), applied_torque=tensor_view(torch.zeros(12,18)))
    env = SimpleNamespace(scene_groups=NAMES, ids=[0,1,4,2,3,5], origins=torch.zeros(12,3),
        commands=torch.tensor([[0.,0.,.305]]*12), calls=Counter())
    env.v5 = SimpleNamespace(requested_motor_effort=torch.zeros(12,6), current_motor_bounds=torch.zeros(12,6))
    def decode(actions, q):
        env.calls["decode"] += 1
        return actions[:,:4].clone(), actions[:,4:].clone(), actions.clone()
    def motor(q,dq,legs,wheels):
        env.calls["motor"] += 1
        env.v5.requested_motor_effort.fill_(10.)
        env.v5.current_motor_bounds.fill_(3.)
        return torch.full((12,6),3.)
    def response(torque):
        env.calls["response"] += 1
        torque.add_(2.)
        return torque
    def write(*,target):
        env.calls["write"] += 1
        data.applied_torque.torch.copy_(target)
    def step(**kwargs):
        env.calls["step"] += 1
    def update(dt):
        env.calls["update"] += 1
    env.v5.decode,env.v5.motor_efforts=decode,motor
    env.actuator_response=SimpleNamespace(enabled=True,apply_torque=response)
    env.robot=SimpleNamespace(data=data,set_joint_effort_target_index=write,update=update)
    env.sim=SimpleNamespace(step=step)
    return env


def native_step(env, recorder, tick, *, terminal=None):
    recorder.begin(tick, ["settle"]*12)
    q,dq=env.robot.data.joint_pos.torch[:,env.ids],env.robot.data.joint_vel.torch[:,env.ids]
    legs,wheels,_=env.v5.decode(torch.zeros(12,6),q)
    for _ in range(4):
        torque=env.v5.motor_efforts(q,dq,legs,wheels)
        torque=env.actuator_response.apply_torque(torque)
        torque.clamp_(max=2.)
        effort=torch.zeros(12,18);effort[:,env.ids]=torque
        env.robot.set_joint_effort_target_index(target=effort)
        env.sim.step(render=False);env.robot.update(.005)
        env.v5.requested_motor_effort.fill_(-100.)
        env.v5.current_motor_bounds.fill_(-100.)
    flags={k:torch.zeros(12,dtype=torch.bool) for k in ("done","terminated","success","time_outs","boundary")}
    if terminal is not None:
        flags["done"][terminal]=flags["terminated"][terminal]=True
    recorder.end(flags["done"],{"time_outs":flags["time_outs"],"diagnostics":{
        "terminated":flags["terminated"],"success":flags["success"],"reasons":{"boundary":flags["boundary"]}}})


def test_physical_hooks_clone_immediately_and_call_original_once():
    env,stream,ledger=fake_env(),io.StringIO(),FirstCohortLedger(NAMES)
    original=env.v5.motor_efforts
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream)
    native_step(env,recorder,1)
    rows=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows)==48 and recorder.physics_tick==4
    assert env.calls=={"decode":1,"motor":4,"response":4,"write":4,"step":4,"update":4}
    row=rows[0]
    assert row["requested_motor_effort"]==[10.]*6
    assert row["current_motor_bounds"]==[3.]*6
    assert row["response_input"]==[3.]*6 and row["response_output_before_reclip"]==[5.]*6
    assert row["pd_capped_torque"]==[3.]*6
    assert row["submitted_active_torque"]==row["applied_torque6"]==[2.]*6
    assert row["root_link_velocity_body_m_s"]!=row["root_com_velocity_body_m_s"]
    recorder.close()
    assert env.v5.motor_efforts is original


def test_whole_population_continues_but_first_terminal_removes_credit():
    env,stream,ledger=fake_env(),io.StringIO(),FirstCohortLedger(NAMES)
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream)
    native_step(env,recorder,1,terminal=0)
    native_step(env,recorder,2)
    rows=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows)==96
    lane0=[r for r in rows if r["lane"]==0]
    assert [r["first_cohort"] for r in lane0]==[True]*4+[False]*4
    assert [r["after_first_terminal"] for r in lane0]==[False]*4+[True]*4
    assert recorder.metrics.rows[0,"settle"]["samples"]==4
    assert recorder.metrics.rows[1,"settle"]["samples"]==8
    assert ledger.rows[0]["terminal"]["physical_tick"]==4
    assert env.calls["motor"]==8 and env.calls["decode"]==2
    recorder.close()


def fake_preflight(monkeypatch):
    import locomotion_exploration_fixture as fixture
    monkeypatch.setattr(fixture,"authenticate_inputs",lambda *a:{"source":{},"receipt":{"sha":"fixed","clocks":{"task_actor_update":4185}}})
    monkeypatch.setattr(fixture,"build_fixture",lambda root,source,out,role:{"receipt":{},"design_preflight":{},
        "fixture_role":role,"task_actor_update":4185 if role=="pipeline_age_only" else 0})


def test_cpu_preflight_finishes_final_identity_and_no_isaac(tmp_path,monkeypatch):
    fake_preflight(monkeypatch)
    calls=[]
    monkeypatch.setattr(probe,"authenticate_source",lambda *a:calls.append(a) or {"identity":"H"})
    before=set(sys.modules)
    result=probe.main(["--arm","mean","--input-dir",str(tmp_path/"in"),"--output",str(tmp_path/"out"),
        "--source-revision","a"*40,"--preflight-only"])
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert result==0 and len(calls)==2
    assert report["source_identity_final_verified"] is True and report["sealed_inputs_final_verified"] is True
    assert report["formal_updates_consumed"]==0 and report["qualification"] is None
    assert report["fixture_role"]=="clean" and report["expected_task_actor_update"]==0
    assert not any(name.startswith(("isaaclab.app","isaacsim","omni.")) for name in set(sys.modules)-before)


def test_preflight_final_source_change_cannot_return_success(tmp_path,monkeypatch):
    fake_preflight(monkeypatch)
    identity=iter([{"identity":"H"},{"identity":"changed"}])
    monkeypatch.setattr(probe,"authenticate_source",lambda *a:next(identity))
    result=probe.main(["--arm","mean","--input-dir",str(tmp_path/"in"),"--output",str(tmp_path/"out"),
        "--source-revision","a"*40,"--preflight-only"])
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert result==1 and report["status"]=="failed_source_identity"
    assert report["execution_status"]=="failed"
    assert report["source_identity_final_verified"] is False


def test_failure_mid_policy_preserves_actual_partial_exposure(tmp_path):
    env,stream,ledger=fake_env(),io.StringIO(),FirstCohortLedger(NAMES)
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream)
    recorder.begin(1,["settle"]*12)
    for _ in range(2):
        env.sim.step(render=False)
        env.robot.update(.005)
    actor=torch.nn.Linear(1,1)
    before=deepcopy(actor.state_dict())
    sampler=SimpleNamespace(draw_counter=1,sample_element_count=72)
    report={"status":"failed","execution_status":"failed","requested_physics_steps":4000}
    probe.save_actual_execution(report,tmp_path,recorder,ledger,actor,before,sampler)
    first=json.loads((tmp_path/"first_cohort.json").read_text())
    assert report["physics_steps"]==2 and report["policy_steps"]==0
    assert report["policy_steps_started"]==1 and report["partial_policy_physics_steps"]==2
    assert report["actor_tensors_unchanged"] is True
    assert report["draw_counter"]==1 and report["sample_element_count"]==72
    assert first["partitions"]["right_censored_unfinished"]==12
    assert first["rows"][0]["exposure_seconds"]==.01
    assert first["lane_phase_metrics"][0]["phases"]["settle"]["observed_physics_frames"]==2
    assert first["cases"]["stand_305mm"]["windows"]["stand"]["mean"] is None
    assert first["cases"]["forward_0p5"]["windows"]["stop"]["complete_count"]==0
    recorder.close()


def test_actor_final_identity_is_checked_on_failure(tmp_path):
    actor=torch.nn.Linear(1,1)
    before=deepcopy(actor.state_dict())
    with torch.no_grad():
        actor.weight.add_(1.)
    report={"status":"failed","execution_status":"failed"}
    probe.save_actual_execution(report,tmp_path,None,None,actor,before,None)
    assert report["status"]=="failed_actor_identity"
    assert report["actor_tensors_unchanged"] is False


def test_metric_wrong_native_phase_is_rejected():
    metrics=probe.PhaseMetrics(NAMES,PROFILES)
    with pytest.raises(ValueError,match="trajectory clock"):
        metrics.observe(frame(tick=1))


def test_pipeline_preflight_binds_authentic_clock_without_scene(tmp_path,monkeypatch):
    fake_preflight(monkeypatch)
    monkeypatch.setattr(probe,"authenticate_source",lambda *a:{"identity":"H"})
    result=probe.main(["--arm","mean","--input-dir",str(tmp_path/"in"),"--output",str(tmp_path/"out"),
        "--source-revision","a"*40,"--fixture-role","pipeline_age_only","--preflight-only"])
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert result==0 and report["expected_task_actor_update"]==4185
    assert report["fixture_role"]=="pipeline_age_only" and report["formal_updates_consumed"]==0


def test_pipeline_preflight_rejects_clock_not_from_authentic_seal(tmp_path,monkeypatch):
    fake_preflight(monkeypatch)
    import locomotion_exploration_fixture as fixture
    monkeypatch.setattr(fixture,"authenticate_inputs",lambda *a:{"source":{},"receipt":{"clocks":{"task_actor_update":0}}})
    monkeypatch.setattr(probe,"authenticate_source",lambda *a:{"identity":"H"})
    result=probe.main(["--arm","mean","--input-dir",str(tmp_path/"in"),"--output",str(tmp_path/"out"),
        "--source-revision","a"*40,"--fixture-role","pipeline_age_only","--preflight-only"])
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert result==1 and "authenticated source seal" in report["error"]


def test_physics_execution_without_feedback_is_separately_reported(tmp_path):
    env,stream,ledger=fake_env(),io.StringIO(),FirstCohortLedger(NAMES)
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream)
    recorder.begin(1,["settle"]*12)
    env.sim.step(render=False)
    report={"status":"failed","execution_status":"failed"}
    probe.save_actual_execution(report,tmp_path,recorder,ledger,None,None,None)
    assert report["physics_executed_steps"]==1 and report["observed_feedback_steps"]==0
    assert report["pending_physics_step_without_feedback"] is True
    assert report["first_cohort"]["rows"][0]["exposure_seconds"]==0.
    recorder.close()


def make_pipeline_delay():
    from wheeled_tasks.chassis.signal_delay import PhysicalSignalDelay
    from wheeled_tasks.chassis.task_campaign import signal_delay_settings
    delay=PhysicalSignalDelay(12,"cpu",signal_delay_settings("speed"),.005,torch.Generator().manual_seed(190619))
    delay.reset(torch.arange(12),torch.zeros(12,18))
    return delay


def pipeline_native_step(env,recorder,reference,tick):
    recorder.begin(tick,["settle"]*12)
    env.signal_delay.begin_policy(4185);reference.begin_policy(4185)
    q=env.robot.data.joint_pos.torch[:,env.ids]
    legs,wheels,_=env.v5.decode(torch.full((12,6),float(tick)),q)
    for _ in range(4):
        delayed=env.signal_delay.targets(legs,wheels)
        expected=reference.targets(legs,wheels)
        assert all(torch.equal(a,b) for a,b in zip(delayed,expected,strict=True))
        q=env.robot.data.joint_pos.torch[:,env.ids];dq=env.robot.data.joint_vel.torch[:,env.ids]
        torque=env.v5.motor_efforts(q,dq,*delayed)
        torque=env.actuator_response.apply_torque(torque)
        effort=torch.zeros(12,18);effort[:,env.ids]=torque
        env.robot.set_joint_effort_target_index(target=effort)
        env.sim.step(render=False);env.robot.update(.005)
        sensors=torch.full((12,18),float(recorder.physics_tick))
        env.signal_delay.observe(sensors);reference.observe(sensors)
        assert torch.equal(env.signal_delay.feedback,reference.feedback)
    zeros=torch.zeros(12,dtype=torch.bool)
    recorder.end(zeros,{"time_outs":zeros,"diagnostics":{"terminated":zeros,"success":zeros,"reasons":{"boundary":zeros}}})


def test_pipeline_wrappers_preserve_native_rng_buffers_and_fresh_pd(monkeypatch):
    env,stream,journal,ledger=fake_env(),io.StringIO(),io.StringIO(),FirstCohortLedger(NAMES)
    env.signal_delay,reference=make_pipeline_delay(),make_pipeline_delay()
    calls=Counter()
    for name in ("sensor_buffer","target_buffer"):
        buffer=getattr(env.signal_delay,name)
        original=buffer.read_groups
        def count_read(ages, original=original, name=name):
            calls[name]+=1
            return original(ages)
        monkeypatch.setattr(buffer,"read_groups",count_read)
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream,journal)
    birth=recorder.actor_delay_snapshot()
    assert all(row["source_physics_tick"]==[0]*4 and row["feedback18"]==[0.]*18 for row in birth)
    for tick in (1,2,3):
        pipeline_native_step(env,recorder,reference,tick)
        assert torch.equal(env.signal_delay.generator.get_state(),reference.generator.get_state())
        assert torch.equal(env.signal_delay.lags,reference.lags)
        for name in ("sensor_buffer","target_buffer"):
            a,b=getattr(env.signal_delay,name),getattr(reference,name)
            assert torch.equal(a.buffer,b.buffer) and torch.equal(a.current_length,b.current_length)
    assert recorder.delay_totals=={"begin_policy":3,"targets":12,"observe":12}
    assert calls=={"sensor_buffer":12,"target_buffer":12}
    rows=[json.loads(line) for line in journal.getvalue().splitlines()]
    assert Counter(row["event"] for row in rows)=={"policy_lag_sample":36,"target_applied":144,"observation_refresh":144}
    assert all(row["raw_jitter_ms"] is None and "not_covered" in row["raw_jitter_coverage"] for row in rows)
    first_targets=[row for row in rows if row["event"]=="target_applied" and row["applied_physics_tick"]==1]
    first_sensors=[row for row in rows if row["event"]=="observation_refresh" and row["append_physics_tick"]==1]
    assert any(any(lag>0 for lag in row["requested_lag_steps"]) for row in first_targets)
    assert all(row["actual_age_steps"]==[0]*2 and row["source_physics_tick"]==[0]*2 for row in first_targets)
    assert all(row["actual_age_steps"]==[0]*4 and row["source_physics_tick"]==[1]*4 for row in first_sensors)
    observed=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert all(row["pd_feedback_matches_fresh_physics"] is True for row in observed)
    assert observed[0]["signal_delay"]["observation_feedback_before_native_refresh"]["event"]=="native_reset_feedback"
    assert observed[12]["signal_delay"]["observation_feedback_before_native_refresh"]["append_physics_tick"]==1
    snapshot=recorder.actor_delay_snapshot()
    assert snapshot[0]["consumed_at_physics_tick"]==12 and snapshot[0]["append_physics_tick"]==12
    recorder.close()


def test_pipeline_clock_mismatch_fails_without_extra_native_draw():
    env,stream,journal,ledger=fake_env(),io.StringIO(),io.StringIO(),FirstCohortLedger(NAMES)
    env.signal_delay,reference=make_pipeline_delay(),make_pipeline_delay()
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream,journal)
    recorder.begin(1,["settle"]*12)
    reference.begin_policy(0)
    with pytest.raises(ValueError,match="authenticated fixed actor clock"):
        env.signal_delay.begin_policy(0)
    assert torch.equal(env.signal_delay.generator.get_state(),reference.generator.get_state())
    recorder.close()


def test_pipeline_pd_rejects_using_delayed_actor_sensors():
    env,stream,journal,ledger=fake_env(),io.StringIO(),io.StringIO(),FirstCohortLedger(NAMES)
    env.signal_delay=make_pipeline_delay()
    recorder=probe.PhysicalRecorder(env,ledger,PROFILES,stream,journal)
    recorder.begin(1,["settle"]*12)
    with pytest.raises(ValueError,match="must not delay physical PD feedback"):
        env.v5.motor_efforts(torch.ones(12,6),torch.zeros(12,6),torch.zeros(12,4),torch.zeros(12,2))
    assert env.calls["motor"]==0
    recorder.close()
