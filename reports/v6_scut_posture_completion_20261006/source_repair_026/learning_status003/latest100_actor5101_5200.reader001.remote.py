# Run through existing remote python -B stdin; stdout is the only report destination.
import json
def remote_main(p):
    import hashlib
    import importlib.util
    import json
    import os
    import sys
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path

    sys.dont_write_bytecode = True

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def read(path):
        return json.loads(Path(path).read_text())

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    result = dict(schema="source023-recovery-native-fixed100-window-readonly-v1", complete=False,
                  source_commit=p["source_commit"], qualification=None, window=p["window"],
                  reader_sha256=p["reader_sha256"], actual_receipts_sha256=p["actual_receipts_sha256"],
                  reader_dependencies_sha256=p["reader_dependencies_sha256"], PPO_updates_by_reader=0,
                  scope="Existing native stochastic PPO event; CPU only, no remote writes, no simulator/model/GPU/PPO/evaluation/TB server.",
                  comparison_scope="Sampled training statistics are separate from deterministic mean-policy qualification; different actor ages/curriculum/DR cannot establish a same-difficulty single-variable benefit.")
    try:
        print("READONLY_READER_PID=" + str(os.getpid()), file=sys.stderr, flush=True)
        root = Path(p["root"])
        run = root / "train/recovery/attempt_009/train"
        source = root / "source_6f07ec298e01"
        bundle = Path(p["bundle"])
        require(read(root / "recovery_entropy_launch_023.json") == p["launch"], "Actual launch receipt changed")
        require(read(root / "source_revision_023.json") == p["revision"], "Actual revision receipt changed")
        require(digest(root / "predecessor_023.json") == p["actual_receipts_sha256"]["predecessor_023.json"], "Actual predecessor bytes changed")
        require({name: digest(bundle / name) for name in p["reader_dependencies_sha256"]} == p["reader_dependencies_sha256"], "Preserved CPU reader bytes differ")
        require({name: digest(source / name) for name in p["source_files_sha256"]} == p["source_files_sha256"], "Frozen metric/controller/contract bytes differ")
        spec = importlib.util.spec_from_file_location("frozen022_for_source023_window", bundle / "read_recovery_window_022.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = read(source / "contracts/v6_task_campaign_recovery_worker_v21.json")
        invocation = read(run.parent / "invocation.json")
        argv = invocation["command"]
        require(argv == p["first_PPO"]["invocation"]["command"], "Actual worker invocation changed")
        rv = read(run / "resume_verification.json")
        require(rv == p["first_PPO"]["resume_verification"], "Actual resume evidence changed")
        startup = read(run / "startup.json")
        native = startup["startup"]
        observed = native["runtime_preflight"]["observed"]
        require(native["runtime_preflight"]["status"] == "runtime_configuration_passed" and
                (observed["physics_dt"], observed["policy_dt"], observed["decimation"], observed["seed"]) ==
                (.005, .02, 4, 290927), "Native timing identity differs")
        require(observed["recovery_hold_cost"] == config["recovery_hold_cost"] and
                observed["recovery_script_revision"] == p["first_PPO"]["actual_recovery_script_revision"] and
                observed["recovery_hold_cost_gate_binding"] == p["first_PPO"]["native_hold_cost_gate_binding"], "Native metric bindings differ")
        progress = read(run / "progress.json")
        completed, actors = progress["successful_updates"], progress["actor_updates_in_block"]
        critic = min(completed, 10)
        require(progress["pid"] == 3662149 and progress["num_envs"] == 4096 and
                completed == critic + actors and progress["parent_updates"] == 33620 and
                progress["task_actor_update"] == 6425 + actors and progress["stage_actor_update"] == 19793 + actors and
                progress["global_budget_actor_updates"] == 33460 + actors and
                progress["entropy_coef"] == config["entropy_coef"] == 0., "Native source023 identity/clocks differ")
        context = dict(consumption_origin=33620, task_actor_origin=6425, actor_consumption_origin=33460,
                       critic_warmup_updates=10, critic_completed=critic, actor_completed=actors,
                       total_completed=completed, first_actor_event_step=33630, first_actor_completion=33631,
                       actual_task_actor_update=progress["task_actor_update"], native_first11_observed=True,
                       consumption_and_actor_are_separate=True)
        adapter, reader = module.make_reader(context, bundle / "read_recovery_window_021.py",
                                             bundle / "read_recovery_window_020.py", bundle / "read_recovery_window_019.py")
        reader.COMMIT = p["source_commit"]
        owner = adapter.native_owner(progress["pid"], argv)
        require(owner["ppid"] == 3662142, "Native worker owner differs")
        groups = [row["name"] for row in config["scene_groups"]]
        pools = config["behavior_pool_membership"]
        bases = [g for g in groups if pools[g] != "scripted_recovery"]
        population = native["scene_group_counts"]
        require(len(groups) == len(set(groups)) == 95 and len(bases) == 41 and
                set(population) == set(groups) and sum(population.values()) == 4096 and
                sum(population[g] for g in bases) == 1449 and
                Counter(pools.values()) == Counter(scripted_recovery=54, stand_anchors=9,
                    stand_continuous=1, height=1, translation=16, transitions=6, yaw=8), "Native 95/41 catalog differs")
        events = list(run.glob("events.out.tfevents.*.3662149.*"))
        require(len(events) == 1, "Expected one event from actual source023 worker")
        metadata = {"startup": run / "startup.json", "invocation": run.parent / "invocation.json",
                    "resume_verification": run / "resume_verification.json"}
        metadata_sha = {name: digest(path) for name, path in metadata.items()}
        authority = dict(identity_verified=True, source_commit=p["source_commit"], source_root=str(source),
                         worker_pid=progress["pid"], event=str(events[0]), groups=groups, base_groups=bases,
                         group_pools=pools, population=population, progress=progress, clocks=context,
                         source_files_sha256=p["source_files_sha256"], invocation=invocation, native_owner=owner,
                         ordinary_posture_coverage=dict(available=True, native_eligible_population=1449,
                             base_group_population=1449, nonstandard_start_leg_bank_present=False))
        result.update(identity=authority, selection=dict(chosen_actor_end_update=p["window"]["end_update"],
                      available_actor_updates=actors, additional_actor_updates_needed=max(0, p["window"]["end_update"] - actors),
                      mode="explicit_nonoverlapping_actor_endpoint", native_zero_baseline_fabricated=False))
        if actors < p["window"]["end_update"]:
            result.update(status="pending_requested_actor_window", reason="Requested actor endpoint has not completed; event not parsed.")
        else:
            size_before = events[0].stat().st_size
            series, audit = reader.parse_events(events[0], reader.wanted_tags(authority) | adapter.extra_tags(authority),
                p["window"]["event_steps_inclusive"][1], min_step=p["window"]["baseline_step"], max_read_bytes=p["max_read_bytes"])
            window = adapter.select_actor_window(reader, series, authority, audit,
                100, p["window"]["end_update"], "after_ten_actor_baseline")
            window["reused_metric_schema"] = window["schema"]
            window["schema"] = "source023-recovery-native-actor-window-summary-v1"
            window["limitations"] = [line for line in window["limitations"] if not line.startswith("Source021 changes")]
            window["limitations"].append("Source023 sets recovery entropy coefficient to zero and charges ten critic adaptation updates; this stochastic window is descriptive, not causal or functional qualification.")
            require(window["window"]["event_steps_inclusive"] == p["window"]["event_steps_inclusive"] and
                    window["window"]["task_actor_updates_inclusive"] == p["window"]["task_actor_updates_inclusive"], "Fixed window boundary differs")
            result.update(status="requested_actor_window_complete" if window["complete"] else "pending_missing_native_tags",
                          complete=window["complete"], windows={"100": window}, parser_metric_reuse=reader.adapter_reuse,
                          event_bytes_before=size_before, event_bytes_after=events[0].stat().st_size,
                          read_byte_limit=p["max_read_bytes"])
        require({name: digest(source / name) for name in p["source_files_sha256"]} == p["source_files_sha256"], "Frozen source changed during read")
        require({name: digest(path) for name, path in metadata.items()} == metadata_sha, "Immutable native metadata changed during read")
        adapter.verify_owner_continuity(owner, adapter.native_owner(progress["pid"], argv))
        after = read(run / "progress.json")
        require(after["successful_updates"] >= completed, "Native completed clock went backwards")
        result.update(progress_after=after, owner_continuity_verified=True, source_files_unchanged=True,
                      immutable_metadata_unchanged=True)
    except Exception as error:
        result.update(status="read_failed", complete=False, error=type(error).__name__ + ": " + str(error))
    result["queried_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "read_failed" else 0

raise SystemExit(remote_main(json.loads('{"source_commit":"6f07ec298e01d2686216787cfba34b394151f1aa","root":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358","bundle":"/tmp/readonly_source022_stopped2440_20261007_001","window":{"actor_updates":100,"actor_updates_inclusive":[5101,5200],"baseline_update":5100,"end_update":5200,"successful_update_endpoints_in_attempt":[5110,5210],"baseline_step":38729,"event_steps_inclusive":[38730,38829],"task_actor_updates_inclusive":[11526,11625],"expected_observed_frames":9830400,"policy_seconds_per_environment":48.0,"critic_warmup_excluded":true,"new_critic_warmup_updates":10},"max_read_bytes":1073741824,"reader_sha256":"45cad594270bacf4a33de896c9dea92d9433918425ccb98ebbc4d52ca466d91c","reader_dependencies_sha256":{"read_recovery_window_022.py":"a5427d51ce7998c98064f0251a632727aceeb5eae5d1ef66ae4e0ada38a55f56","read_recovery_window_021.py":"5faa75e24b0dd484bdacb0273756642caab8fcc6d6e66f90c034a0bda6ac19a6","read_recovery_window_020.py":"521b10353125272aa0f21ff9dc986c65b958ea7dd225051cfc6cfdd269a215e5","read_recovery_window_019.py":"ffefbaae2e1cbf30b520224fc286a8adbc194708570e4ce0fcb00fa7f86cdb17"},"actual_receipts_sha256":{"launch_023.json":"4e179761ab2ebfb68fb7b78efcef02002f9ee04ce6fd7c22f7d3a19316411352","first_PPO_verification_023.json":"a1431d9261f080f6d367b89bd2cb3728414fa7cc85c52d35a4dec5f3afb4a2a3","predecessor_023.json":"45b55296d4bcd151c0bfd015c413d9303a5350ce53bdfea4b996e235db920e5d","source_revision_023.json":"b17e79369bdf23c862d619281c70cf93f5e924d2bf738bfbb19e00ae63b5ffd2","frozen_source_inventory_023.json":"f1b20cfce97329447f037a422c546728735fbc190825035ada3227e28832e3d3"},"launch":{"schema":"task-recovery-entropy-launch-023-v1","status":"launched_waiting_for_real_restore_critic_adaptation_and_actor_PPO","admitted_at_utc":"2026-10-06T22:33:11.740512+00:00","source_commit":"6f07ec298e01d2686216787cfba34b394151f1aa","source_root":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01","run_root":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358","tmux":"v6-task-campaign-20261001T110045Z-282358-r23","plan_sha256":"119c77c21da31238ac1af285d0e8df1c627816b0d9645fa099899a6496f11755","source_archive_sha256":"ee2288079e27d8e0f03fe5cd4dbfb675c4b414c1eb1e222559fa84be353c445a","command":["/home/kaiser/robot-rl-sim60/env/bin/python","-B","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/scripts/run_task_campaign.py","--contract","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/contracts/v6_task_campaign_v21.json","--num-envs","4096","--device","cuda:0","--research","--max-runtime-seconds","15552000","--run-dir","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train"],"wrapper_command":["/home/kaiser/robot-rl-sim60/env/bin/python","-B","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/scripts/chassis_remote_job.py","--run-root","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358","--resume-existing","--source-revision-receipt","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_revision_023.json","--","/home/kaiser/robot-rl-sim60/env/bin/python","-B","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/scripts/run_task_campaign.py","--contract","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/contracts/v6_task_campaign_v21.json","--num-envs","4096","--device","cuda:0","--research","--max-runtime-seconds","15552000","--run-dir","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train"],"evidence_sha256":"370a3d6427f3b5a60aa10b9d806229bc26be1959d70197a96a461647a3f32cd7","parent":{"path":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_008/train/model_final.pt","sha256":"8dd37ba76092fa634e174facc9665d65b1df1fd9bbd1dfb42e852c19be9883b0","seal":{"schema":"chassis-learning-checkpoint-v1","sealed":true,"checkpoint_sha256":"8dd37ba76092fa634e174facc9665d65b1df1fd9bbd1dfb42e852c19be9883b0","contract_sha256":"a64eb6c03231e56d8efd8d83b87f70ee88c198de55fee0c2809aa766c7e19f62","learning_updates":33620,"learning_lineage_updates":19873,"stage_actor_update":19793,"global_actor_update_offset":0,"global_actor_update":19793,"actor_updates_consumed":33460,"task_campaign":{"campaign_id":"v6-task-domains-20261001","domain":"recovery","source_contract_file":"contracts/v6_task_campaign_flat_worker_v6.json","source_contract_sha256":"666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7","updates":20000,"clock":"domain_actor_updates","learning_audit":true,"evaluation_interval":0,"initialization":"full_learning_state","parent_candidate_scope":"research_not_accepted","execution_mode":"continuous_review_or_cap_then_next","revision_migration":{"schema":"task-worker-revision-v20","source_contract_file":"contracts/v6_task_campaign_recovery_worker_v19.json","source_contract_sha256":"7918e7850e16b96edfa54a8854d3f3571e8101dfa6527cad78a06b4e15baea95","source_recovery_file":"src/wheeled_tasks/chassis/recovery.py","source_recovery_sha256":"04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6"},"worker_revision_id":"v6-task-campaign-recovery-worker-v20"},"task_actor_update":6425,"confirmed_jump_bins_m":[],"gate_result":"not_assessed","role":"learning_state","budget_consumption_is_separate_from_checkpoint_lineage":true},"seal_sha256":"7f55fc69aab2821d6eedde6af93598ad5131e43a3e4d8ba209d274bc78741510","contract":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_008/train/model_final.contract.json","contract_sha256":"a64eb6c03231e56d8efd8d83b87f70ee88c198de55fee0c2809aa766c7e19f62","full_learning_state_integrity":{"actor_tensors":9,"critic_tensors":8,"std_axes":6,"Adam_parameter_states":17,"all_learning_tensors_finite":true},"authenticated_infos":{"contract_sha256":"a64eb6c03231e56d8efd8d83b87f70ee88c198de55fee0c2809aa766c7e19f62","successful_updates_total":33620,"stage_actor_update":19793,"task_actor_update":6425,"actor_updates_consumed":33460,"learning_lineage_updates":19873,"critic_warmup_remaining":0}},"expected_critic_warmup":10,"expected_entropy_coef":0.0,"runtime_shell":"/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh","runtime_shell_sha256":"c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7","qualification":null,"tmux_exit_code":0,"tmux_stderr":"","launched_at_utc":"2026-10-06T22:33:11.744669+00:00"},"revision":{"schema":"task-plan-revision-v21","previous_job_sha256":"6c8184d688b903a824760f5bdfb0bf87601c707bf00844cd0167ccbe8836aaf4","command":["/home/kaiser/robot-rl-sim60/env/bin/python","-B","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/scripts/run_task_campaign.py","--contract","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/contracts/v6_task_campaign_v21.json","--num-envs","4096","--device","cuda:0","--research","--max-runtime-seconds","15552000","--run-dir","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train"],"source_commit":"6f07ec298e01d2686216787cfba34b394151f1aa","old_source_root":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_723732adae37","new_source_root":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01","new_contract_relative_path":"contracts/v6_task_campaign_v21.json","old_contract_sha256":"5d94e35e3fe1bcfc5ae8488e4ce1b1f5779b8ce9598ee1b5642ceb879f47dfac","new_contract_sha256":"119c77c21da31238ac1af285d0e8df1c627816b0d9645fa099899a6496f11755","source_archive":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01.tar.gz","source_archive_sha256":"ee2288079e27d8e0f03fe5cd4dbfb675c4b414c1eb1e222559fa84be353c445a"},"first_PPO":{"invocation":{"command":["/home/kaiser/robot-rl-sim60/env/bin/python","-B","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/scripts/train_chassis.py","--contract","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_6f07ec298e01/contracts/v6_task_campaign_recovery_worker_v21.json","--stage","mixed","--research","--publish-state","--num-envs","4096","--updates","8585","--seed","290927","--block-index","0","--device","cuda:0","--run-dir","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_009/train","--resume","/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_008/train/model_final.pt","--resume-task-change","--learning-rate-scale","1.0","--consumed-updates","33620","--actor-updates-consumed","33460","--stage-actor-update","19793","--task-actor-update","6425","--confirmed-jump-bins-json","[]","--global-actor-update-offset","0","--max-runtime-seconds","15551997.55835433"],"parent_role":"unassessed_research_candidate_not_accepted","source_checkpoint":"/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_008/train/model_final.pt","source_checkpoint_sha256":"8dd37ba76092fa634e174facc9665d65b1df1fd9bbd1dfb42e852c19be9883b0","requested_updates":8585,"scheduled_end_actor_update":15000},"resume_verification":{"kind":"task_domain_recovery","actor_critic_optimizer":"preserved","environment_state":"new_episodes","task_clock":"restored","accepted_stage":null,"hardware_release":false,"source_contract_sha256":"a64eb6c03231e56d8efd8d83b87f70ee88c198de55fee0c2809aa766c7e19f62","target_contract_sha256":"eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2","actor_exact":true,"critic_exact":true,"optimizer_exact":true,"parent_checkpoint_sha256":"8dd37ba76092fa634e174facc9665d65b1df1fd9bbd1dfb42e852c19be9883b0","parent_updates":33620,"restored_checkpoint_updates":33620,"learning_lineage_updates":19873,"requested_learning_rate_scale":1.0,"restored_optimizer_learning_rate":0.0001,"diagnostic_origin":null,"parent_training_transitions":4049669376,"effective_learning_rate":0.0001,"learning_rate_override_applied":true,"parent_entropy_coef":null,"effective_entropy_coef":0.0,"entropy_coef_source":"contract"},"actual_recovery_script_revision":{"source_recovery_sha256":"04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6","side_speed_rad_s":1.5,"capture_prepare_gyro_gate":"native-current-tick-norm-below-existing-handover-gyro-max","handover_gyro_max_rad_s":0.75},"native_hold_cost_gate_binding":{"status":"authenticated_native_fsm_and_geometry","source":"RecoveryFSM.constants+native_fsm_config+ClosedChainHeightModel.wheel_radius_m","wheel_radius_m":0.06,"wheel_rim_gate_m_s":0.25,"leg_gate_rad_s":2.0,"gyro_gate_rad_s":0.75}},"source_files_sha256":{"contracts/v6_task_campaign_recovery_worker_v18.json":"7b5ecc7acc6871e072c2ca131d2ad2fa169dfd321b8a6c875cf2df643c5fd013","contracts/v6_task_campaign_recovery_worker_v21.json":"eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2","contracts/v6_task_campaign_v18.json":"027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022","contracts/v6_task_campaign_v21.json":"119c77c21da31238ac1af285d0e8df1c627816b0d9645fa099899a6496f11755","scripts/train_chassis.py":"15f7224187a645bd58b0b073382446ac5dcbfc76b4013c0fece30c00920db2fc","src/wheeled_tasks/chassis/env.py":"380887f8951a2de511264e828f262d362de67c2b77b909d042163eb69e8b7525","src/wheeled_tasks/chassis/episode_metrics.py":"26438717a12b3ab37b7279cbcae09dda91c549438c7a6090e7d6876c7a046511","src/wheeled_tasks/chassis/eval_env.py":"6ce02c158f1cf0ed76f030067e6d9b6d36376c99ba15c23d1e382b87f784625a","src/wheeled_tasks/chassis/learning_audit.py":"0ac7a3502199c0912bebf6a1689cf3d388ffe03e23fcb3217628a9a70d324715","src/wheeled_tasks/chassis/ordinary_posture_curriculum.py":"b3424cf23f9d7afec294ad00576d8fc2f8c9de4f2d3560df3bb852c46f9a4dff","src/wheeled_tasks/chassis/recovery.py":"04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6","src/wheeled_tasks/chassis/recovery_curriculum.py":"6617b7667dc703447b10c0ccd701424f833df51cf828f54e67b63e875b8e2a78","src/wheeled_tasks/chassis/recovery_prefix_curriculum.py":"2188f108930a6f8898ee145c397342ef8a235bfd0007c1c82f7a54f878253766","src/wheeled_tasks/chassis/recovery_quality_revision.py":"6ac2ce49d57c1061ed03a519e7290904fb831d33dca6f64cbc351a0b9d9b4972","src/wheeled_tasks/chassis/recovery_script_revision.py":"4d59036e4e0f5e7662f56e94976a373de14a1f1f6a562368b51c517f8e1d28f0","src/wheeled_tasks/chassis/recovery_training.py":"b475b0166e4cfc0c9a079c06c65bdd20fb744de30d00b92f0965febd5e3d24c5","src/wheeled_tasks/chassis/rewards.py":"b6f402f275114719bb8fe30a5c5a43a27edabc36d3e3a58e7c9f393022293c2d","src/wheeled_tasks/chassis/root_kinematics.py":"0c63be06fc026b0687aa4e9d9fd7dabb009f7c1ab24446244b672928368f610b","src/wheeled_tasks/chassis/task_campaign.py":"d4521ec3bb13ebf924d8529491e518c4ea06654cf720489fca4d7871ff654f7b"}}')))
