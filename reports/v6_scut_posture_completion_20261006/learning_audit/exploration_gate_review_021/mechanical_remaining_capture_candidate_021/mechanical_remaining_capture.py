"""Read-only, post-PhysX/pre-autoreset mechanical evidence in one isolated app.

No controller, RNG, simulation, terminal, or reset operation is implemented here.
The frozen native loop remains the only owner of those operations.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import types

import torch

LANES = (881, 1877, 2121)
PROFILES = ('front_down_90', 'step200_lip_near_shell', 'step200_upright_top')
LEVELS = (2, 2, 1)
GROUPS = ('recovery_front_down_90_forward', 'recovery_step200_lip_near_shell_stand', 'recovery_step200_prone_belly_top_forward')
TRACE_STEPS = 2400
GUARD_M = .003


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def plain(value):
    """Clone reused native views now; never retain an alias across a getter/reset."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            require(torch.isfinite(value).all(), "Nonfinite captured tensor")
        return value.detach().cpu().clone().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def same_callback(actual, expected):
    return actual is expected or (isinstance(actual, types.MethodType)
        and isinstance(expected, types.MethodType) and actual.__self__ is expected.__self__
        and actual.__func__ is expected.__func__)


class HookSet:
    """Restore only our hooks, including inherited descriptors without shadowing."""
    def __init__(self):
        self.rows = []
        self.closed = False

    def install(self, owner, name, callback):
        require(not self.closed, "Hooks already closed")
        prior = getattr(owner, name)
        self.rows.append((owner, name, callback, prior, name in vars(owner)))
        setattr(owner, name, callback)
        return prior

    def restore(self):
        if self.closed:
            return []
        errors = []
        for owner, name, callback, prior, had_local in reversed(self.rows):
            actual = getattr(owner, name)
            if same_callback(actual, callback):
                if had_local:
                    setattr(owner, name, prior)
                else:
                    delattr(owner, name)
            elif not same_callback(actual, prior):
                errors.append("Mechanical hook ownership changed: " + name)
        self.closed = True
        return errors


def pin_geometry(env):
    """The native env.closure_gap algebra, retaining every actual endpoint/pin."""
    pose = env.robot.data.body_link_pose_w.torch[:, env._closure_body_ids]
    local = env._closure_points.expand(env.num_envs, -1, -1, -1)
    cross = torch.linalg.cross(pose[..., 3:6], local)
    world = pose[..., :3] + local + 2 * (pose[..., 6:] * cross
        + torch.linalg.cross(pose[..., 3:6], cross))
    gaps = (world[:, :, 0] - world[:, :, 1]).norm(dim=-1)
    require(torch.isfinite(gaps).all(), "Nonfinite actual pin geometry")
    require(torch.equal(gaps.amax(-1), env.closure_gap()), "Captured pin max differs from native closure_gap")
    return world, gaps


def contact_slices(view, sensors, dt, convert):
    """Copy documented PhysX buffers before friction overwrites counts/starts."""
    require(view.filter_count > 0 and view.max_contact_data_count > 0, "Detailed native terrain contacts unavailable")
    matrix = convert(view.get_contact_force_matrix(dt=dt))
    net = convert(view.get_net_contact_forces(dt=dt))
    require(tuple(matrix.shape) == (view.sensor_count, view.filter_count, 3), "Native contact matrix layout differs")
    force, point, normal, distance, counts, starts = map(convert, view.get_contact_data(dt))
    require(tuple(counts.shape) == tuple(starts.shape) == (view.sensor_count, view.filter_count), "Native contact indices differ")
    require(len(force) == len(point) == len(normal) == len(distance) == view.max_contact_data_count, "Native contact buffer capacity differs")
    rows = []
    for sensor in sensors:
        patches = []
        for surface in range(view.filter_count):
            count, start = int(counts[sensor, surface]), int(starts[sensor, surface])
            require(0 <= count <= view.max_contact_data_count and (count == 0 or 0 <= start <= len(force)-count), "Native contact slice outside capacity")
            section = slice(start, start+count) if count else slice(0, 0)
            scalar = force[section]
            aggregate = (scalar * normal[section]).sum(dim=0)
            patches.append(dict(surface_index=surface, count=count,
                force_scalar_n=plain(scalar), point_physx_m=plain(point[section]),
                normal_physx=plain(normal[section]), separation_m=plain(distance[section]),
                normal_force_aggregate_n=plain(aggregate),
                normal_aggregate_minus_native_matrix_n=plain(aggregate-matrix[sensor,surface]),
                impulse_scalar_ns_derived_force_times_dt=plain(scalar * dt)))
        rows.append(dict(sensor_index=sensor, matrix_force_n=plain(matrix[sensor]),
            net_force_all_contacts_n=plain(net[sensor]), normal_patches=patches))
    # This is a capacity indicator, not a claimed PhysX overflow flag.
    capacity_indicator = dict(sum_reported_normal_patch_counts=int(counts.long().sum()),
        max_contact_data_count=int(view.max_contact_data_count),
        observed_count_at_capacity=bool(counts.long().sum() >= view.max_contact_data_count),
        native_overflow_flag_available=False)
    friction, point, counts, starts = map(convert, view.get_friction_data(dt))
    require(tuple(counts.shape) == tuple(starts.shape) == (view.sensor_count, view.filter_count), "Native friction indices differ")
    require(len(friction) == len(point) == view.max_contact_data_count, "Native friction capacity differs")
    for row, sensor in zip(rows, sensors):
        patches = []
        for surface in range(view.filter_count):
            count, start = int(counts[sensor, surface]), int(starts[sensor, surface])
            require(0 <= count <= view.max_contact_data_count and (count == 0 or 0 <= start <= len(friction)-count), "Native friction slice outside capacity")
            section = slice(start, start+count) if count else slice(0, 0)
            patches.append(dict(surface_index=surface, count=count, force_physx_n=plain(friction[section]),
                friction_force_aggregate_n=plain(friction[section].sum(dim=0)),
                point_physx_m=plain(point[section]), impulse_physx_ns_derived_force_times_dt=plain(friction[section] * dt)))
        row['friction_patches'] = patches
    return rows, capacity_indicator


class MechanicalCapture:
    def __init__(self, env, env_module, output, writer, convert):
        self.env, self.module, self.output, self.writer, self.convert = env, env_module, output, writer, convert
        self.hooks = HookSet()
        self.physics_steps = self.policy_steps = self.substep = 0
        self.in_policy = self.in_reset = self.complete = self.cleanup_done = False
        self.pending = None
        self.alive = {lane: True for lane in LANES}
        self.birth_ids = [1] * env.num_envs
        self.row_counts = {lane: 0 for lane in LANES}
        self.terminals, self.pre_reset, self.first_guard_crossings = {}, {}, {}
        self.max_gap = -1.
        self.contact_capacity_indicators = []
        self.trace = self.peak = None
        require(env.num_envs == 4096 and env.dt == .005 and env.policy_dt == .02 and env.decimation == 4, "Original physical clocks/population differ")
        require(env.cfg['position_iterations'] == 64 and env.cfg['velocity_iterations'] == 32
            and env.cfg['closure_gap_termination_m'] == GUARD_M, "Frozen solver/3mm guard differs")
        for lane, name, level, group in zip(LANES, PROFILES, LEVELS, GROUPS):
            r = env.recovery
            require(bool(r.enabled[lane]) and int(r.pose_curriculum.level_ids[lane]) == level
                and env.scene_groups[lane] == group
                and r.pose_curriculum.profile_names[int(r.pose_curriculum.profile_ids[lane])] == name, "Original lane/lesson/profile differs")
        self.ids = torch.tensor(LANES, device=env.device)
        self.contact_map = self._contact_mapping()
        metadata = dict(schema='source021-mechanical-capture-layout-v1', lanes=list(LANES), profiles=list(PROFILES),
            selected_env_origins_world_m=plain(env.origins[self.ids]),
            num_envs=4096, physics_dt_s=.005, policy_dt_s=.02, solver=[64,32], guard_m=GUARD_M,
            full_joint_names=list(env.robot.joint_names), control_joint_names=list(env.manifest['control_joint_names']),
            spring_joint_names=list(env.manifest['spring_joint_names']), body_names=list(env.robot.body_names),
            closure_constraints=plain(env.model_spec['constraints']),
            contact_mapping=self.contact_map, contact_scope='existing terrain-filtered patches; net force covers all contacts without their identity',
            self_contact_detail=dict(available=False, reason='Existing filtered views identify terrain, not robot/self pairs'),
            patch_coordinates='native PhysX coordinate values; original native support reader subtracts env.origins',
            impulse_scope='derived force * declared dt; not an independent impulse measurement',
            native_overflow_flag_available=False, PPO_updates=0, qualification=None)
        self.writer(output/'mechanical_layout.json', metadata)
        self.trace = gzip.open(output/'mechanical_5ms.jsonl.gz', 'wt', encoding='utf-8')
        self.peak = gzip.open(output/'mechanical_global_peaks.jsonl.gz', 'wt', encoding='utf-8')
        try:
            self._snapshot_initial()
        except BaseException:
            self.cleanup();raise

    def _contact_mapping(self):
        mappings = []
        for view_index, (ids, view) in enumerate(self.env.contact_views):
            members = plain(ids)
            sensors = list(view.sensor_paths)
            expected = [self.env.env_paths[lane]+'/Robot/'+body for lane in members for body in self.env.robot.body_names]
            require(sensors == expected and view.sensor_count == len(expected), 'Native sensor path/order differs from frozen env mapping')
            raw_filters = plain(view.filter_paths)
            flat_filters = set()
            ordered_filters = []
            def collect(value):
                if isinstance(value, str):
                    flat_filters.add(value); ordered_filters.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value: collect(item)
                else: raise ValueError('Unknown native filter path shape')
            collect(raw_filters)
            for lane in LANES:
                if lane not in members: continue
                paths = [self.env.env_paths[lane]+f'/Terrain/surface_{j}/geometry/mesh' for j in range(view.filter_count)]
                require(all(p in flat_filters for p in paths), 'Native terrain filter path absent')
                mappings.append(dict(lane=lane, view_index=view_index, local_lane_index=members.index(lane),
                    sensor_indices=[members.index(lane)*self.env.body_count+j for j in range(self.env.body_count)],
                    expected_filter_paths_by_source_index=paths,
                    filter_index_binding='unaltered frozen env group_filters order; actual filter path membership verified',
                    actual_sensor_paths=[sensors[index] for index in range(members.index(lane)*self.env.body_count,
                        (members.index(lane)+1)*self.env.body_count)],
                    actual_selected_filter_paths_in_native_order=[p for p in ordered_filters
                        if p.startswith(self.env.env_paths[lane]+'/Terrain/')],
                    native_filter_paths_sha256=hashlib.sha256(json.dumps(raw_filters,separators=(',',':')).encode()).hexdigest(),
                    selected_expected_filter_paths_present_in_native_paths=True))
        require(sorted(v['lane'] for v in mappings) == list(LANES), 'Each selected lane must map to exactly one existing native contact view')
        return mappings

    def _write(self, stream, value):
        stream.write(json.dumps(plain(value), separators=(',', ':'), allow_nan=False)+'\n')

    def state(self, lane, world, gaps):
        e, r = self.env, self.env.recovery
        data = e.robot.data
        return plain(dict(lane=lane, birth_id=self.birth_ids[lane], group=e.scene_groups[lane],
            full_q=data.joint_pos.torch[lane], full_dq=data.joint_vel.torch[lane],
            root_link_pose_world=data.root_link_pose_w.torch[lane],
            root_link_velocity_world=data.root_link_lin_vel_w.torch[lane],
            gyro_body=data.root_com_ang_vel_b.torch[lane], gravity_body=data.projected_gravity_b.torch[lane],
            body_link_poses_world=data.body_link_pose_w.torch[lane],
            pin_endpoints_world_m=world[lane], pin_gaps_m=gaps[lane],
            pin_argmax=int(gaps[lane].argmax()), pin_max_gap_m=gaps[lane].max(),
            applied_full_joint_effort=data.applied_torque.torch[lane],
            applied_effort_units='rotary N.m; prismatic gas joints N',
            phase=int(r.fsm.phase[lane]), route=int(r.fsm.route[lane]),
            release_finished=r.release_finished[lane], elapsed_s=r.elapsed[lane],
            blend=r.fsm.blend[lane], pure_rl=r.pure_rl[lane], scripted=r.fsm.scripted[lane],
            learning_mask_so_far=r.learning_mask[lane], policy_motion_hold_at_begin=r.policy_motion_hold[lane],
            observed_support_sensor=r.observed_support[lane], observed_height_sensor_m=r.observed_height[lane]))

    def _snapshot_initial(self):
        world, gaps = pin_geometry(self.env)
        for lane in LANES:
            self._write(self.trace, dict(physical_step=0, simulated_s=0., stage='after_sealed_native_reset',
                state=self.state(lane, world, gaps), applied_control=None, contacts=None,
                absent_fields='No physical interval has elapsed; no contact/effort inference'))
            self.row_counts[lane] += 1

    def control(self, target):
        e, r = self.env, self.env.recovery
        return plain(dict(full_joint_effort_target=target[self.ids],
            script_targets_control6=r.fsm.script_targets[self.ids],
            script_target_units='leg rad; wheel rad/s', native_requested_motor_effort_nm_at_setter=e.v5.requested_motor_effort[self.ids],
            native_request_scope='Frozen recovery.torque active-lane script/mixed preclip request override; not shaft torque',
            separately_persisted_script_preclip_request_available=False,
            motor_bounds_nm=e.v5.current_motor_bounds[self.ids], motor_strength=e.motor_strength[self.ids],
            spring_strength=e.spring_strength[self.ids], phase=r.fsm.phase[self.ids], route=r.fsm.route[self.ids],
            release_finished=r.release_finished[self.ids], blend=r.fsm.blend[self.ids],
            feedback_full_q_prephysics=e.robot.data.joint_pos.torch[self.ids],
            feedback_full_dq_prephysics=e.robot.data.joint_vel.torch[self.ids],
            delay_lags_5ms=(None if e.signal_delay is None else e.signal_delay.lags[self.ids]),
            transport_delay_s=(None if e.command_transport is None else e.command_transport.delay[self.ids]),
            transport_age_s=(None if e.command_transport is None else e.command_transport.age[self.ids]),
            target_scope='exact full effort passed to original setter, after native strength/transport/response; before PhysX'))

    def post_physics(self):
        require(self.in_policy and not self.in_reset and self.pending is not None, 'Physical sample lacks native interval effort')
        self.physics_steps += 1; self.substep += 1
        require(self.substep <= 4, 'More than four physical steps in native policy interval')
        world, gaps = pin_geometry(self.env)
        flat = int(gaps.argmax()); lane, pin = divmod(flat, gaps.shape[1]); maximum = float(gaps[lane,pin])
        if maximum > self.max_gap:
            self.max_gap = maximum
            self._write(self.peak, dict(physical_step=self.physics_steps, policy_tick=self.policy_steps,
                substep=self.substep, simulated_s=self.physics_steps*.005, lane=lane, pin=pin,
                birth_id=self.birth_ids[lane], gap_m=maximum,
                native_terminal_claim=False, state=self.state(lane, world, gaps)))
        contacts = {}
        if self.physics_steps <= TRACE_STEPS:
            for mapping in self.contact_map:
                lane = mapping['lane']
                if not self.alive[lane]: continue
                view = self.env.contact_views[mapping['view_index']][1]
                rows, capacity = contact_slices(view, mapping['sensor_indices'], .005, self.convert)
                if capacity['observed_count_at_capacity']:
                    self.contact_capacity_indicators.append(dict(lane=lane, physical_step=self.physics_steps, **capacity))
                contacts[lane] = dict(sensor_body_rows=rows, capacity_indicator=capacity,
                    existing_view_index=mapping['view_index'], env_contact_force_cache_consumed=False)
            for index, lane in enumerate(LANES):
                if not self.alive[lane]: continue
                if lane not in self.first_guard_crossings and float(gaps[lane].max()) > GUARD_M:
                    self.first_guard_crossings[lane] = dict(physical_step=self.physics_steps,
                        policy_tick=self.policy_steps, substep=self.substep, max_gap_m=float(gaps[lane].max()),
                        pin_argmax=int(gaps[lane].argmax()), native_terminal_claim=False)
                driven = {k:(v[index] if isinstance(v,list) and len(v)==len(LANES) else v) for k,v in self.pending.items()}
                self._write(self.trace, dict(physical_step=self.physics_steps, policy_tick=self.policy_steps,
                    substep=self.substep, simulated_s=self.physics_steps*.005,
                    stage='after_original_update_robot_data_before_native_contact_aggregation',
                    state=self.state(lane, world, gaps), applied_control=driven, contacts=contacts[lane]))
                self.row_counts[lane] += 1
        self.pending = None

    def before_reset(self, ids):
        members = plain(ids)
        if self.in_policy:
            world, gaps = pin_geometry(self.env)
            for lane in LANES:
                if lane in members and self.alive[lane]:
                    self.pre_reset[lane] = dict(physical_step=self.physics_steps, policy_tick=self.policy_steps,
                        stage='before_original_autoreset', state=self.state(lane, world, gaps))
                    self.alive[lane] = False
            for lane in members:
                self.birth_ids[lane] += 1

    def after_policy(self, result):
        require(self.substep == 4 and self.pending is None, 'Native physical update/effort cadence differs')
        diagnostic = result[3]['diagnostics']
        for lane in LANES:
            if lane in self.pre_reset and lane not in self.terminals:
                require(bool(diagnostic['done'][lane]), 'Reset was not a native terminal')
                self.terminals[lane] = plain(dict(policy_tick=self.policy_steps,
                    fields={k:v[lane] for k,v in diagnostic.items() if isinstance(v,torch.Tensor)},
                    reasons={k:bool(v[lane]) for k,v in diagnostic['reasons'].items()}))

    def install(self):
        e = self.env
        prior_step = e.step
        def step(actions):
            require(not self.in_policy, 'Nested native policy interval')
            self.in_policy = True; self.policy_steps += 1; self.substep = 0
            try:
                result = prior_step(actions)
                self.after_policy(result)
                return result
            finally:
                self.in_policy = False
        self.hooks.install(e, 'step', step)
        prior_reset = e.reset
        def reset(ids, *args, **kwargs):
            self.before_reset(ids)
            previous = self.in_reset; self.in_reset = True
            try: return prior_reset(ids, *args, **kwargs)
            finally: self.in_reset = previous
        self.hooks.install(e, 'reset', reset)
        prior_effort = e.robot.set_joint_effort_target_index
        def effort(*args, **kwargs):
            if self.in_policy and not self.in_reset:
                require(self.pending is None and set(kwargs) == {'target'} and not args,
                    'Native effort call differs or duplicate before physical step')
                self.pending = self.control(kwargs['target'])
            return prior_effort(*args, **kwargs)
        self.hooks.install(e.robot, 'set_joint_effort_target_index', effort)
        prior_update = self.module.update_robot_data
        def update(env, dt):
            result = prior_update(env, dt)
            if env is e and self.in_policy and not self.in_reset:
                require(dt == .005, 'Native physical dt changed')
                self.post_physics()
            return result
        self.hooks.install(self.module, 'update_robot_data', update)

    def finish(self):
        require(self.policy_steps == 600 and self.physics_steps == 2400 and self.pending is None, 'Original600/2400 loop not complete')
        require(all(lane in self.terminals or self.row_counts[lane] == TRACE_STEPS+1 for lane in LANES), 'Selected firstbirth12s capture incomplete')
        self.complete = True
        errors = self.cleanup()
        require(not errors, 'Mechanical hooks could not restore: '+str(errors))
        return self.summary()

    def summary(self):
        return dict(schema='source021-native-mechanical-capture-summary-v1', complete=self.complete,
            policy_steps=self.policy_steps, physics_steps=self.physics_steps, trace_cutoff_physics_steps=TRACE_STEPS,
            lane_row_counts=self.row_counts, native_first_terminals=self.terminals,
            pre_autoreset_snapshots=self.pre_reset, first_substep_guard_crossings=self.first_guard_crossings,
            full_window_max_pin_gap_m=self.max_gap, hooks_restored=self.hooks.closed,
            contact_count_at_capacity_indicators=self.contact_capacity_indicators,
            PPO_updates=0, optimizer_instances=0, qualification=None,
            limitations=['No self-contact pair identity in existing terrain-filtered views',
                'Postphysics getter before native contact aggregation; stale env.contact_force never consumed',
                'Impulse is force-times-dt derived evidence',
                'Substep threshold crossing is not itself a native policy terminal',
                'Native buffer overflow flag unavailable; counts/capacity and invalid slices recorded/rejected'])

    def cleanup(self):
        if self.cleanup_done: return []
        errors = self.hooks.restore()
        for stream in (self.trace, self.peak):
            if stream is not None: stream.close()
        self.cleanup_done = True
        if not self.complete or errors:
            self.writer(self.output/'mechanical_capture_failure.json', dict(status='incomplete_or_hook_restore_failed',
                errors=errors, summary=self.summary(), native_admission=False, qualification=None))
        return errors
