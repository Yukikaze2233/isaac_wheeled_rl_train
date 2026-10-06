# SIDE 6 → 1.5 rad/s mechanical counterfactual

This is a local artifact candidate, not a campaign, PPO branch, deployment, or
functional qualification. The native run is pending. Production source021,
contracts, formal continuous PPO, and the immutable003 capture are untouched.

The one control change is the instance-owned
`RecoveryFSM.config.side_speed_rad_s`, admitted only when its actual original
value is 6.0. It becomes 1.5 before the first policy interval. Every original
interval checks the entire config before and after its one original call. All
other config entries, the native FSM implementation, 4096 lanes, REC3076 mean
actor, full seal/RNG/std, 50 Hz policy, 200 Hz physics, solver64/32, 3 mm guard,
PD, springs, gravity, response/delays and profiles retain the original source.

The native formulas remain `duration = theta / speed`, the 80 ms turnaround
dwell, and `elapsed >= 2 * duration + .4`; the global script budget remains8s.
With theta=pi/2 the outward duration changes .261799 →1.047198s (4×), while
SIDE timeout changes .923599 →2.494395s (not4×). No additional clock override
or controller rewrite is present.

The frozen003 factory lifecycle is reused: create classes before Kit, resolve
the native environment/FSM only after the original observer/env is constructed.
The candidate restores its step wrapper before003 restores its inner wrappers,
then restores its owned config to6 before Kit.close. Applied and preclose
override receipts record the actual1.5 interval separately from the original
pre-override runtime preflight. A foreign config/hook mutation rejects instead
of being silently overwritten. Partial construction and late parent-finish
failure have CPU coverage; failed native observations remain preserved.

Measurement coverage expands from003's first260 physical steps to the selected
first birth's entire600 policy/2400 physics window, stopping at its native
pre-autoreset terminal. This changes one freshly loaded observer module's
TRACE_STEPS, not its files or any physical parameter/step. It captures actual
native side_start/direction/axis_signs/phase_ticks at each original effort
setter. There is no second FSM.update, controller calculation, RNG draw, or
physical step. Existing terrain contacts and all6 actual body-pose pin gaps
retain003's getter/clone semantics. Full-window peak lane/pin/birth remains
observed even after these two first births terminate.

Baseline is actual003, whose real exit0 process and independent readback are
bound in the manifest, with the same true ownREC3076 full seal. Its first SIDE
entries are1708/phys179 and2216/phys176. The read-only comparison checks every
captured selected row through each entry inclusively: state, phase/route,
pins, velocities, script references, efforts, ownership/masks, and detailed
terrain contacts use decoded numeric-bit equality, including signed zero.
Only added telemetry and the explicitly reported capacity count/at-capacity
indicator summing ALL contact-view lanes are excluded; selected patches and
forces remain checked. Failure preserves the experiment and rejects a clean
single-variable interpretation. PostSIDE terminal histories are expected to
differ and are not required to reproduce baseline guard times.

Coupling readback checks actual paired periodic relative target angle and one
signed common stroke across all4 motor roots, using only a float32 roundoff
bound. The native source owns theta/v and path interpolation; telemetry never
recomputes targets. Converted exit intervals whose setter phase is no longer
SIDE are explicitly outside this invariant's scope.

Run CPU and native in different NEW candidate-only output directories using
the commands in `invocation.json`. CPU mode authenticates the original source,
full REC seal and artifact bindings without constructing App or applying the
override. Native mode checks GPU0 has no compute owner and keeps the original
600/2400 driver. Only root launches it. Kit.close may directly exit Python;
postclose continuation is not evidence. Root records the actual PID/argv/exit,
then calls `--mode review --process-receipt <actual receipt>`. Review checks
the original driver, exact initial birth/private sampling/std, complete native
counts/ownership/payment/cleanup, override, preSIDE and coupling. It may
return2 for a valid preserved but rejected counterfactual; no result is
silently promoted or rewritten.

No-guard alone is not recovery success. Native timeout and12s unfinished
handoff/right-censoring remain distinct. Self-contact pair identity, native
overflow flags, and the actual randomized mass/COM/friction draw remain
unavailable. A changed impact outcome does not establish contact or solver as
the sole cause. The old unlocalized stochastic730mm peak stays unassigned.

Prepared scope:0 PPO,0 formal consumption, no SSH/GPU launch, no production
edit, qualification=null. Next root-owned step is one exclusive native mean
counterfactual after independent CPU review; preserve all negative outcomes.
