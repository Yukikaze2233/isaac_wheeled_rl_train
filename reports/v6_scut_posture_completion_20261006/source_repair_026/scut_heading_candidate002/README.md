This is an unwired heading-command candidate. It adds no sampler, manager,
observation field or action channel. The pure function accepts explicitly owned
heading targets/masks and only modifies the 24 ordinary rows selected from the
frozen recipe. All eight explicit terrain bindings and dedicated/native-high
sequences are protected by the separate ordinary-row mask.

The equation follows SCUT `commands.py:577–604`: ordinary non-special heading,
wrap then stiffness and clip, standing last. The CPU oracle extracts the actual
source method AST and inspected IsaacLab `wrap_to_pi` AST with SHA checks,
without importing IsaacLab packages or constructing an App/Env. Exact `+pi` and
`-pi` conventions are checked. SCUT source copyright is SCUTRobotLab 2026, MIT;
the inspected wrap source is Isaac Lab Project Developers 2022–2026,
BSD-3-Clause. Input paths and complete file hashes are in `inputs001.json`.

Native commands are `[vx, yaw, height]`, so standing clears only the first two
columns. World heading is a command-generator input; the original native
35-dimensional actor builder receives resolved yaw speed only. No source model
or actor/critic/optimizer state is loaded.

No target/mask lifecycle, reset, RNG persistence, runtime hook or active special
selection is implemented. The tests depend on the frozen local source files
identified in the manifest. Existing command slew, feasibility/difficulty
projection and command precedence must remain when a future integration is
reviewed. Matching the source equation is not equality of final command
distributions after these transforms. No native/PPO/deployment qualification is
granted here.

`cpu001` retains a failed oracle fixture: it selected an earlier same-named
method instead of the source SpecialModeUniformVelocityCommand class method.
The fixture now selects the explicit class and checks its source line/special
attribute. The candidate resolver itself was unchanged by this repair.
`run_cpu_checks.py cpu002` produces a separate repair receipt without replacing
the first result.
