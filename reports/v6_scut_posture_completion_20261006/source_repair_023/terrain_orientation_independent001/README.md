# Terrain orientation independent CPU reproduction

2026-10-07；scope: CPU USD/API checks and CPU geometry reconstruction. No Kit,
PhysX, GPU work, PPO, optimizer, SSH or training lifecycle action was performed.
CUDA was hidden for the saved executions. These checks do not grant native
physics, task-performance or hardware qualification.

## Result and evidence

`execution.json` records the actual successful process, exit code 0, elapsed
2.4996 seconds, exact command, environment and output hashes. `stdout.json` is
its unedited JSON stdout; `stderr.txt` is empty. `reproduce.py` is the executed
script. It hashes its own source, the installed Isaac Lab implementation, frozen
v22 dependencies, current reviewed files and the original native failure report,
then verifies those files did not change during execution.

The frozen input is `/tmp/terrain-runout-frozen-v22-1j8x0cux`, commit identity
`5ab8b54ea78f532a847ce71ca11dc7679d02cc01` taken from the existing native report.
The worker and available dependency hashes are checked against that report;
the script does not independently infer a Git commit from an extracted tree.
The original native result remains failed at `terrain_runout_runtime.py:155`.

Installed implementation files are under:

`/home/yukikaze/Documents/workspace/robot_rl/.deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab/`

- `sim/spawners/shapes/shapes.py`: `spawn_cuboid` documents xyzw; it uses
  Cube size=min(dimensions), component scale=dimensions/min(dimensions), and
  forwards orientation through `_spawn_geom_from_prim_type` to `create_prim`.
- `sim/utils/prims.py`: `create_prim` forwards local orientation to
  `standardize_xform_ops`.
- `sim/utils/transforms.py`: executed directly by the small in-memory USD
  cases; it constructs `Gf.Quatd(orientation[3], orientation[0], orientation[1],
  orientation[2])`. SHA-256 is
  `ef40d7ddf3b4977b13a145bddc241abbb02efacc64d8dfe2ca23406273f07791`.

All three installed file hashes and all input hashes are in `stdout.json`.
The complete simulator spawner was not initialized or executed. The small cases
use its verified Cube sizing convention, the actual installed transform function,
and the unchanged frozen `_mesh_top_vertices` readback function.

## Mechanism

`Surface.box()` returns scalar-first wxyz. Frozen `ChassisEnv` forwarded it
unchanged to an xyzw native API. The old test fixture directly constructed a
correct `Gf.Quatd(w, x, y, z)`, bypassing and masking that API mismatch.

The saved script extracts both old and corrected orientation expressions from
the respective `env.py` ASTs. With old `quat`, positive 15-degree cross slope has
support-bound error 1.0482171326652088 m. Its [4, 6.2] extension has error
0.49841679378022086 m. With `(*quat[1:], quat[0])`, both pass the unchanged
1e-4 m tolerance. Tests include flat, solid step, positive/negative longitudinal
slope, positive/negative cross slope and extension.

For a flat symmetric box, the wrong 180-degree orientation swaps the identified
local top/bottom faces without moving the occupied box volume. Consequently the
old readback's 0.1 m top-face error is not evidence that flat ground physically
moved down by 0.1 m. Sloped geometry does change incorrectly. Solver 64/32
settings cannot correct a static USD orientation mismatch.

## Complete reconstruction scope and correction

The final CPU reconstruction follows frozen scene selection, difficulty levels,
terrain builders, the release-stop repeated terrain branch, runout extension,
and solid-step bottoms. It enumerates 4096 lanes, including 1724 selected runout
lanes and 150 lanes using release-stop repetition. It checks 16,720 Cuboid corner
sets; 1650 flat/jump triangle-mesh lanes are explicitly excluded from this
quaternion check. Contact-domain friction changes do not alter the corner
geometry; physical materials are outside this population enumeration.

Old orientation gives 232 support-bound failures, all cross slope. Corrected
orientation has maximum support error 8.881784197001252e-16 m and plane error
1.3877787807814457e-16 m. The first construction-order candidate is clone 1882,
`cross_slope__terrain_easy`, surface 0, negative 5 degrees, support error
0.352980758128016 m. It is also first under hypothetical lexicographic clone
ordering. This is a CPU prediction, not a native failing-prim readback; the
original native report did not record its failing prim or actual failure lane.

An earlier unsaved simplified enumeration reported 11,032 Cuboids and omitted
release-stop repetition. That number is superseded. The first saved attempt
explicitly asserted release-stop was disabled and failed before enumeration;
`attempt001_reproduce.py`, `attempt001_execution.json`, `attempt001_stdout.json`
and `attempt001_stderr.txt` preserve that failed attempt. The script was then
corrected to call frozen `release_stop_surfaces`, rather than removing the
coverage requirement. The quaternion diagnosis, 232 cross-slope failures and
first CPU candidate remain unchanged.

## Read-only review of the proposed source diff

No actionable defect found in the reviewed three-file diff:

- `env.py` changes only the Cuboid API boundary to xyzw. The extracted production
  expression is exercised by this report. No surface dimensions, translation,
  solver, thresholds or training settings are changed by that hunk.
- `task.py` adds only an explicit wxyz return-format docstring. Public geometry
  output remains unchanged; the direct-Gf live viewer remains correct.
- `test_terrain_runout_runtime.py` now extracts the production boundary expression,
  interprets it using the installed API's xyzw convention, and includes a negative
  control proving the old scalar-first pass-through fails coverage readback.

The fixture still represents CPU USD construction, not PhysX acceptance. This
independent report binds its orientation convention to the installed function.
The root agent reported 186 targeted CPU tests passing; that suite was not rerun
or counted as this independent script's own test result. Native geometry readback
and a bounded physical interval remain separate required evidence.
