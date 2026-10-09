# Periodic sealed speed endpoints

This adapter accepts a genuine periodic speed26 export from the fixed REC41393
learning origin. It preserves the e5 runtime, original full-role contract checks,
shared 35/81/6 ABI, per-file checkpoint seal, PT/ONNX provenance, and deterministic
native actor comparison. CPU preparation does not qualify a policy or run a scene.

The checkpoint save clock is total successful learning updates, at interval 500.
The parent has consumed 41393 updates; the speed transition pays 10 critic updates.
For an authenticated total `T`, the speed actor count is `A = T - 41393 - 10`.
Actor consumption is `41223 + A`, stage actor is `27556 + A`, and learning lineage
is `27646 + 10 + A`. Both integrated-lineage actor fields must equal `A`, and
`domain_complete` must be exactly false. Every clock must be an integer.

The requested actor500 point is total41903, which is not periodic. The first
periodic endpoint is total42000/actor597. Admission had 5093 speed updates already
charged against the 20000 domain cap, leaving 14907 learning updates and 14897
actor updates after critic adaptation. This adapter therefore accepts only
periodic totals from 42000 through 56000 with the exact clocks and origin. The
final budget point total56300 is not a periodic endpoint; total56500 exceeds the
admitted actor allowance. Arithmetic eligibility does not authenticate a file.

| Total learning | Speed actor | Actor consumed | Stage actor | Lineage |
| ---: | ---: | ---: | ---: | ---: |
| 42000 | 597 | 41820 | 28153 | 28253 |
| 43500 | 2097 | 43320 | 29653 | 29753 |

Both preparation entry points report the authenticated selected endpoint and
retain the first-periodic endpoint separately. The existing smoke mode remains
an explicit same-REC21 interface check and cannot execute a physical comparison.
No control, observation, command, environment, or recovery behavior changes.

CPU validation uses the local original sealed REC41393, speed42000, and speed43500
exports. Tests reject individual clock drift, unpaid critic clocks, early or
off-cycle totals, over-budget totals, wrong learning origin, completed lineage,
float/bool clocks, missing per-file seals, actor-only checkpoints, and disagreement
between the seal and checkpoint total. Existing native driver tests use CPU
fixtures; they do not start a simulator.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /home/yukikaze/isaacsim60-venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_rec_speed_mean_replay.py tests/test_rec_speed_mean_native_driver.py \
  tests/test_rec_speed_mean_independent_world.py tests/test_rec_speed_periodic_endpoints.py
```
