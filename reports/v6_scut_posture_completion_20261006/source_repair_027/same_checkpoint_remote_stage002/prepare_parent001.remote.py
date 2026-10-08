from pathlib import Path
import json
parent=Path('/home/kaiser/robot-rl-sim60/assessments')
parent.mkdir(parents=True,exist_ok=True)
paths=['/home/kaiser/robot-rl-sim60/assessments/evaluator_source_16ed8f6_20261008', '/home/kaiser/robot-rl-sim60/assessments/evaluator_source_16ed8f6_20261008.tar.gz', '/home/kaiser/robot-rl-sim60/assessments/evaluator_source_16ed8f6_20261008.manifest.json', '/home/kaiser/robot-rl-sim60/assessments/cpu_preflight_39000_16ed8f6_20261008']
assert all(not Path(p).exists() for p in paths),"Independent staging path already exists; inspect instead of overwriting"
print(json.dumps({"parent":str(parent),"independent_paths_absent":True,"training_actions":0}))
