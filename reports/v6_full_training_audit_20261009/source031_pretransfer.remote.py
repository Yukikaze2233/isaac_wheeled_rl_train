from pathlib import Path
import hashlib,json
r=Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
paths=['source031_incoming001','source_serial_dispatch031_20261009','source031_deploy','source031-ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a-umask0022.tar.gz','deploy_source031_663efe61.py']
assert all(not (r/x).exists() for x in paths)
assert (r/'source_serial_dispatch030_20261009').is_dir()
owners=[]
for pid,start in [(4141407,113423130),(4141412,113423416),(4141576,113428247)]:
 p=Path('/proc')/str(pid);t=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert int(t[19])==start and t[0] not in ('Z','T')
 owners.append(dict(pid=pid,starttime=start,state=t[0]))
print(json.dumps(dict(status='source031_pretransfer_rejection_readback_no_mutation',new_paths_absent=paths,original_source030_preserved=True,original_training_owners=owners,remote_writes=0,PPO_updates=0)))
