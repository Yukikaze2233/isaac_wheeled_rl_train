"""Bounded /proc and nvidia-smi observation; no training or CUDA calls."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

OWNERS = {4141407: 113423130, 4141412: 113423416,
          4141576: 113428247, 4166067: 114488334,
          4166078: 114488781}

def proc(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    start = int(stat[19])
    if pid in OWNERS and start != OWNERS[pid]:
        raise RuntimeError(f'Original process identity changed: {pid}')
    status = (root / 'status').read_text().splitlines()
    return dict(pid=pid, starttime=start, state=stat[0], ppid=int(stat[1]),
                cpu_ticks=int(stat[11]) + int(stat[12]),
                resources=[s for s in status if s.startswith(('VmRSS:', 'Threads:'))])

def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except OSError as error:
        return dict(exit_code=None, stdout='', stderr=str(error))
    return dict(exit_code=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())

first = {pid: proc(pid) for pid in OWNERS}
start = time.monotonic()
time.sleep(2)
elapsed = time.monotonic() - start
second = {pid: proc(pid) for pid in OWNERS}
for pid, row in second.items():
    row['sample_cpu_percent_one_core'] = (
        (row['cpu_ticks'] - first[pid]['cpu_ticks']) /
        os.sysconf('SC_CLK_TCK') / elapsed * 100)

nvidia_smi = shutil.which('nvidia-smi') or next((p for p in (
    '/usr/lib/wsl/lib/nvidia-smi', '/usr/bin/nvidia-smi',
    '/usr/local/cuda/bin/nvidia-smi',
    '/home/kaiser/robot-rl-sim60/sysroot/usr/bin/nvidia-smi')
    if Path(p).is_file()), '/usr/bin/nvidia-smi')
gpu = command([nvidia_smi, '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit',
               '--format=csv,noheader,nounits'])
apps = command([nvidia_smi, '--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits'])
gpu_processes = []
if apps['exit_code'] == 0:
    for line in apps['stdout'].splitlines():
        try:
            pid = int(line.split(',', 1)[0])
            row = proc(pid)
            root = Path('/proc') / str(pid)
            argv = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            row['comm'] = (root / 'comm').read_text().strip()
            row['script_paths'] = [arg for arg in argv if arg.endswith('.py')]
            row['cwd'] = str((root / 'cwd').resolve(strict=True))
            row['parent_comm'] = (Path('/proc') / str(row['ppid']) / 'comm').read_text().strip()
            row['inline_python'] = '-c' in argv
            gpu_processes.append(row)
        except (OSError, ValueError) as error:
            gpu_processes.append(dict(pid=pid, observation_error=str(error)))
memory = {line.split(':')[0]: line.split(':', 1)[1].strip()
          for line in Path('/proc/meminfo').read_text().splitlines()
          if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:'))}
disk = shutil.disk_usage('/home/kaiser/robot-rl-sim60')
print(json.dumps(dict(status='read_only_bounded_resource_observation',
    at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    original_processes=list(second.values()), sample_seconds=elapsed,
    host_load_average=os.getloadavg(), logical_cpu_count=os.cpu_count(),
    memory=memory, disk=dict(total=disk.total, used=disk.used, free=disk.free),
    gpu=gpu, compute_apps=apps, gpu_processes=gpu_processes,
    remote_writes=0, training_signals=0,
    PPO_updates_by_reader=0, CUDA_calls_by_reader=0,
    scope='Instantaneous resource sample; not a causal profile of the previous 100 updates.')))
