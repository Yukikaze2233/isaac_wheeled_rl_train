"""Read-only current resource snapshot, not a causal trial or training action."""
import datetime,json,os,subprocess,time
from pathlib import Path
pid=3662149; expected_start=96629601

def worker():
 raw=Path(f'/proc/{pid}/stat').read_text();s=raw[raw.rfind(')')+2:].split()
 if int(s[19])!=expected_start:raise RuntimeError('Original worker identity changed')
 status=Path(f'/proc/{pid}/status').read_text().splitlines();fields={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in status if ':' in x}
 return {'pid':pid,'starttime_ticks':int(s[19]),'state':s[0],'cpu_ticks':int(s[11])+int(s[12]),'threads':int(s[17]),'VmRSS':fields.get('VmRSS'),'VmSwap':fields.get('VmSwap'),'allowed_cpus':fields.get('Cpus_allowed_list')}

def cpu():
 s=Path('/proc/stat').read_text().splitlines()[0].split()[1:];v=[int(x) for x in s]
 return {'total':sum(v[:8]),'idle':v[3]+v[4],'iowait':v[4]}

def cli(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=10)
 return {'command':args,'exit_code':r.returncode,'stdout':r.stdout[:7000],'stderr':r.stderr[:1000]}

start=datetime.datetime.now(datetime.timezone.utc).isoformat();a=worker();ca=cpu();tick=time.monotonic();time.sleep(5);elapsed=time.monotonic()-tick;b=worker();cb=cpu();hz=os.sysconf('SC_CLK_TCK')
out={'read_started_at_utc':start,'read_ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_logical_cpus':os.cpu_count(),'sample_seconds':elapsed,'worker_before':a,'worker_after':b,'worker_cpu_percent_of_one_core':(b['cpu_ticks']-a['cpu_ticks'])/hz/elapsed*100,'host_busy_percent':100*(1-(cb['idle']-ca['idle'])/(cb['total']-ca['total'])),'host_iowait_percent':100*(cb['iowait']-ca['iowait'])/(cb['total']-ca['total']),'loadavg':Path('/proc/loadavg').read_text().strip(),'pressure':{name:Path('/proc/pressure/'+name).read_text().strip() for name in ['cpu','memory','io']},'GPU':cli(['nvidia-smi','--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw,power.limit','--format=csv']),'GPU_compute_processes':cli(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),'scope':'One current 5s read-only resource sample after actor6001–6100; no historical performance causality, no GPU workload, no process signal or remote write.','PPO_actions':0,'qualification':None}
print(json.dumps(out,indent=2))
