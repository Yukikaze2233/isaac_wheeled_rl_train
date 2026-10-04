import json, statistics, time
from pathlib import Path
import torch

N, DT, STEPS = 4096, .005, 800

def init(device, fraction, hold_steps=3000):
    selected = torch.arange(N, device=device) < round(N * fraction)
    return dict(remaining=torch.where(selected, hold_steps, 0).long(),
                elapsed=torch.zeros(N, device=device), episode=torch.zeros(N, dtype=torch.long, device=device), selected=selected)

def old(s):
    held = s['remaining'] > 0
    s['elapsed'][held] += DT
    s['remaining'].sub_(held.long())
    s['episode'].add_(s['selected'].long())

def candidate(s):
    held = s['remaining'] > 0
    s['elapsed'].add_(held.to(s['elapsed'].dtype) * DT)
    s['remaining'].sub_(held.long())
    s['episode'].add_(s['selected'].long())

def benchmark(fn, fraction):
    results=[]
    for rep in range(4):
        s=init('cuda', fraction)
        for _ in range(80): fn(s)
        torch.cuda.synchronize()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start=time.perf_counter(); begin.record()
        for _ in range(STEPS): fn(s)
        end.record(); end.synchronize()
        results.append({'host_wall_us_per_step':(time.perf_counter()-start)*1e6/STEPS,
                        'cuda_event_interval_us_per_step':begin.elapsed_time(end)*1000/STEPS})
    return {key:statistics.median(r[key] for r in results) for key in results[0]} | {'repeats':results}

def profile(fn, fraction):
    s=init('cuda', fraction)
    for _ in range(30): fn(s)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(16):
            with torch.profiler.record_function('advance_step'): fn(s)
        torch.cuda.synchronize()
    keys=[]
    for row in p.key_averages():
        if row.key in ('advance_step','aten::nonzero','aten::index','aten::index_put_','aten::_index_put_impl_',
                       'cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync'):
            keys.append({'key':row.key,'count':row.count,'cpu_time_total_us':row.cpu_time_total,
                         'self_cpu_time_total_us':row.self_cpu_time_total,'device_time_total_us':row.device_time_total})
    return {'steps':16,'explicit_device_sync_at_end':1,'operators':keys}

def verify(device, fraction, hold_steps):
    states=[init(device,fraction,hold_steps) for _ in range(2)]
    filters=[torch.full((N,6),.2,device=device) for _ in range(2)]
    first_driven=[None,None]
    all_equal=True
    for tick in range(96):
        if tick % 4 == 0:
            learn=[~s['remaining'].gt(0) for s in states]
            interval=[torch.zeros(N,dtype=torch.bool,device=device) for _ in states]
        outputs=[]
        for i,(fn,s) in enumerate(zip((old,candidate),states)):
            held=s['remaining']>0
            pre=torch.where(held[:,None],0.,torch.full_like(filters[i],1.))
            filters[i].mul_(.9).add_(pre*.1)
            effort=torch.where(held[:,None],0.,filters[i])
            interval[i] |= held
            learn[i] &= ~held
            if first_driven[i] is None and bool((effort[s['selected']]!=0).any()): first_driven[i]=tick
            fn(s)
            outputs.append(effort)
        all_equal &= torch.equal(outputs[0],outputs[1])
        all_equal &= all(torch.equal(states[0][k],states[1][k]) for k in ('remaining','elapsed','episode','selected'))
        all_equal &= torch.equal(learn[0],learn[1]) and torch.equal(interval[0],interval[1])
        # Returned-observation bootstrap can resume while the elapsed action interval remains excluded.
        all_equal &= torch.equal(states[0]['remaining']<=0,states[1]['remaining']<=0)
    assert all_equal
    if fraction: assert first_driven==[hold_steps,hold_steps]
    return {'device':device,'held_fraction':fraction,'hold_steps':hold_steps,'hold_seconds':hold_steps*DT,
            'substeps':96,'policy_decimation':4,'bitwise_equal_at_every_substep':all_equal,
            'first_driven_substep_zero_based':first_driven}

result={'schema':'nonstandard-advance-local-microbenchmark-v1','created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'num_envs':N,'physics_dt':DT,'policy_dt':.02,
        'scope':'Torch only, no Isaac scene, no Kaiser access, no frozen source changes','measurements':[], 'verification':[],
        'limitations':['CUDA event interval includes CPU launch stalls, not just summed kernel execution.',
                       'Profiler adds instrumentation cost; use it for operator and synchronization counts.',
                       'This isolates NonstandardStart.advance, not full PPO or simulation speed.']}
for device in ('cpu','cuda'):
    for fraction in (0.,.01,1.):
        for hold_steps in (39,40,41): result['verification'].append(verify(device,fraction,hold_steps))
for fraction in (0.,.01,1.):
    row={'held_fraction':fraction,'held_lanes':round(N*fraction)}
    for label,fn in (('old',old),('full_length_candidate',candidate)):
        row[label]=benchmark(fn,fraction)
        row[label]['profile']=profile(fn,fraction)
    row['local_wall_speedup']=row['old']['host_wall_us_per_step']/row['full_length_candidate']['host_wall_us_per_step']
    result['measurements'].append(row)
path=Path('/tmp/v6_nonstandard_advance_benchmark_20261005.json'); path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'report':str(path),'verification_cases':len(result['verification']),'all_bitwise_equal':all(v['bitwise_equal_at_every_substep'] for v in result['verification']),
                 'measurements':result['measurements']},indent=2))
