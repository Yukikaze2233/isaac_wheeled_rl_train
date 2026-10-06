#!/usr/bin/env python3
"""Root-owned, exclusive local diagnostic child; no PPO or process signals."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = '7081140dc8b54b48eeb647df578d6abfdfb5dee21c5a70c7d4b4e82d672cb437'
INVOCATION_SHA = '5749a5cc73e4a53a981194164edfd7db826c91b3fa557bf0535be543e866a1b4'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    assert sha(args.review) == args.review_sha256
    review = json.loads(args.review.read_text())
    assert review['candidate_manifest_sha256'] == MANIFEST_SHA
    assert review['PPO_updates'] == 0 and review['qualification'] is None
    assert review['all_bound_inputs_unchanged'] is True
    assert review['native_execution_admitted'] is True
    assert json.loads((HERE/'CPU_preflight_receipt_002.json').read_text())['exit_code'] == 0
    spec = importlib.util.spec_from_file_location('root_capture_bound', HERE/'run_capture_speed_022.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, manifest = module.verify(MANIFEST_SHA)
    assert sha(HERE/'invocation_022.json') == INVOCATION_SHA
    invocation = json.loads((HERE/'invocation_022.json').read_text())
    command = invocation['native_command']
    output = Path(command[command.index('--output')+1])
    assert output == HERE/'native_CAPTURE_SPEED_001' and not output.exists()
    assert not (HERE/'native_CAPTURE_SPEED_001.started.json').exists()
    capture = source.load(source.CAPTURE, 'root_capture_resources')
    pair = capture.load(capture.ORCHESTRATOR, 'root_capture_pair')
    resources = pair.local_GPU_empty()
    environment = dict(os.environ, **invocation['environment'])
    begin = time.monotonic()
    with (HERE/'native_CAPTURE_SPEED_001.stdout.log').open('x') as stream:
        child = subprocess.Popen(command, cwd=manifest['source_root'], env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
        proc = Path('/proc')/str(child.pid)
        stat = (proc/'stat').read_text()
        start = dict(pid=child.pid, command=command,
            starttime_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            candidate_manifest_sha256=MANIFEST_SHA,
            independent_review_sha256=args.review_sha256,
            root_launcher_sha256=sha(__file__), resource_check=resources,
            environment=invocation['environment'], cwd=manifest['source_root'],
            PPO_updates=0, formal_updates_consumed=0, qualification=None)
        write(HERE/'native_CAPTURE_SPEED_001.started.json', start)
        try:
            code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            write(HERE/'native_CAPTURE_SPEED_001.lifecycle_limit.json', dict(
                **start, status='root_attention_not_signalled', automatic_signal_sent=False))
            raise
    result = dict(**start, process_exit_observed=True, process_exit_code=code,
        process_present_after_wait=proc.exists(), process_wall_seconds=time.monotonic()-begin,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), App_close_returned=None)
    write(HERE/'native_CAPTURE_SPEED_001.process.json', result)
    print(json.dumps({key: result[key] for key in ('pid', 'process_exit_code', 'process_wall_seconds', 'PPO_updates')}))
    if code:
        raise RuntimeError('Native diagnostic failed; preserve evidence and do not retry')

if __name__ == '__main__':
    main()
