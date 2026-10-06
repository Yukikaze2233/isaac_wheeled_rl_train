"""One engineering mean-only observation; not a learning parent or PPO run."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
AUDIT = OUT.parent / 'learning_audit' / 'hold_cost_candidate_021'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta = json.loads((OUT / 'source_candidate_021.json').read_text())
    root = Path(meta['clean_root'])
    pt = REPO / 'models/v6_model_delivery_20261003T2235/recommended_flat/model_final.pt'
    assert sha(pt) == '7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f'
    inventory = Path(meta['source_inventory_file'])
    assert sha(inventory) == meta['source_inventory_sha256']
    worker = root / 'contracts/v6_task_campaign_recovery_worker_v19.json'
    admission = dict(schema='source021-native-teaching-admission-v1', context='engineering_flat_prefix',
                     source_commit=meta['source_commit'], source_root=str(root),
                     source_manifest=str(inventory), source_manifest_sha256=sha(inventory),
                     contract_relative=str(worker.relative_to(root)), contract_sha256=sha(worker),
                     checkpoint=str(pt), checkpoint_sha256=sha(pt),
                     checkpoint_contract_sha256=sha(pt.with_suffix('.contract.json')),
                     checkpoint_seal_sha256=sha(pt.with_suffix('.checkpoint.json')),
                     seed=290927, block_index=0)
    parent = OUT / 'native_engineering_parent_021.json'
    target = OUT / 'native_teaching_021'
    assert not parent.exists() and not target.exists(), 'Existing observation: inspect, never overwrite'
    parent.write_text(json.dumps(admission, indent=2) + '\n')
    driver = AUDIT / 'validate_native_teaching_021.py'
    command = ['/home/yukikaze/isaacsim60-venv/bin/python', '-B', str(driver),
               '--admission', str(parent), '--admission-sha', sha(parent), '--arm', 'mean',
               '--output', str(target)]
    invocation = dict(source_commit=meta['source_commit'], archive_sha256=meta['archive_sha256'],
                      driver_sha256=sha(driver), parent_admission_sha256=sha(parent), command=command,
                      scope='engineering_flat_prefix_not_formal_learning_parent',
                      formal_updates_consumed=0, qualification=None)
    (OUT / 'native_teaching_021.invocation.json').write_text(json.dumps(invocation, indent=2) + '\n')
    started = time.monotonic()
    with (OUT / 'native_teaching_021.stdout.log').open('x') as stream:
        result = subprocess.run(command, cwd=root,
                                env=dict(os.environ, OMP_NUM_THREADS='1',
                                         OMNI_KIT_ACCEPT_EULA='YES', PYTHONDONTWRITEBYTECODE='1'),
                                stdout=stream, stderr=subprocess.STDOUT)
    assert sha(driver) == invocation['driver_sha256'], 'Observer changed while running'
    # Independent terminal readback checks actual native counts, payments,
    # ordinary/excluded intervals, source bytes and all three original goals.
    import importlib.util
    spec = importlib.util.spec_from_file_location('source021_teaching_readback', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = module.review_output(target)
    assert result.returncode == 0 and receipt['all_required_passed'] is True
    receipt.update(status='passed', source_commit=meta['source_commit'],
                   archive_sha256=meta['archive_sha256'], process_exit_code=result.returncode,
                   process_wall_seconds=time.monotonic() - started,
                   driver_sha256=sha(driver), invocation_sha256=sha(OUT / 'native_teaching_021.invocation.json'))
    (OUT / 'native_teaching_readback_021.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
