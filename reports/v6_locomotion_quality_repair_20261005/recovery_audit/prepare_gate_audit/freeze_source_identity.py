#!/usr/bin/env python3
"""Bind one complete clean prefix H/C checkout to an external JSON manifest.

No source changes, scene imports, GPU, SSH, hardware or training occur here.
The common harness revision is supplied explicitly, never guessed from HEAD.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--harness-revision', required=True)
    parser.add_argument('--mode', choices=('baseline', 'prepare_slew_1_candidate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Manifest must be outside the immutable source checkout')
    if output.exists():
        raise ValueError('External source manifest already exists; do not overwrite')
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True).stdout
    path = root / 'scripts/probe_recovery_physical_prefix.py'
    spec = importlib.util.spec_from_file_location('prefix_manifest_probe', path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)  # imports only CPU support; never AppLauncher
    support = sys.modules['recovery_prefix_support']
    revision = git('rev-parse', 'HEAD').decode().strip()
    actual = probe.sources()
    baseline = {name: hashlib.sha256(git('show', f'{support.CORE_REVISION}:{name}')).hexdigest()
                for name in support.CORE_FILES}
    identity = {'schema': 'recovery-prefix-source-identity-v1', 'mode': args.mode, 'source_root': str(root),
        'base_core_revision': support.CORE_REVISION, 'harness_revision': args.harness_revision,
        'source_revision': revision, 'base_core_files_sha256': baseline,
        'core_files_sha256': {name: actual[name] for name in support.CORE_FILES},
        'runtime_sources_sha256': actual,
        'candidate_change': None if args.mode == 'baseline' else support.PREPARE_CHANGE,
        'candidate_diff_sha256': hashlib.sha256(git('diff', '--no-ext-diff', '--no-textconv', args.harness_revision, revision)).hexdigest(),
        'engineering_expectations_sha256': support.ENGINEERING_EXPECTATIONS_SHA}
    output.parent.mkdir(parents=True, exist_ok=True)
    support.write_json(output, identity)
    digest = support.digest(output)
    try:
        receipt = probe.core_sources(root, identity_path=output, identity_sha256=digest, actual_sources=actual)
    except Exception:
        output.unlink()  # never leave a failed identity masquerading as a frozen manifest
        raise
    print(json.dumps({'status': 'external_source_identity_authenticated', 'manifest': str(output),
        'manifest_sha256': digest, 'mode': receipt['mode'], 'source_revision': receipt['revision'],
        'base_revision': receipt['base_revision'], 'harness_revision': receipt['harness_revision'],
        'core_files': len(receipt['core_files_sha256']), 'runtime_files': len(receipt['runtime_sources_sha256']),
        'candidate_diff_sha256': receipt['candidate_diff_sha256'],
        'engineering_expectations_sha256': receipt['engineering_expectations_sha256']}))


if __name__ == '__main__':
    main()
