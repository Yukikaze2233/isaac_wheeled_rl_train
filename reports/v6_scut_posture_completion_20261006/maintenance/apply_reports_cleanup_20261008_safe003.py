#!/usr/bin/env python3
"""Once-only, local, verified cleanup of explicit historical report artifacts."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
WORKSPACE = Path('/home/yukikaze/Documents/workspace/robot_rl')
ROOTS = tuple(WORKSPACE / name / 'reports' for name in
              ('isaac_wheeled_rl_train', 'isaac_wheeled_rl_schedule'))
INPUTS = (
    ('dedup', Path('/tmp/robot_rl_reports_cleanup_inventory_20261008.safe002.json'),
     '839bb2d0c87b808e5d97876ceea359cb595a837398677f041a37ba878325bf72'),
    ('frames', Path('/tmp/robot_rl_reports_cleanup_frames_20261008.safe002.json'),
     '569840e381185f785b28ca4b720b22d236de01fd39051fd01275629386376d9c'),
)
RESULT = BASE / 'reports_cleanup_20261008_safe003.result.json'
JOURNAL = BASE / 'reports_cleanup_20261008_safe003.actions.jsonl'
PROTECTED = ('source_repair_023', 'latest100', 'native004', 'native_004',
             'current_live', 'current_launch')
MODEL_SUFFIXES = {'.pt', '.onnx', '.bag', '.mcap', '.db3', '.stl', '.usd', '.usdc',
                  '.usda', '.urdf', '.srdf', '.npy', '.obj', '.xml', '.gz', '.zst'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_result(result):
    temporary = RESULT.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, RESULT)


def journal(action):
    action['recorded_at_utc'] = now()
    with JOURNAL.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(action, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def path_allowed(path, *, allow_directory=False):
    assert path.is_absolute() and path.resolve() == path, str(path)
    assert any(path.is_relative_to(root) for root in ROOTS), str(path)
    assert not any(value in str(path) for value in PROTECTED), str(path)
    assert not any(value.lower() in {'model', 'models', 'bag', 'bags', 'checkpoints'}
                   for value in path.parts), str(path)
    if not allow_directory:
        assert path.suffix.lower() not in MODEL_SUFFIXES, str(path)
    return path


def current(path):
    value = path.lstat()
    assert stat.S_ISREG(value.st_mode) and not path.is_symlink(), str(path)
    return value


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_mode, value.st_uid, value.st_gid)


def check_row(row, *, include_mode=True):
    path = path_allowed(Path(row['path']))
    value = current(path)
    keys = ('device', 'inode', 'size_bytes', 'mtime_ns')
    actual = identity(value)[:4]
    assert actual == tuple(row[key] for key in keys), str(path)
    if include_mode:
        assert identity(value)[4:] == tuple(row[key] for key in ('mode', 'uid', 'gid'))
    assert value.st_nlink == row['nlink'] == 1, str(path)
    return path, value


def open_inodes_and_paths(paths):
    """Read accessible /proc only. Never signal or inspect remote processes."""
    open_inodes = set()
    path_hits = []
    checked = 0
    directories = tuple(str(p) for p in paths if p.is_dir())
    exact = {str(p) for p in paths}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = str((process / 'cwd').resolve())
            checked += 1
            if any(value in command for value in exact) or cwd in exact or any(
                    cwd == value or cwd.startswith(value + '/') for value in directories):
                path_hits.append({'pid': int(process.name), 'kind': 'argv_or_cwd'})
            for descriptor in (process / 'fd').iterdir():
                try:
                    value = descriptor.stat()
                    open_inodes.add((value.st_dev, value.st_ino))
                    target = str(descriptor.resolve())
                    if target in exact or any(target.startswith(value + '/') for value in directories):
                        path_hits.append({'pid': int(process.name), 'kind': 'fd', 'path': target})
                except OSError:
                    pass
        except OSError:
            continue
    return open_inodes, {'accessible_processes': checked, 'path_hits': path_hits}


def tree_snapshot(directory):
    assert directory.is_dir() and not directory.is_symlink()
    files = []
    seen = set()
    allocated = 0
    apparent = 0
    for parent, names, leaves in os.walk(directory, followlinks=False):
        for name in names:
            assert not (Path(parent) / name).is_symlink()
        for leaf in leaves:
            path = Path(parent) / leaf
            value = current(path)
            assert value.st_nlink == 1, str(path)
            assert path.suffix.lower() not in MODEL_SUFFIXES, str(path)
            key = value.st_dev, value.st_ino
            assert key not in seen
            seen.add(key)
            allocated += value.st_blocks * 512
            apparent += value.st_size
            files.append({'path': str(path), 'identity': identity(value),
                          'allocated_bytes': value.st_blocks * 512, 'sha256': digest(path)})
    return {'files': files, 'file_count': len(files), 'allocated_bytes': allocated,
            'apparent_bytes': apparent}


def free_bytes():
    value = os.statvfs(ROOTS[0])
    return value.f_bavail * value.f_frsize


def preflight(inputs):
    dedup, frame_input = inputs['dedup'], inputs['frames']
    chosen, skipped = [], []
    candidate_paths = []
    for group in dedup['exact_duplicate_hardlink_groups']:
        rows = [group['canonical'], *group['replace_aliases']]
        values = [check_row(row) for row in rows]
        attrs = {(s.st_dev, s.st_mode, s.st_uid, s.st_gid) for _, s in values}
        if len(attrs) != 1:
            skipped.append({'kind': 'dedup_group', 'canonical': str(values[0][0]),
                            'reason': 'mode_uid_gid_or_device_mismatch',
                            'reclaim_bytes_not_taken': group['reclaim_allocated_bytes'],
                            'paths_and_attrs': [{'path': str(p), 'device': s.st_dev,
                                                'mode': s.st_mode, 'uid': s.st_uid,
                                                'gid': s.st_gid} for p, s in values]})
            continue
        for path, value in values:
            assert digest(path) == group['sha256'], str(path)
            assert identity(current(path)) == identity(value), str(path)
        chosen.append((group, values))
        candidate_paths.extend(path for path, _ in values)

    frames = []
    for item in frame_input['verified_frames_directories']:
        directory = path_allowed(Path(item['frames_directory']), allow_directory=True)
        assert directory.name == 'viewport_frames' and directory.is_dir()
        assert directory.is_relative_to(ROOTS[0])
        rows = item['delete_files']
        retained = item['retain_endpoint_pngs']
        assert len(retained) == 2 and rows
        indexed = sorted([*rows, *retained], key=lambda row: Path(row['path']).name)
        assert len(indexed) == item['capture_frame_count']
        assert [Path(row['path']).name for row in indexed] == [
            f'frame_{number:04d}.png' for number in range(len(indexed))]
        assert [Path(row['path']).name for row in retained] == [
            indexed[0]['path'].split('/')[-1], indexed[-1]['path'].split('/')[-1]]
        assert {str(path) for path in directory.iterdir()} == {row['path'] for row in indexed}
        for row in indexed:
            path, value = check_row(row, include_mode=False)
            assert path.parent == directory and path.suffix.lower() == '.png'
            assert digest(path) == row['sha256'], str(path)
            assert identity(current(path)) == identity(value)
        video_row = item['retain_video_verification']['video']
        video, _ = check_row(video_row, include_mode=False)
        assert video.suffix.lower() == '.mp4' and digest(video) == video_row['sha256']
        command = item['retain_video_verification']['ffprobe_command']
        assert command[0] == 'ffprobe' and command[-1] == str(video)
        probe = subprocess.run(command, capture_output=True, text=True, check=True)
        output = json.loads(probe.stdout)
        expected = item['capture_frame_count'] - 1
        assert int(output['streams'][0]['nb_frames']) == expected
        assert output['streams'][0]['avg_frame_rate'] == f"{item['fps']}/1"
        assert abs(float(output['format']['duration']) - expected / item['fps']) < 1e-4
        capture, _ = check_row(item['capture'], include_mode=False)
        assert digest(capture) == item['capture']['sha256']
        assert json.loads(capture.read_text())['frame_count'] == len(indexed)
        frames.append((item, rows, retained))
        candidate_paths.extend(Path(row['path']) for row in indexed)
        candidate_paths.extend((directory, video, capture))

    caches = []
    for item in frame_input['cache_directories']:
        directory = path_allowed(Path(item['delete_directory']), allow_directory=True)
        assert directory.name in {'.mplconfig', '__pycache__', '.pytest_cache',
                                  'client-profile', 'client-profile-02'}
        snapshot = tree_snapshot(directory)
        assert all(snapshot[key] == item[key] for key in
                   ('file_count', 'allocated_bytes', 'apparent_bytes'))
        if directory.name == '__pycache__':
            assert all(Path(row['path']).suffix == '.pyc' for row in snapshot['files'])
        caches.append((item, snapshot))
        candidate_paths.append(directory)

    opened, proc = open_inodes_and_paths(candidate_paths)
    assert not proc['path_hits'], proc
    selected = []
    for group, values in chosen:
        if any((value.st_dev, value.st_ino) in opened for _, value in values):
            skipped.append({'kind': 'dedup_group', 'canonical': str(values[0][0]),
                            'reason': 'candidate_inode_open_in_local_process',
                            'reclaim_bytes_not_taken': group['reclaim_allocated_bytes']})
        else:
            selected.append((group, values))
    assert all((current(Path(row['path'])).st_dev, current(Path(row['path'])).st_ino)
               not in opened for _, rows, retained in frames for row in rows + retained)
    assert all((row['identity'][0], row['identity'][1]) not in opened
               for _, snapshot in caches for row in snapshot['files'])
    return selected, skipped, frames, caches, candidate_paths, proc


def main():
    assert not RESULT.exists() and not JOURNAL.exists(), 'Once-only cleanup already has a receipt/journal'
    assert not (BASE / 'report_dedup_20261008.result.json').exists(), 'Unexpected old mutation result'
    lock = (BASE / 'reports_cleanup_20261008_safe003.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    inputs = {}
    for name, path, expected in INPUTS:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        inputs[name] = json.loads(raw)
        (BASE / f'reports_cleanup_20261008_safe003.{name}.inventory.json').write_bytes(raw)
    chosen, skipped, frames, caches, paths, proc = preflight(inputs)
    result = {'schema': 'actual-local-historical-reports-cleanup-v3', 'started_at_utc': now(),
              'status': 'preflight_verified_before_mutations', 'input_manifests': [
                  {'name': name, 'sha256': sha, 'original': str(path)} for name, path, sha in INPUTS],
              'script_sha256': digest(Path(__file__).resolve()),
              'old_root_attempt': 'No report_dedup_20261008.result.json present; prior preflight failed before mutations.',
              'PPO_actions': 0, 'remote_operations': 0,
              'all_model_ONNX_bag_mechanical_assets_and_current_source023_preserved': True,
              'protected_tokens': list(PROTECTED), 'proc_preflight': proc,
              'linked_aliases': [], 'deleted_middle_pngs': [], 'deleted_caches': [],
              'retained_video_and_endpoint_pngs': [], 'skipped': skipped,
              'dedup_allocated_bytes_released': 0, 'png_allocated_bytes_released': 0,
              'cache_allocated_bytes_released': 0, 'free_bytes_before': free_bytes()}
    save_result(result)
    try:
        opened, immediately_before = open_inodes_and_paths(paths)
        assert not immediately_before['path_hits']
        result['proc_immediately_before_mutations'] = immediately_before
        for group, values in chosen:
            canonical, original_canonical = values[0]
            assert identity(current(canonical)) == identity(original_canonical)
            assert (original_canonical.st_dev, original_canonical.st_ino) not in opened
            for alias, original_alias in values[1:]:
                value = current(alias)
                assert identity(value) == identity(original_alias) and value.st_nlink == 1
                assert (value.st_dev, value.st_ino) not in opened
                temporary = alias.with_name(alias.name + '.cleanup-safe003-hardlink-tmp')
                assert not temporary.exists()
                action = {'kind': 'hardlink_exact_duplicate', 'alias': str(alias),
                          'canonical': str(canonical), 'sha256': group['sha256'],
                          'old_identity': identity(original_alias),
                          'freed_allocated_bytes': original_alias.st_blocks * 512}
                journal({'state': 'intent', **action})
                os.link(canonical, temporary)
                try:
                    assert identity(current(alias)) == identity(original_alias)
                    os.replace(temporary, alias)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                new = current(alias)
                assert (new.st_dev, new.st_ino) == (original_canonical.st_dev, original_canonical.st_ino)
                action['new_inode'] = new.st_ino
                result['linked_aliases'].append(action)
                result['dedup_allocated_bytes_released'] += original_alias.st_blocks * 512
                journal({'state': 'applied_verified_inode', **action})
            assert digest(canonical) == group['sha256']
            save_result(result)

        for item, rows, retained in frames:
            # PNG paths are disjoint from hardlink candidates. Reconfirm before unlink.
            for row in rows:
                path, value = check_row(row, include_mode=False)
                assert (value.st_dev, value.st_ino) not in opened
                action = {'kind': 'remove_obsolete_middle_png', 'path': str(path),
                          'sha256': row['sha256'], 'old_inode': value.st_ino,
                          'freed_allocated_bytes': value.st_blocks * 512}
                journal({'state': 'intent', **action})
                path.unlink()
                assert not path.exists()
                result['deleted_middle_pngs'].append(action)
                result['png_allocated_bytes_released'] += value.st_blocks * 512
                journal({'state': 'applied_verified_absent', **action})
            preserved = {'frames_directory': item['frames_directory'],
                         'video': item['retain_video_verification']['video'],
                         'endpoint_pngs': retained, 'capture': item['capture'],
                         'removed_png_count': len(rows)}
            result['retained_video_and_endpoint_pngs'].append(preserved)
            for row in [*retained, item['capture'], item['retain_video_verification']['video']]:
                path, _ = check_row(row, include_mode=False)
                assert digest(path) == row['sha256']
            save_result(result)

        for item, snapshot in caches:
            directory = Path(item['delete_directory'])
            current_tree = tree_snapshot(directory)
            assert current_tree == snapshot
            _, fresh_proc = open_inodes_and_paths([directory])
            assert not fresh_proc['path_hits']
            action = {'kind': 'remove_idle_regenerable_cache', 'path': str(directory),
                      'class': item['class'], 'snapshot': snapshot,
                      'freed_allocated_bytes': snapshot['allocated_bytes']}
            journal({'state': 'intent', **action})
            shutil.rmtree(directory)
            assert not directory.exists()
            result['deleted_caches'].append(action)
            result['cache_allocated_bytes_released'] += snapshot['allocated_bytes']
            journal({'state': 'applied_verified_absent', **action})
            save_result(result)

        # Verify all retained alias identities and all retained contents after cleanup.
        for group, values in chosen:
            canonical = values[0][0]
            assert digest(canonical) == group['sha256']
            inode = current(canonical).st_ino
            assert all(current(path).st_ino == inode for path, _ in values)
        for item, _, endpoints in frames:
            directory = Path(item['frames_directory'])
            assert {str(p) for p in directory.iterdir()} == {row['path'] for row in endpoints}
            for row in endpoints + [item['capture'], item['retain_video_verification']['video']]:
                assert digest(Path(row['path'])) == row['sha256']
        result['status'] = 'complete_verified'
    except BaseException as error:
        result['status'] = 'attention_required_no_automatic_retry'
        result['error'] = repr(error)
        raise
    finally:
        result['finished_at_utc'] = now()
        result['elapsed_s'] = time.monotonic() - started
        result['allocated_bytes_released'] = sum(result[key] for key in
            ('dedup_allocated_bytes_released', 'png_allocated_bytes_released', 'cache_allocated_bytes_released'))
        result['free_bytes_after'] = free_bytes()
        result['filesystem_free_delta_bytes'] = result['free_bytes_after'] - result['free_bytes_before']
        result['filesystem_delta_is_not_exclusive_causal_measurement'] = True
        result['journal_sha256'] = digest(JOURNAL) if JOURNAL.exists() else None
        save_result(result)
    print(json.dumps({key: result[key] for key in
                      ('status', 'allocated_bytes_released', 'dedup_allocated_bytes_released',
                       'png_allocated_bytes_released', 'cache_allocated_bytes_released',
                       'elapsed_s', 'filesystem_free_delta_bytes')}, ensure_ascii=False))
    print(json.dumps({'receipt': str(RESULT), 'receipt_sha256': digest(RESULT),
                      'linked_alias_count': len(result['linked_aliases']),
                      'deleted_middle_png_count': len(result['deleted_middle_pngs']),
                      'deleted_cache_count': len(result['deleted_caches']),
                      'skipped_group_count': len(result['skipped'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
