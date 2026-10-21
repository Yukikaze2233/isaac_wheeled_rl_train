"""Read recorded source layouts without changing their content hashes."""
from wheeled_tasks.chassis.source_layout import MODULES


FLAT_ENVIRONMENT = 'src/wheeled_tasks/chassis/env.py'
PACKAGED_ENVIRONMENT = 'src/wheeled_tasks/chassis/application/environment.py'
ENTRY_PATHS = {
    'application.training.train': 'scripts/train_chassis.py',
    'application.evaluation.evaluate_policy': 'scripts/evaluate_flat_policy.py',
    'application.export': 'scripts/export_chassis_candidate.py',
    'application.evaluation.replay': 'scripts/replay_v7_ordinary.py',
    'application.runtime.checkpoints': 'scripts/chassis_checkpoints.py',
    'application.runtime.batch_export': 'scripts/chassis_batch_export.py',
    'application.runtime.remote_job': 'scripts/chassis_remote_job.py',
    'application.console': 'tools/v7ctl.py',
}


def recorded_source_requirements(current_requirements, recorded_hashes):
    """Choose required file names for a seal's own layout, never new hashes.

    A historical checkpoint is authenticated against its original dictionary.
    This function does not declare old code equivalent to current code and
    cannot authorize evaluation or export against a different implementation.
    """
    if FLAT_ENVIRONMENT not in recorded_hashes:
        return set(current_requirements)
    if PACKAGED_ENVIRONMENT in recorded_hashes:
        raise ValueError('Checkpoint mixes flat and packaged environment source layouts')
    names = {}
    for name, module in MODULES.items():
        current = 'src/wheeled_tasks/chassis/' + module.replace('.', '/') + '.py'
        names[current] = ENTRY_PATHS.get(module, 'src/wheeled_tasks/chassis/' + name + '.py')
    return {names.get(path, path) for path in current_requirements}
