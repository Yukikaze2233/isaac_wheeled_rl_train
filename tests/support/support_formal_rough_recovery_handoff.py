"""Shared CPU fixtures for current runtime tests."""
import json
from tests.support.support_flat_scripted_recovery import ROOT, config

def contracts():
    folder = ROOT / 'configs/v7_formal_20261012'
    old = json.loads((folder / 'rough_candidate.json').read_bytes())
    new = config()
    new.pop('flat_special_mode_iteration_cap')
    new.pop('evaluation')
    new['formal_stage_policy'] = json.loads((folder / 'flat_special_candidate.json').read_bytes())['formal_stage_policy']
    new['formal_stage_transition'] = {'from': 'terrain', 'to': 'foundation', 'reset_stage_actor_clock': True}
    return (old, new)
