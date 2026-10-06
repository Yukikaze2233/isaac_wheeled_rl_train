"""CPU parser/admission fixtures; no actual running capture opened."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('SIDE_outcome_test', HERE/'compare_SIDE_outcome_001.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def birth_rows():
    return [dict(lane=lane, birth_id=1, lesson='prefix_stand', partition='right_censored_unfinished',
        first_handover_reference_tick=-1, first_motion_release_reference_tick=-1) for lane in range(4096)]


def test_fixed_denominator_censor_and_ever_flags():
    rows = birth_rows()
    rows[0].update(partition='success', first_handover_reference_tick=7, first_motion_release_reference_tick=40)
    value = m.first_birth_summary(rows)
    assert value['fixed_birth_denominator'] == 4096
    assert value['partitions'] == {'success': 1, 'right_censored_unfinished': 4095}
    assert value['by_lesson']['prefix_stand']['ever_handed_over'] == 1
    assert value['by_lesson']['prefix_stand']['ever_motion_released'] == 1


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'second_birth'])
def test_invalid_native_denominator_rejected(change):
    rows = birth_rows()
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows[-1]['lane'] = 0
    if change == 'second_birth': rows[-1]['birth_id'] = 2
    with pytest.raises(ValueError, match='denominator'): m.first_birth_summary(rows)


def test_native_exit_is_required_before_opening_review_or_gzip(tmp_path):
    path = tmp_path/'process.json'
    path.write_text(json.dumps(dict(process_exit_observed=False, process_exit_code=None, process_present_after_wait=True)))
    with pytest.raises(ValueError, match='has not completed'): m.completed_admission(tmp_path, path, 'absent')
    assert not (tmp_path/'side_speed_review.json').exists()


def test_actual_body_scalar_force_scope():
    frame = dict(actual_terrain_contact_readback=dict(sensor_body_rows_with_observed_force_or_patch=[
        dict(body_name='RR_link1', normal_patches=[dict(force_scalar_n=[[10.], [20.]]), dict(force_scalar_n=[[2.]])]),
        dict(body_name='base_link', normal_patches=[dict(force_scalar_n=[[900.]])])]))
    assert m.body_normal_force(frame, 'RR_link1') == 32.
    assert m.body_normal_force(dict(actual_terrain_contact_readback=None), 'RR_link1') == 0.
