"""Failure injection only for the new mechanical-vs-whole admission boundary."""
import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mechanical_admission_v2',HERE/'read_remaining_capture_002.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def fixture():
    return (dict(schema='source021-remaining-mechanical-analysis-admission-v1',
        status='completed_native_mechanical_data_admitted_for_descriptive_readonly_analysis',
        strict_whole_reproduction=False,payment_aggregate_float_bits_exact=False,qualification=None,PPO_updates=0,
        formal_updates_consumed=0,actual_policy_steps=600,actual_physics_steps=2400,selected_lanes=[881,1877,2121],
        selected_lanes_5ms_complete=True,full4096_firstbirth_events_global_record_states_float_bits_exact=True,
        identity_teaching_override_float_bits_exact=True,original_control_source_RNG_birth_clocks_report_values_exact=True),
        dict(process_exit_observed=True,process_exit_code=0,process_present_after_wait=False))

def test_mechanical_admission_preserves_rejected_whole_scope():
    admission,process=fixture();r.require_admission_flags(admission,process)
    admission['strict_whole_reproduction']=True
    with pytest.raises(ValueError,match='misrepresented'):r.require_admission_flags(admission,process)

def test_not_complete_or_current_process_cannot_parse_actual():
    admission,process=fixture();process['process_present_after_wait']=True
    with pytest.raises(ValueError,match='misrepresented'):r.require_admission_flags(admission,process)
    process['process_present_after_wait']=False;admission['selected_lanes_5ms_complete']=False
    with pytest.raises(ValueError,match='misrepresented'):r.require_admission_flags(admission,process)

def test_false_claim_of_payment_exact_or_level_subset_not_accepted():
    admission,process=fixture();admission['payment_aggregate_float_bits_exact']=True
    with pytest.raises(ValueError,match='misrepresented'):r.require_admission_flags(admission,process)
    admission['payment_aggregate_float_bits_exact']=False;admission['selected_lanes']=[881,1877]
    with pytest.raises(ValueError,match='misrepresented'):r.require_admission_flags(admission,process)
