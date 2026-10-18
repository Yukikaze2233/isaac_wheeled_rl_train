"""Shared CPU fixtures for current runtime tests."""
from datetime import datetime, timezone, timedelta
import json
from wheeled_tasks.chassis.training_budget import BudgetOwner, BudgetError, BudgetLimit, PAID_CAP, REVIEW_SCHEMA, digest_file
NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)

class TestMeter:
    __test__ = False
    owner_id = 'explicit_cpu_fixture_meter'

    def __init__(self, value=0.0):
        self.value = value

    def read_total_gpu_seconds(self):
        return self.value

def make_review(tmp_path, *, pilot_cap=3, allocation=10, roles=None):
    roles = roles or {'a': 'pilot', 'b': 'diagnostic', 'main': 'main'}
    identity = {'contract_sha256': 'a' * 64, 'training_source_sha256': 'b' * 64, 'seed': 1011, 'stage': 'foundation', 'requested_updates': allocation, 'parent_checkpoint_sha256': None}
    review = {'schema': REVIEW_SCHEMA, 'campaign_id': 'fixture_campaign', 'ledger_id': 'fixture_ledger', 'ledger_path': str(tmp_path / 'paid.json'), 'paid_update_cap': PAID_CAP, 'pilot_paid_update_cap': pilot_cap, 'absolute_deadline_utc': '2026-10-11T01:00:00+00:00', 'max_attempt_wall_seconds': 300.0, 'gpu_occupied_seconds_cap': 100.0, 'gpu_meter_owner': {'kind': 'test_only_cpu_meter_v1', 'id': TestMeter.owner_id}, 'review_reference': 'CPU test data, NOT a human authorization or production cap', 'attempts': {key: {**identity, 'role': role} for key, role in roles.items()}}
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(review))
    return (path, review)

def publish(owner, path):
    metadata = owner.prepare_checkpoint(path)
    path.write_bytes(b'CPU checkpoint placeholder; never load as torch')
    path.with_suffix('.checkpoint.json').write_text(json.dumps({'checkpoint_sha256': digest_file(path), 'contract_sha256': owner.identity['contract_sha256'], 'training_budget': metadata}))
    owner.register_checkpoint(path, metadata)
    return metadata
