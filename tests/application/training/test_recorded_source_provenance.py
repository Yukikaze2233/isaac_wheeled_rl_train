"""Historical checkpoint records retain their own paths and source hashes."""
from copy import deepcopy
import hashlib
import json

import pytest

from wheeled_tasks.chassis.application.compatibility.source_provenance import (
    FLAT_ENVIRONMENT, PACKAGED_ENVIRONMENT, recorded_source_requirements,
)


def test_flat_layout_is_authenticated_without_relabeling_its_hash_dictionary():
    original_pd = 'src/wheeled_tasks/chassis/v5_control.py'
    packaged_pd = 'src/wheeled_tasks/chassis/control/pd.py'
    sources = {FLAT_ENVIRONMENT: 'a' * 64, original_pd: 'b' * 64}
    before = deepcopy(sources)
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    assert recorded_source_requirements({PACKAGED_ENVIRONMENT, packaged_pd}, sources) == {
        FLAT_ENVIRONMENT, original_pd,
    }
    assert sources == before
    assert hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest() == digest


def test_packaged_layout_keeps_current_requirements():
    sources = {PACKAGED_ENVIRONMENT: 'a' * 64}
    assert recorded_source_requirements({PACKAGED_ENVIRONMENT}, sources) == {PACKAGED_ENVIRONMENT}


def test_mixed_environment_layout_is_rejected():
    with pytest.raises(ValueError, match='mixes'):
        recorded_source_requirements({PACKAGED_ENVIRONMENT}, {
            FLAT_ENVIRONMENT: 'a' * 64, PACKAGED_ENVIRONMENT: 'b' * 64,
        })
