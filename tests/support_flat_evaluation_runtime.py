"""Shared CPU fixtures for current runtime tests."""
from functools import lru_cache
from wheeled_tasks.chassis.flat_evaluation_runtime import BoundEvaluationInputs, BoundFile, EvaluationActor, PreResetLedgerAdapter, _stock_actor, canonical, rng_digest, runtime_source_files, sha, strict_json, validate_overlay

@lru_cache(maxsize=1)
def stock_state():
    return _stock_actor(None).state_dict()
