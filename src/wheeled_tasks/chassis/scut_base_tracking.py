"""Reuse the native v14 quiet-speed and posture costs across training domains."""
from copy import deepcopy

from .locomotion_quality_revision import POSTURE, STATIONARY


def apply_training_tracking(config):
    """Keep fixed-case contracts intact; no new phase-dependent reward path."""
    if config.get("evaluation_exact_cases"):
        return
    dense = config["dense_tracking"]
    if dense.get("enabled") is not True:
        raise ValueError("SCUT-style training requires the existing dense tracking assembly")
    dense["stand_still_lin_vel"] = deepcopy(STATIONARY)
    dense["locomotion_posture"] = deepcopy(POSTURE)


def update_runtime_expectation(config, runtime):
    """Describe independently materialized native modules, including eval roles."""
    dense = config["dense_tracking"]
    quality = {key: deepcopy(dense[key]) for key in
               ("stand_still_lin_vel", "locomotion_posture") if key in dense}
    if quality:
        runtime["modules"].update(stand_still_lin_vel="stand_still_lin_vel" in quality,
            locomotion_posture_floor="locomotion_posture" in quality)
        runtime["locomotion_quality_rewards"] = quality
