"""Domain validation shared by reward selection and physical reward adapters."""
from collections.abc import Mapping
from copy import deepcopy
import math

def validate_command_decoupling(config):
    """Validate the explicit V7 pure-translation yaw-cost candidate."""
    if config is None:
        return None
    if (not isinstance(config, Mapping)
            or config.get("version") != "v7-pure-translation-yaw-cost-v2"
            or type(config.get("enabled")) is not bool):
        raise ValueError("V7 command decoupling requires its version and explicit boolean enabled")
    fields = {"version", "enabled"}
    if config["enabled"]:
        fields.add("coefficient_per_rad2_s2")
        coefficient = config.get("coefficient_per_rad2_s2")
        if (type(coefficient) not in (int, float)
                or not math.isfinite(coefficient) or coefficient <= 0):
            raise ValueError("V7 command decoupling requires a positive finite coefficient_per_rad2_s2")
    if set(config) != fields:
        raise ValueError("V7 command decoupling contains missing or unsupported fields")
    return deepcopy(dict(config))


V7_FINAL_UNDESIRED_BODY_NAMES = (
    "base_link", "L_link1", "L_link2", "R_link1", "R_link2",
    "LL_link1", "LL_link2", "LL_link3", "LL_link4",
    "RR_link1", "RR_link2", "RR_link3", "RR_link4",
)
