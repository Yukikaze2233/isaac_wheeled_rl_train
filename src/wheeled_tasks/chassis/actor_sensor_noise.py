"""Policy-only physical-unit sensor noise, applied before observation encoding."""
from copy import deepcopy

KEY = "actor_sensor_noise"
SETTINGS = {
    "version": "actor-physical-uniform-v1", "enabled": True,
    "clock": "policy_steps", "application": "after_sensor_delay_before_encoding",
    "gyro_rad_s": .25, "projected_gravity": .05, "leg_position_rad": .025,
    "leg_velocity_rad_s": .5, "wheel_velocity_rad_s": 1.,
    "wheel_position_observation": "zero",
}


def validate_actor_sensor_noise(contract):
    """Select one noise producer; disabled legacy configuration stays explicit."""
    if KEY not in contract:
        return None
    value = contract[KEY]
    if (not isinstance(value, dict) or set(value) != set(SETTINGS)
            or any(value[key] != expected or (type(value[key]) is bool) != (type(expected) is bool)
                   for key, expected in SETTINGS.items())):
        raise ValueError("Actor sensor noise requires the complete physical uniform settings")
    if (contract.get("observation_noise_enabled") is not True
            or contract.get("actor_observation_source") != "scut35_encoders_imu_commands"
            or contract.get("actor_dim") != 35 or contract.get("actor_frame_dim") != 35
            or contract.get("critic_dim") != 78 or contract.get("history_length") != 1):
        raise ValueError("Physical sensor noise requires the declared current-frame 35D/78D interface")
    legacy = contract.get("signal_perturbations", {})
    if legacy.get("enabled") is not False or type(legacy.get("max_delay_steps")) is not int or legacy["max_delay_steps"] != 0:
        raise ValueError("Physical sensor noise must replace legacy policy noise and lag")
    return deepcopy(value)


class ActorSensorNoise:
    """Cache one raw sensor-noise draw per policy tick and reset environment.

    Sensor order is six control-axis positions, six velocities, gyro xyz and
    projected gravity xyz. PC feedback, commands, action history and critic
    are separate consumers; no caller-owned sensor tensor is mutated.
    """

    def __init__(self, count, device, contract, generator):
        import torch

        self.cfg = validate_actor_sensor_noise(contract)
        if self.cfg is None:
            raise ValueError("Physical sensor noise must be explicitly selected")
        if type(count) is not int or count <= 0:
            raise ValueError("Physical sensor noise requires a positive environment count")
        self.generator = generator
        self.noise = torch.zeros(count, 18, dtype=torch.float32, device=device)
        if not isinstance(generator, torch.Generator) or generator.device != self.noise.device:
            raise ValueError("Physical sensor noise must share the environment RNG device")
        self.input = torch.zeros_like(self.noise)
        self.output = torch.zeros_like(self.noise)
        c = self.cfg
        self.half_width = self.noise.new_tensor([
            c["leg_position_rad"], c["leg_position_rad"], 0.,
            c["leg_position_rad"], c["leg_position_rad"], 0.,
            c["leg_velocity_rad_s"], c["leg_velocity_rad_s"], c["wheel_velocity_rad_s"],
            c["leg_velocity_rad_s"], c["leg_velocity_rad_s"], c["wheel_velocity_rad_s"],
            *([c["gyro_rad_s"]] * 3), *([c["projected_gravity"]] * 3)])
        self.policy_tick = None
        self.samples = 0
        self._reset_ids = []

    def reset(self, ids):
        if len(ids):
            self._reset_ids.append(ids.clone())

    def apply(self, sensors, policy_tick):
        import torch

        if (sensors.shape != self.noise.shape or sensors.dtype != torch.float32
                or sensors.device != self.noise.device or type(policy_tick) is not int or policy_tick < 0):
            raise ValueError("Actor sensor noise requires float32 [environment,18] and an integer policy tick")
        if self.policy_tick != policy_tick:
            self.noise.uniform_(-1., 1., generator=self.generator).mul_(self.half_width)
            self.samples += len(self.noise)
            self._reset_ids.clear()
            self.policy_tick = policy_tick
        elif self._reset_ids:
            ids = torch.unique(torch.cat(self._reset_ids))
            self.noise[ids] = (2 * torch.rand(len(ids), 18, device=sensors.device,
                generator=self.generator) - 1) * self.half_width
            self.samples += len(ids)
            self._reset_ids.clear()
        self.input.copy_(sensors)
        torch.add(sensors, self.noise, out=self.output)
        return self.output

    def metrics(self):
        result = {"/sensor_noise/noise_vector_draws": self.samples,
                  "/sensor_noise/enabled_fraction": 1.}
        for name, columns in (("leg_position_rad", [0, 1, 3, 4]),
                              ("leg_velocity_rad_s", [6, 7, 9, 10]),
                              ("wheel_velocity_rad_s", [8, 11]),
                              ("gyro_rad_s", [12, 13, 14]),
                              ("projected_gravity", [15, 16, 17])):
            result[f"/sensor_noise/{name}_absmax"] = self.noise[:, columns].abs().amax()
        return result

    def readback(self):
        return {"settings": deepcopy(self.cfg), "policy_tick": self.policy_tick,
                "noise_vector_draws": self.samples,
                "sensor_input18": self.input.clone(), "noise18": self.noise.clone(),
                "sensor_output18": self.output.clone(),
                "scope": "policy sensor path only; control-axis order; physical units before encoding"}
