"""Observe native dense rewards and PPO batches without modifying either."""
import torch

from .diagnostic_curriculum import PolicyUpdateProbe


class LearningAudit:
    def __init__(self, env, algorithm):
        self.env, self.algorithm = env, algorithm
        self.probe = PolicyUpdateProbe(algorithm.actor, algorithm.storage, env.skills.batches)
        self.hold = torch.zeros(env.num_envs, device=env.device)
        self.previous_height = env.commands[:, 2].clone()
        self.rollout_height = self.previous_height.clone()
        self.stable_height = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.sums = None
        self.steps = 0
        self.pending = {}

    def observe(self, data, components):
        if self.steps == 0:
            self.rollout_height.copy_(data["commands"][:, 2])
            self.stable_height.fill_(True)
        self.stable_height &= (data["commands"][:, 2] - self.rollout_height).abs() < 1e-6
        quiet = (data["commands"][:, :2].abs() < .001).all(-1)
        unchanged = (data["commands"][:, 2] - self.previous_height).abs() < 1e-6
        self.hold.copy_(torch.where(quiet & unchanged, self.hold + self.env.policy_dt, 0.))
        self.previous_height.copy_(data["commands"][:, 2])
        self.last_commands = data["commands"]
        values = {"reward_density/" + k: v for k, v in components.items()}
        values.update(total_reward_density=data["reward"] / self.env.policy_dt,
            planar_speed_m_s=data["planar_speed_world"],
            height_error_m=(data["height"] - data["commands"][:, 2]).abs(),
            quiet_nonfall=((data["commands"][:, :2].abs() < .001).all(-1)
                             & (data["reasons"]["fall"] == 0)).float(),
            continuous_zero_10s=(self.hold >= 10.).float(),
            continuous_zero_15s=(self.hold >= 15.).float(),
            done=data["done"].float())
        if self.sums is None:
            self.names = tuple(values)
            # observe runs under rollout inference mode, drain under optimization.
            with torch.inference_mode(False):
                self.sums = torch.zeros(self.env.num_envs, len(values), device=self.env.device)
        if tuple(values) != self.names:
            raise ValueError("Reward audit component keys changed within a rollout")
        self.sums += torch.stack(list(values.values()), -1)
        self.steps += 1
        self.hold.masked_fill_(data["done"].bool(), 0.)

    def groups(self):
        for ids, spec in self.env.skills.batches:
            yield spec["group_name"], ids
            if spec["kind"] == "stand":
                for h in self.env.cfg["new_asset_curriculum"]["height_sampling_schedule"]["fixed_heights_m"]:
                    selected = ids[((self.last_commands[ids, 2] - h).abs() < 1e-5) & self.stable_height[ids]]
                    if len(selected):
                        yield f"{spec['group_name']}/height_{round(h * 1000)}mm", selected

    def capture(self):
        self.probe.capture()
        storage = self.algorithm.storage
        for name, ids in self.groups():
            prefix = "LearningAudit/" + name + "/"
            advantage = storage.advantages[:, ids]
            residual = storage.returns[:, ids] - storage.values[:, ids]
            self.pending.update({prefix + "advantage_mean": advantage.mean(),
                prefix + "advantage_positive_fraction": (advantage > 0).float().mean(),
                prefix + "return_target_residual_mean": residual.mean(),
                prefix + "rollout_samples": advantage.new_tensor(advantage.numel())})

    def drain(self):
        result = {**self.pending, **self.probe.metrics()}
        for name, ids in self.groups():
            mean = self.sums[ids].mean(0) / max(1, self.steps)
            result.update({f"LearningAudit/{name}/{key}": value for key, value in zip(self.names, mean)})
        keys = tuple(result)
        values = torch.stack([result[k].detach() for k in keys]).cpu().tolist()
        self.sums.zero_()
        self.steps = 0
        self.pending.clear()
        return dict(zip(keys, values))
