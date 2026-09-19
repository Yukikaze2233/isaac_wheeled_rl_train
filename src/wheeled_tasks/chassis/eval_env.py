"""Fixed commands and seeded reset perturbations for first-episode evaluation."""
import torch

from .env import ChassisEnv


class FixedCaseEnv(ChassisEnv):
    def __init__(self, config, *args, **kwargs):
        self.case_commands = {c["name"]: c["command"] for c in config["evaluation"]["cases"]}
        self.eval_seed = config["evaluation"]["seed"]
        self.eval_generator = torch.Generator(device=kwargs["device"]).manual_seed(self.eval_seed)
        super().__init__(config, *args, **kwargs)

    def resample_commands(self, ids, *, reset_height=False):
        self.commands[ids] = self.commands.new_tensor([self.case_commands[self.scene_groups[i]] for i in ids.tolist()])
        self.mode[ids] = (self.commands[ids, :2].abs().amax(-1) > .01).long()
        self.command_clock[ids] = 1e9
        self.height_clock[ids] = 1e9
        self.push_clock[ids] = 1e9
        self.push_enabled[ids] = False

    def reset(self, ids):
        super().reset(ids)
        count = len(ids)
        # At 0.3 m/s, ten seconds from tile center stays within the real floor.
        root = torch.zeros(count, 7, device=self.device)
        root[:, :3] = self.origins[ids]
        root[:, 2] += .324
        samples = torch.rand(count, 3, generator=self.eval_generator, device=self.device)
        roll = (samples[:, 0] * 2 - 1) * .01
        pitch = (samples[:, 1] * 2 - 1) * .02
        sr, cr = torch.sin(roll / 2), torch.cos(roll / 2)
        sp, cp = torch.sin(pitch / 2), torch.cos(pitch / 2)
        root[:, 3:] = torch.stack((sr * cp, cr * sp, -sr * sp, cr * cp), -1)
        root[:, 2] += samples[:, 2] * .002
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        self.robot.update(self.dt)

    def reset_suite(self):
        self.eval_generator.manual_seed(self.eval_seed)
        self.reset(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()
