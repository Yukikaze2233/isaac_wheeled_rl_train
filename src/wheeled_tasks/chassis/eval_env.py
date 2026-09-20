"""Fixed commands and seeded reset perturbations for first-episode evaluation."""
import zlib

import torch
import warp as wp

from .env import ChassisEnv


class FixedCaseEnv(ChassisEnv):
    def __init__(self, config, *args, **kwargs):
        self.case_commands = {c["name"]: c["command"] for c in config["evaluation"]["cases"]}
        self.cases = {c["name"]: c for c in config["evaluation"]["cases"]}
        self.eval_seed = config["evaluation"]["seed"]
        self.eval_generator = torch.Generator(device=kwargs["device"]).manual_seed(self.eval_seed)
        self.fixed_reset_samples = None
        super().__init__(config, *args, **kwargs)

    def resample_commands(self, ids, *, reset_height=False):
        previous = self.commands[ids, :2].clone()
        self.commands[ids] = self.commands.new_tensor([self.case_commands[self.scene_groups[i]] for i in ids.tolist()])
        self.mode[ids] = (self.commands[ids, :2].abs().amax(-1) > .01).long()
        for i in ids.tolist():
            case = self.cases[self.scene_groups[i]]
            if case.get("task") == "jump":
                self.mode[i] = 4
            elif case.get("task") == "traverse":
                self.mode[i] = 3 if case.get("terrain") in ("step_down", "stairs_down", "slope_down") else 2
        if self.skills is not None:
            self.skills.sample(ids)
        self.command_clock[ids] = 1e9
        self.height_clock[ids] = 1e9
        self.push_clock[ids] = 1e9
        self.push_enabled[ids] = False
        for i in ids.tolist():
            case = self.cases[self.scene_groups[i]]
            if "push_velocity_m_s" in case:
                self.push_enabled[i] = True
                self.push_clock[i] = case.get("push_at_s", 3.)
        if self.cfg.get("command_slew"):
            self.command_target[ids] = self.commands[ids, :2]
            self.commands[ids, :2] = 0. if reset_height else previous

    def reset(self, ids):
        super().reset(ids)
        count = len(ids)
        # At 0.3 m/s, ten seconds from tile center stays within the real floor.
        root = torch.zeros(count, 7, device=self.device)
        root[:, :3] = self.origins[ids]
        for row, i in enumerate(ids.tolist()):
            if self.kinds[i] not in ("flat", "jump"):
                root[row, 0] -= .8 if self.cases[self.scene_groups[i]].get("task") == "traverse" else 1.5
        all_xy = self.robot.data.root_link_pose_w.torch[:, :2] - self.origins[:, :2]
        all_xy[ids] = root[:, :2] - self.origins[ids, :2]
        root[:, 2] += self.ground_height(all_xy)[ids]
        root[:, 2] += .324
        if self.cfg.get("skill_specs"):
            if self.fixed_reset_samples is None:
                self.fixed_reset_samples = torch.zeros(self.num_envs, 3, device=self.device)
                for name in self.cases:
                    members = [i for i, group in enumerate(self.scene_groups) if group == name]
                    generator = torch.Generator(device=self.device).manual_seed(self.eval_seed + zlib.crc32(name.encode()))
                    values = torch.rand(len(members), 3, generator=generator, device=self.device)
                    replicas = [self.clone_indices[i] % len(members) for i in members]
                    self.fixed_reset_samples[members] = values[replicas]
            samples = self.fixed_reset_samples[ids]
        else:
            samples = torch.rand(count, 3, generator=self.eval_generator, device=self.device)
        roll = (samples[:, 0] * 2 - 1) * .01
        pitch = (samples[:, 1] * 2 - 1) * .02
        sr, cr = torch.sin(roll / 2), torch.cos(roll / 2)
        sp, cp = torch.sin(pitch / 2), torch.cos(pitch / 2)
        root[:, 3:] = torch.stack((sr * cp, cr * sp, -sr * sp, cr * cp), -1)
        root[:, 2] += samples[:, 2] * .002
        reset_velocity = torch.zeros(count, 6, device=self.device)
        if self.skills is not None:
            self.skills.reset_pose(ids, root, reset_velocity)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=reset_velocity, env_ids=ids)
        self.robot.update(self.dt)
        if self.full_tasks is not None:
            self.full_tasks.reset(ids, self.robot.data.root_link_pose_w.torch[:, :3] - self.origins)
        if self.perturbations is not None:
            enabled = torch.tensor([self.cases[self.scene_groups[i]].get("perturbed", False) for i in ids.tolist()], device=self.device)
            self.perturbations.set_enabled(ids, enabled)
            clean = ids[~enabled]
            self.motor_strength[clean] = 1.
            self.spring_strength[clean] = 1.
        self.update_targets()

    def reset_suite(self):
        self.eval_generator.manual_seed(self.eval_seed)
        self.generator.manual_seed(self.eval_seed)
        self.reset(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def apply_push(self, ids):
        velocity = wp.to_torch(self.robot.root_view.get_root_velocities())[ids].clone()
        velocity[:, :2] += velocity.new_tensor([self.cases[self.scene_groups[i]]["push_velocity_m_s"] for i in ids.tolist()])
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=ids, full_data=False)
        self.push_clock[ids] = 1e9
