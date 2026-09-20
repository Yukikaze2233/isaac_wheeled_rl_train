"""Task-specific completion and dense jump references for the full V5 curriculum."""
import torch

from .task import Phase


class FullTaskSemantics:
    def __init__(self, count, device, dt, config):
        self.cfg, self.dt = config, dt
        self.origin_xy = torch.zeros(count, 2, device=device)
        self.clearance_peak = torch.zeros(count, device=device)
        self.height_peak = torch.zeros(count, device=device)
        self.clear_air_time = torch.zeros(count, device=device)
        self.clear_air_time_peak = torch.zeros(count, device=device)
        self.previous_x = torch.zeros(count, device=device)
        self.release_velocity = torch.zeros(count, device=device)
        self.release_recorded = torch.zeros(count, dtype=torch.bool, device=device)
        self.com_release_height = torch.zeros(count, device=device)
        self.com_release_speed = torch.zeros(count, device=device)
        self.com_rise = torch.zeros(count, device=device)
        self.com_released = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids, position):
        self.origin_xy[ids] = position[ids, :2]
        self.previous_x[ids] = position[ids, 0]
        self.release_velocity[ids] = 0.
        self.release_recorded[ids] = False
        self.com_released[ids] = False
        self.com_rise[ids] = 0.
        self.com_release_speed[ids] = 0.
        for value in (self.clearance_peak, self.height_peak, self.clear_air_time, self.clear_air_time_peak):
            value[ids] = 0.

    def jump_reference(self, command_height, phase, phase_time, air_time, contact_time):
        preload_time = self.cfg.get("preload_seconds", .25)
        preload_depth = self.cfg.get("preload_depth_m", .02)
        release_offset = self.cfg.get("release_height_offset_m", .035)
        apex = self.cfg.get("jump_apex_delta_m", .06)
        if bool((torch.as_tensor(apex) <= torch.as_tensor(release_offset)).any()):
            raise ValueError("Jump apex must exceed the release height")
        release_speed = (2 * 9.81 * (apex - release_offset)) ** .5
        push_time = 2 * (preload_depth + release_offset) / release_speed
        u = (phase_time / preload_time).clamp(0., 1.)
        h_pre = command_height - preload_depth * (3 * u.square() - 2 * u.pow(3))
        v_pre = -preload_depth * 6 * u * (1 - u) / preload_time
        t_push = torch.minimum((phase_time - preload_time).clamp_min(0.),
                               torch.as_tensor(push_time, device=phase_time.device))
        acceleration = release_speed / push_time
        h_push = command_height - preload_depth + .5 * acceleration * t_push.square()
        v_push = acceleration * t_push
        h_takeoff = torch.where(phase_time < preload_time, h_pre, h_push)
        v_takeoff = torch.where(phase_time < preload_time, v_pre, v_push)
        h_flight = command_height + release_offset + release_speed * air_time - .5 * 9.81 * air_time.square()
        v_flight = release_speed - 9.81 * air_time
        decay = torch.exp(-contact_time / .15)
        h_land, v_land = command_height - .015 * decay, .1 * decay
        takeoff, flight = phase == Phase.TAKEOFF, phase == Phase.FLIGHT
        landing = (phase == Phase.LANDING) | (phase == Phase.RECOVERY)
        height = torch.where(takeoff, h_takeoff, torch.where(flight, h_flight, torch.where(landing, h_land, command_height)))
        speed = torch.where(takeoff, v_takeoff, torch.where(flight, v_flight, torch.where(landing, v_land, 0.)))
        return height, speed

    def observe(self, mode, height, wheel_clearance, phase, vertical_velocity):
        jumping = mode == 4
        minimum = wheel_clearance.amin(-1)
        self.clearance_peak = torch.maximum(self.clearance_peak, minimum * jumping)
        self.height_peak = torch.maximum(self.height_peak, height * jumping)
        clear = jumping & (minimum >= self.cfg.get("jump_min_clearance_m", .01))
        self.clear_air_time = torch.where(clear, self.clear_air_time + self.dt, 0.)
        self.clear_air_time_peak = torch.maximum(self.clear_air_time_peak, self.clear_air_time)
        entered_air = jumping & (phase == Phase.FLIGHT) & ~self.release_recorded
        self.release_velocity[entered_air] = vertical_velocity[entered_air]
        self.release_recorded |= entered_air

    def completion(self, mode, position, wheel_x, route_goal, contacts, stable, phase, command_height, planar_speed=None):
        route = ((mode == 2) | (mode == 3)) & (wheel_x.amin(-1) >= route_goal)
        displacement = (position[:, :2] - self.origin_xy).norm(dim=-1)
        jump = ((mode == 4) & (self.clear_air_time_peak >= self.cfg.get("jump_min_air_seconds", .06))
                & (self.release_velocity >= self.cfg.get("jump_release_velocity_min_m_s", .2))
                & (self.height_peak >= command_height + self.cfg.get("jump_apex_delta_m", .06) - .01)
                & (displacement <= self.cfg.get("jump_landing_radius_m", .25))
                & ((phase == Phase.RECOVERY) | (phase == Phase.GROUND)))
        if planar_speed is not None:
            jump &= planar_speed <= self.cfg.get("jump_landing_speed_max_m_s", .1)
        if "jump_com_rise_m" in self.cfg:
            jump &= (self.com_rise >= self.cfg["jump_com_rise_m"]) & (self.com_release_speed > .1)
        forward = torch.as_tensor(self.cfg.get("jump_forward_distance_m", 0.), device=mode.device)
        jump &= (forward <= 0) | (position[:, 0] - self.origin_xy[:, 0] >= forward)
        return (route | jump) & contacts.all(-1) & stable

    def observe_com(self, mode, phase, height, vz):
        airborne = (mode == 4) & (phase == Phase.FLIGHT)
        released = airborne & ~self.com_released
        self.com_release_height[released] = height[released]
        self.com_release_speed[released] = vz[released]
        self.com_released |= released
        self.com_rise = torch.maximum(self.com_rise, (height - self.com_release_height).clamp_min(0) * airborne)

    def dense_jump_reward(self, mode, phase_tracker, height, vz, command_height, leg_extension):
        h_ref, v_ref = self.jump_reference(command_height, phase_tracker.phase, phase_tracker.time,
                                          phase_tracker.air_time, phase_tracker.contact_time)
        active = (mode == 4) & (phase_tracker.phase != Phase.GROUND)
        height_term = torch.exp(-((height - h_ref) / .035).square())
        speed_term = torch.exp(-((vz - v_ref) / .35).square())
        tuck = (phase_tracker.phase == Phase.FLIGHT) * torch.exp(-((leg_extension - .20) / .05).square())
        return active * (3 * height_term + 3 * speed_term + tuck) * self.dt

    def route_progress_reward(self, mode, position):
        delta = position[:, 0] - self.previous_x
        self.previous_x.copy_(position[:, 0])
        return ((mode == 2) | (mode == 3)) * delta.clamp(-.02, .02)
