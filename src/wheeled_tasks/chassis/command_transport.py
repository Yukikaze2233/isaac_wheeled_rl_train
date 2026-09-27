"""Ordered USB downlink with command loss and periodic cached CAN output.

The host computes torque before transmission. No feedback delay or motor PD is
invented here. USB arrivals are continuous-time; CAN consumes the newest arrived
batch on the control tick. A lost batch leaves the previous CAN command intact.
"""
import math

import torch


class UsbCommandTransport:
    def __init__(self, count, device, dt, config, seed, profiles=None, domain_draw=None):
        self.cfg, self.dt, self.tick = dict(config), float(dt), 0
        self.transport_dt = float(config["can_period_ms"]) * .001
        self.device = device
        self.low, self.high = [x * .001 for x in config["delay_ms"]]
        self.fraction = float(config["enabled_fraction"])
        self.loss_max = float(config["drop_probability_max"])
        self.burst_probability = float(config["burst_probability"])
        self.burst_low, self.burst_high = config["burst_packets"]
        if (not all(math.isfinite(x) for x in (self.low, self.high, self.dt, self.transport_dt, self.fraction,
                                              self.loss_max, self.burst_probability))
                or not 0 <= self.low <= self.high or self.dt <= 0
                or not 0 <= self.fraction <= 1 or not 0 <= self.loss_max <= 1
                or not 0 <= self.burst_probability <= 1
                or not 1 <= self.burst_low <= self.burst_high
                or self.transport_dt <= 0 or self.dt < self.transport_dt
                or abs(self.dt / self.transport_dt - round(self.dt / self.transport_dt)) > 1e-8):
            raise ValueError("Invalid USB downlink/CAN timing contract")
        self.substeps = round(self.dt / self.transport_dt)
        self.generator = torch.Generator(device=device).manual_seed(seed ^ 0x43414E)
        self.rows = torch.arange(count, device=device)
        self.slots = math.ceil(self.high / self.transport_dt) + 3
        self.pending = torch.zeros(self.slots, count, 6, device=device)
        self.valid = torch.zeros(self.slots, count, dtype=torch.bool, device=device)
        self.generated = torch.zeros(self.slots, count, device=device)
        self.held = torch.zeros(count, 6, device=device)
        self.held_generated = torch.zeros(count, device=device)
        self.last_arrival = torch.zeros(count, device=device)
        self.delay = torch.zeros(count, device=device)
        self.base_delay = torch.zeros(count, device=device)
        self.loss_probability = torch.zeros(count, device=device)
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.burst_remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.sent = torch.zeros(count, device=device)
        self.lost = torch.zeros(count, device=device)
        self.age = torch.zeros(count, device=device)
        self.profiles = profiles
        self.domain_draw = domain_draw
        if profiles is not None and len(profiles) != count:
            raise ValueError("One communication profile is required per environment")
        self.fixed_burst_every = torch.zeros(count, dtype=torch.long, device=device)
        self.fixed_burst_length = torch.zeros_like(self.fixed_burst_every)
        if profiles is not None:
            self.profile_enabled = torch.tensor([p is not None for p in profiles], device=device)
            self.profile_delay = torch.tensor([self.low if p is None else float(p["delay_ms"]) * .001
                                               for p in profiles], device=device)
            self.profile_loss = torch.tensor([0. if p is None else float(p.get("drop_probability", 0.))
                                              for p in profiles], device=device)
            self.fixed_burst_every = torch.tensor([0 if p is None else int(p.get("burst_every_packets", 0))
                                                   for p in profiles], device=device)
            self.fixed_burst_length = torch.tensor([0 if p is None else int(p.get("burst_packets", 0))
                                                    for p in profiles], device=device)
            if not bool(((self.profile_delay >= self.low) & (self.profile_delay <= self.high)
                         & (self.profile_loss >= 0) & (self.profile_loss <= 1)
                         & torch.isfinite(self.profile_delay) & torch.isfinite(self.profile_loss)
                         & (self.fixed_burst_every >= self.fixed_burst_length)
                         & (self.fixed_burst_length >= 0)).all()):
                raise ValueError("Communication evaluation profile outside declared domain")
        self.reset(self.rows)

    def reset(self, ids):
        count = len(ids)
        self.valid[:, ids] = False
        self.held[ids] = 0.
        self.held_generated[ids] = self.tick * self.transport_dt
        self.last_arrival[ids] = self.tick * self.transport_dt
        self.burst_remaining[ids] = 0
        self.sent[ids] = self.lost[ids] = self.age[ids] = 0.
        self.enabled[ids] = torch.rand(count, generator=self.generator, device=self.device) < self.fraction
        if self.domain_draw is not None:
            self.enabled[ids] = self.domain_draw[ids] < self.fraction
        self.base_delay[ids] = self.low + (self.high - self.low) * torch.rand(
            count, generator=self.generator, device=self.device)
        self.loss_probability[ids] = self.loss_max * torch.rand(count, generator=self.generator, device=self.device)
        if self.profiles is not None:
            # Fixed evaluation is deterministic, independent of training mixture.
            self.enabled[ids] = self.profile_enabled[ids]
            self.base_delay[ids] = self.profile_delay[ids]
            self.loss_probability[ids] = self.profile_loss[ids]
        self.delay[ids] = self.base_delay[ids]

    def apply_torque(self, command):
        """Preserve the transport impulse across a coarser physics interval.

        The host PD command is held over this interval. Repeated 1ms transport
        ticks do not pretend to have new physical feedback between solver steps.
        """
        if self.substeps == 1:
            return self._transport_tick(command)
        impulse = torch.zeros_like(command)
        for _ in range(self.substeps):
            impulse += self._transport_tick(command)
        return impulse / self.substeps

    def _transport_tick(self, command):
        now = self.tick * self.transport_dt
        slot = self.tick % self.slots
        arrived = self.valid[slot]
        self.held = torch.where(arrived[:, None], self.pending[slot], self.held)
        self.held_generated = torch.where(arrived, self.generated[slot], self.held_generated)
        self.valid[slot] = False
        # Correlated jitter preserves the stated USB bound. FIFO ordering avoids
        # creating UDP-like command reordering on an ordered USB bulk stream.
        jitter = (torch.rand(len(command), device=self.device, generator=self.generator) - .5) * .001
        if self.profiles is None:
            self.delay = (.9 * self.delay + .1 * self.base_delay + jitter).clamp(self.low, self.high)
        arrival = torch.maximum(now + self.delay, self.last_arrival)
        due = torch.ceil(arrival / self.transport_dt - 1e-5).long().clamp_min(self.tick + 1)
        start = torch.rand(len(command), device=self.device, generator=self.generator) < self.burst_probability
        if self.profiles is not None:
            start = (self.fixed_burst_every > 0) & (self.tick > 0) & (self.tick % self.fixed_burst_every.clamp_min(1) == 0)
        length = torch.randint(self.burst_low, self.burst_high + 1, (len(command),),
                               device=self.device, generator=self.generator)
        if self.profiles is not None:
            length = self.fixed_burst_length
        self.burst_remaining = torch.where(start & (self.burst_remaining == 0), length, self.burst_remaining)
        dropped = ((torch.rand(len(command), device=self.device, generator=self.generator) < self.loss_probability)
                   | (self.burst_remaining > 0)) & self.enabled
        self.burst_remaining = (self.burst_remaining - 1).clamp_min(0)
        send = self.enabled & ~dropped
        destination = due % self.slots
        self.pending[destination, self.rows] = torch.where(send[:, None], command, self.pending[destination, self.rows])
        self.generated[destination, self.rows] = torch.where(send, now, self.generated[destination, self.rows])
        self.valid[destination, self.rows] |= send
        self.last_arrival = torch.where(send, arrival, self.last_arrival)
        self.sent += self.enabled
        self.lost += dropped
        self.age = torch.where(self.enabled, now - self.held_generated, 0.)
        self.tick += 1
        return torch.where(self.enabled[:, None], self.held, command)

    def metrics(self):
        return {"/transport/downlink_enabled_fraction": self.enabled.float().mean(),
                "/transport/usb_downlink_delay_ms": (self.delay * 1000).mean(),
                "/transport/executed_command_age_ms": (self.age * 1000).mean(),
                "/transport/executed_command_age_max_ms": (self.age * 1000).max(),
                "/transport/command_loss_fraction": self.lost.sum() / self.sent.sum().clamp_min(1)}
