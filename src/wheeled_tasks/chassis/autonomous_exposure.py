"""GPU reductions for the autonomous curriculum; no per-step CPU synchronization.

This adapter does not assign curricula or inspect privileged state. The caller
provides episode-latched node IDs, the final PPO ownership mask and verified
physical event pulses. Flush once after each optimizer attempt. Persist this
state together with the environment for an exact mid-episode resume.
"""
import torch

from .autonomous_curriculum import validate_definition, _digest, measurement_only


class ExposureAccumulator:
    def __init__(self, definition, num_envs, device="cpu", *, allow_provisional=False):
        validate_definition(definition)
        if (not measurement_only(definition) and definition.get("approval", {}).get("status") != "user_approved"
                and not allow_provisional):
            raise ValueError("Provisional course is a test fixture, not an approved runtime definition")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.definition_sha256 = _digest(definition)
        self.node_ids = [node["id"] for phase in definition["phases"] for node in phase["nodes"]]
        self.node_indices = {nid: i for i, nid in enumerate(self.node_ids)}
        self.node_phase_keys = {node["id"]: list(node["minimum_exposure"]["key_phase_counts"])
                               for phase in definition["phases"] for node in phase["nodes"]}
        self.event_keys = sorted({key for keys in self.node_phase_keys.values() for key in keys})
        self.num_envs, self.device = num_envs, torch.device(device)
        self.episode_has_rl = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode_started = torch.zeros_like(self.episode_has_rl)
        self.episode_nodes = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        # Columns: RL transitions, complete RL episodes, then event pulses.
        self.counts = torch.zeros((len(self.node_ids), 2 + len(self.event_keys)), dtype=torch.long, device=device)
        self.invalid_node_count = torch.zeros(1, dtype=torch.long, device=device)
        self.device = self.counts.device
        self.actor_enabled = False
        self.collecting = False

    def reset(self, ids=None):
        """Clear only episode history on real reset; retain already earned counts.

        Call observe(done=...) before a normal reset. For an external/debug reset
        with no complete episode, reset directly; it earns no completed episode.
        """
        if ids is None:
            self.episode_has_rl.zero_()
            self.episode_started.zero_()
            self.episode_nodes.fill_(-1)
            return
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long or ids.device != self.device:
            raise ValueError("reset IDs must be a one-dimensional long tensor on the accumulator device")
        self.episode_has_rl[ids] = False
        self.episode_started[ids] = False
        self.episode_nodes[ids] = -1

    def begin_update(self, *, actor_enabled):
        if type(actor_enabled) is not bool:
            raise ValueError("actor_enabled must be bool")
        if self.collecting:
            raise ValueError("Flush the preceding rollout before starting another update")
        self.actor_enabled = actor_enabled
        self.collecting = True

    def _vector(self, tensor, name, dtype):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (self.num_envs,) or tensor.dtype != dtype or tensor.device != self.device:
            raise ValueError(f"{name} must have shape [num_envs], dtype {dtype}, device {self.device}")

    def observe(self, node_indices, learning_mask, done, episode_complete, *, key_phase_events=None):
        """Call once per policy transition, before reset erases episode history.

        node_indices=-1 denotes unlabelled foundation/rehearsal and never earns
        current-node exposure. A concrete node is latched throughout an episode.
        complete=True means a normal complete task opportunity/physical terminal,
        not a debug cutoff; failures remain in exposure but never pass evaluation.
        Events are once-per-occurrence pulses, not phase occupancy masks.
        """
        if not self.collecting:
            raise ValueError("begin_update must declare rollout ownership before observation")
        self._vector(node_indices, "node_indices", torch.long)
        for name, tensor in (("learning_mask", learning_mask), ("done", done), ("episode_complete", episode_complete)):
            self._vector(tensor, name, torch.bool)
        events = key_phase_events or {}
        if set(events) - set(self.event_keys):
            raise ValueError("Unknown key-phase event")
        for key, tensor in events.items():
            self._vector(tensor, key, torch.bool)
        valid_node = (node_indices >= 0) & (node_indices < len(self.node_ids))
        invalid = (node_indices < -1) | (node_indices >= len(self.node_ids))
        changed = self.episode_started & (node_indices != self.episode_nodes)
        self.invalid_node_count += (invalid | changed).sum().reshape(1)
        self.episode_nodes = torch.where(self.episode_started, self.episode_nodes, node_indices)
        self.episode_started.fill_(True)
        eligible = learning_mask & valid_node & ~changed & ~invalid & self.actor_enabled
        self.episode_has_rl |= eligible
        # A script failure following earlier actor motion is not a completed
        # actor-owned terminal. The genuine actor FAILED/disarm frame retains
        # learning_mask=True and is counted here.
        completed = done & episode_complete & self.episode_has_rl & eligible
        safe_nodes = node_indices.clamp(0, len(self.node_ids) - 1)
        self.counts[:, 0].scatter_add_(0, safe_nodes, eligible.long())
        self.counts[:, 1].scatter_add_(0, safe_nodes, completed.long())
        for column, key in enumerate(self.event_keys, start=2):
            if key in events:
                self.counts[:, column].scatter_add_(0, safe_nodes, (events[key] & eligible).long())
        self.episode_has_rl &= ~done
        self.episode_started &= ~done
        self.episode_nodes = torch.where(done, -1, self.episode_nodes)

    def flush(self, node_id, *, actor_optimized):
        """One device-to-host synchronization; returns current-node credit only.

        All other nodes are diagnostics. A failed/critic-only optimizer gets zero
        progression credit, even if its physical rollout contained RL samples.
        Flush does not clear cross-rollout episode ownership history.
        """
        if node_id not in self.node_indices or type(actor_optimized) is not bool:
            raise ValueError("Known reserved node ID and explicit actor_optimized bool required")
        if not self.collecting:
            raise ValueError("No unflushed rollout")
        if actor_optimized and not self.actor_enabled:
            raise ValueError("Cannot claim actor optimization from a critic-only rollout")
        flat = torch.cat((self.counts.reshape(-1), self.invalid_node_count))
        values = flat.detach().cpu().tolist()  # Only steady-state CPU sync in this adapter.
        if values[-1]:
            raise ValueError("Episode node changed before reset or invalid node index; exposure cannot be credited")
        width = self.counts.shape[1]
        rows = {nid: values[index * width:(index + 1) * width] for index, nid in enumerate(self.node_ids)}
        current = rows[node_id]
        allowed = self.node_phase_keys[node_id]
        exposure = dict(valid_rl_transitions=current[0] if actor_optimized else 0,
            completed_rl_episodes=current[1] if actor_optimized else 0,
            key_phase_counts={key: current[2 + self.event_keys.index(key)] if actor_optimized else 0 for key in allowed})
        diagnostics = {nid: dict(valid_rl_transitions=row[0], completed_rl_episodes=row[1])
                       for nid, row in rows.items() if row[0] or row[1]}
        self.counts.zero_()
        self.invalid_node_count.zero_()
        self.actor_enabled = False
        self.collecting = False
        return dict(exposure=exposure, diagnostics_by_node=diagnostics)

    def state_dict(self):
        return dict(schema="v6-autonomous-exposure-v1", definition_sha256=self.definition_sha256,
            num_envs=self.num_envs, actor_enabled=self.actor_enabled, collecting=self.collecting,
            **{key: getattr(self, key).clone() for key in
               ("episode_has_rl", "episode_started", "episode_nodes", "counts", "invalid_node_count")})

    def load_state_dict(self, state):
        if (state.get("schema") != "v6-autonomous-exposure-v1" or
                state.get("definition_sha256") != self.definition_sha256 or state.get("num_envs") != self.num_envs):
            raise ValueError("Exposure checkpoint identity mismatch")
        if type(state.get("actor_enabled")) is not bool or type(state.get("collecting")) is not bool:
            raise ValueError("Missing exposure rollout ownership state")
        restored = {}
        for key in ("episode_has_rl", "episode_started", "episode_nodes", "counts", "invalid_node_count"):
            original, value = getattr(self, key), state[key]
            if not isinstance(value, torch.Tensor) or value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError(f"Invalid exposure state tensor: {key}")
            restored[key] = value.to(self.device).clone()
        for key, value in restored.items():
            setattr(self, key, value)
        self.actor_enabled = state["actor_enabled"]
        self.collecting = state["collecting"]
