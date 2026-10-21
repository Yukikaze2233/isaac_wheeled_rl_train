"""Episode-latched curriculum allocation for the actual source flat sampler.

flat baseline's double-standing/heading, mutable-cache, special-command and timing
structure is retained. Sampling intervals are intersected with explicit local
capabilities and training anchors. An anchor is NOT a fixed evaluation case,
success event, evidence of coverage, or native admission. Coordinate transforms,
scripted stop/reverse, recovery chains, terrain and jump nodes are unsupported.
"""
from copy import deepcopy
import math

import torch

from wheeled_tasks.chassis.tasks.curricula.autonomous_curriculum import _digest, measurement_only, validate_definition
from wheeled_tasks.chassis.tasks.commands.restart_state import _restart_tensor
from wheeled_tasks.chassis.tasks.commands.flat_command_sampling import ScutFlatCommands, SPECIAL_MODES, COMMAND_STATE_VERSION, _COMMAND_TENSORS, _COMMAND_CLOCKS, _GROUND_SENTINELS


VERSION = "curriculum_source_flat_commands_v1"
KINDS = ("stand", "pure_translation", "pure_spin", "source_flat_ordinary")
ROLES = ("foundation", "current", "previous")
_LANES = ("episode_ids", "episode_node_indices", "sampling_node_indices", "category_indices", "case_anchor_indices")


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} requires a finite nonnegative number")
    return float(value)


def _caps(value):
    if not isinstance(value, dict) or set(value) != {"height_range_m", "vx_cap_m_s", "yaw_cap_rad_s"}:
        raise ValueError("Explicit height/vx/yaw capabilities are required")
    h = value["height_range_m"]
    if not isinstance(h, (list, tuple)) or len(h) != 2:
        raise ValueError("Height capability requires two bounds")
    lo, hi = (_number(x, "height capability") for x in h)
    vx, yaw = (_number(value[k], k) for k in ("vx_cap_m_s", "yaw_cap_rad_s"))
    if not .2 <= lo <= hi <= .42 or vx > 3. or yaw > 4.5 * math.pi:
        raise ValueError("Capabilities exceed the source flat domain")
    return (vx, yaw, lo, hi)


def _weights(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must cover its complete declared population")
    values = [_number(v, name) for v in value.values()]
    if not value or not math.isclose(sum(values), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError(f"{name} weights must sum to one")
    return value


class CurriculumFlatCommands(ScutFlatCommands):
    """Drop-in source command owner with an explicitly extended restart schema.

    ``sampling_plan`` is called only by nonempty, real episode reset. The outer
    mechanical reset transaction supplies ``next_episode_ids`` (long[K]); that
    integer check alone does not prove native writes. A stale/repeated reset is
    refused. The provider's curriculum state must be restored separately.

    Slots isolate new state from the frozen base owner's strict tensor inventory;
    the subclass serializes that entire extra state, never just the base owner.
    """
    __slots__ = ("_course",)

    def __init__(self, num_envs, device, *, definition, capabilities, sampling_plan, allow_provisional=False):
        definition = deepcopy(validate_definition(definition))
        if type(allow_provisional) is not bool or (definition.get("approval", {}).get("status") != "user_approved"
                and not allow_provisional and definition.get("evaluation", {}).get("mode") != "measurement_only"):
            raise ValueError("A provisional definition requires explicit CPU candidate scope")
        if not callable(sampling_plan):
            raise ValueError("A reset-only sampling_plan provider is required")
        limits = _caps(capabilities)
        super().__init__(num_envs, device)
        nodes = {n["id"]: deepcopy(n) for p in definition["phases"] for n in p["nodes"]}
        self._course = {"definition": definition, "definition_sha256": _digest(definition),
            "capabilities": deepcopy(capabilities), "limits": limits, "provider": sampling_plan,
            # A numeric command envelope is not evidence of learned ability.
            # V7 measurement-only has no reviewed ability/exposure gate, so
            # source special modes remain disabled even after their clocks.
            "special_eligibility": not measurement_only(definition),
            "nodes": nodes, "node_ids": tuple(nodes), "case_ids": tuple(definition["cases"]),
            "lanes": {n: torch.full((num_envs,), -1, dtype=torch.long, device=self.device) for n in _LANES},
            "caps": torch.zeros(num_envs, 4, dtype=torch.float32, device=self.device),
            "category_counts": torch.zeros(3, dtype=torch.long, device=self.device),
            "anchor_counts": torch.zeros(len(definition["cases"]), dtype=torch.long, device=self.device),
            "node_counts": torch.zeros(len(nodes), dtype=torch.long, device=self.device)}

    @property
    def episode_ids(self):
        return self._course["lanes"]["episode_ids"]

    @property
    def episode_node_indices(self):
        return self._course["lanes"]["episode_node_indices"]

    @property
    def category_indices(self):
        return self._course["lanes"]["category_indices"]

    def _node_anchors(self, node_id):
        c = self._course
        if node_id not in c["nodes"]:
            raise ValueError("Unknown curriculum node")
        node = c["nodes"][node_id]
        difficulty = node["difficulty"]
        if (node.get("recovery_lesson_fractions") or set(difficulty) - {
                "height_range_m", "vx_cap_m_s", "yaw_cap_rad_s", "yaw_cap_rps"}
                or {"yaw_cap_rad_s", "yaw_cap_rps"} <= set(difficulty)):
            raise ValueError("Unsupported source-flat node difficulty or recovery lesson")
        limits = c["limits"]
        requested = {"height_range_m": difficulty.get("height_range_m", list(limits[2:])),
                     "vx_cap_m_s": difficulty.get("vx_cap_m_s", limits[0]),
                     "yaw_cap_rad_s": (2 * math.pi * _number(difficulty["yaw_cap_rps"], "yaw rps")
                                       if "yaw_cap_rps" in difficulty else difficulty.get("yaw_cap_rad_s", limits[1]))}
        bounds = _caps(requested)
        if bounds[0] > limits[0] or bounds[1] > limits[1] or bounds[2] < limits[2] or bounds[3] > limits[3]:
            raise ValueError("Whole node exceeds explicit chassis capabilities")
        anchors = {}
        for cid in node["required_cases"]:
            case = c["definition"]["cases"][cid]
            kind, p = case["kind"], case["parameters"]
            if kind not in KINDS or not isinstance(p, dict):
                raise ValueError("Unsupported source-flat case anchor: " + cid)
            allowed = {"height_m", "vx_m_s", "yaw_rad_s", "height_range_m", "vx_cap_m_s", "yaw_cap_rad_s",
                "command_source", "implicit_drift_to_command_allowed", "warmup_s", "steady_s", "brake_s", "center_reference"}
            if set(p) - allowed or p.get("implicit_drift_to_command_allowed", False) is not False:
                raise ValueError("Unsupported coordinate/implicit command case anchor")
            h = [p["height_m"], p["height_m"]] if "height_m" in p else p.get("height_range_m", list(bounds[2:]))
            if kind == "source_flat_ordinary":
                vx, yaw = p.get("vx_cap_m_s", bounds[0]), p.get("yaw_cap_rad_s", bounds[1])
                if "vx_m_s" in p or "yaw_rad_s" in p:
                    raise ValueError("Ordinary anchor uses ranges/caps, not fixed requests")
            else:
                if not {"height_m", "vx_m_s", "yaw_rad_s"} <= set(p):
                    raise ValueError("Named stand/translation/spin anchor needs explicit axes and height")
                if any(type(p[k]) not in (int, float) or not math.isfinite(p[k]) for k in ("vx_m_s", "yaw_rad_s")):
                    raise ValueError("Anchor commands must be finite")
                vx, yaw = abs(p["vx_m_s"]), abs(p["yaw_rad_s"])
                if ((kind == "stand" and (vx != 0 or yaw != 0))
                        or (kind == "pure_translation" and (vx == 0 or yaw != 0))
                        or (kind == "pure_spin" and (vx != 0 or yaw == 0))):
                    raise ValueError("Anchor kind and explicit command axes disagree")
            cap = _caps(dict(height_range_m=h, vx_cap_m_s=vx, yaw_cap_rad_s=yaw))
            if cap[0] > bounds[0] or cap[1] > bounds[1] or cap[2] < bounds[2] or cap[3] > bounds[3]:
                raise ValueError("Whole node contains an anchor outside its explicit/source capabilities")
            anchors[cid] = cap
        return anchors

    def _validate_plan(self, plan):
        if not isinstance(plan, dict) or plan.get("latch_until_episode_reset") is not True:
            raise ValueError("An episode-latched sampling plan is required")
        current = plan.get("node_id")
        all_anchors = {current: self._node_anchors(current)}
        if plan.get("difficulty") != self._course["nodes"][current]["difficulty"]:
            raise ValueError("Plan difficulty differs from bound node")
        fractions = _weights(plan.get("fractions"), ROLES, "category")
        for name in ("foundation_nodes", "previous_nodes"):
            nodes = plan.get(name)
            if not isinstance(nodes, list) or len(nodes) != len(set(nodes)) or current in nodes:
                raise ValueError("Plan node populations must be unique and exclude current")
            for nid in nodes:
                all_anchors[nid] = self._node_anchors(nid)
        if set(plan["foundation_nodes"]) & set(plan["previous_nodes"]):
            raise ValueError("Foundation and previous populations overlap")
        if any(plan.get(k) for k in ("recovery_lesson_fractions", "recovery_lesson_node_ids", "recovery_total_fractions")):
            raise ValueError("Source-flat command owner cannot execute recovery lessons")
        _weights(plan.get("current_case_fractions"), all_anchors[current], "current anchor")
        foundation = {cid: nid for nid in plan["foundation_nodes"] for cid in all_anchors[nid]}
        if len(foundation) != sum(len(all_anchors[nid]) for nid in plan["foundation_nodes"]):
            raise ValueError("Foundation anchor ownership is ambiguous")
        for role, population, weights in (
                ("foundation", foundation, plan.get("foundation_case_fractions")),
                ("previous", plan["previous_nodes"], plan.get("previous_node_fractions"))):
            if population:
                _weights(weights, population, role)
            elif weights != {} or fractions[role] != 0:
                raise ValueError("Positive quota lacks its full sampling population")
        return all_anchors, foundation

    def _allocation(self, ids, weights, generator):
        """Randomized systematic allocation: <1 lane/category rounding error.

        Small partial-reset batches use a random offset and permutation, so a
        low-weight population is not permanently rounded away. Reset-lane
        fractions are not asserted to equal effective PPO transition fractions.
        """
        names = tuple(weights)
        if not ids.numel():
            return {name: ids for name in names}
        boundaries = torch.tensor(list(weights.values()), dtype=torch.float64, device=self.device).cumsum(0)
        boundaries[-1] = 1.  # Validated unit sum: close only floating-point roundoff.
        positions = (torch.arange(ids.numel(), dtype=torch.float64, device=self.device)
                     + torch.rand((), dtype=torch.float64, device=self.device, generator=generator)) / ids.numel()
        labels = torch.searchsorted(boundaries, positions, right=True)
        shuffled = ids[torch.randperm(ids.numel(), device=self.device, generator=generator)]
        return {name: shuffled[labels == i] for i, name in enumerate(names)}

    def reset(self, ids, heading, ground_mask, generator, iteration=0, *, next_episode_ids):
        ids = self._ids(ids)
        heading = self._vector(heading, "heading", torch.float32, ids)
        ground = self._vector(ground_mask, "ground_mask", torch.bool, ids)
        generator, iteration = self._generator(generator), self._iteration(iteration)
        if (not isinstance(next_episode_ids, torch.Tensor) or next_episode_ids.dtype != torch.long
                or next_episode_ids.device != self.device or next_episode_ids.shape != ids.shape
                or not torch.equal(next_episode_ids, self.episode_ids[ids] + 1)):
            raise ValueError("Real reset requires strict next_episode_ids = previous + 1, long[K]")
        if not ids.numel():
            return
        if not bool(torch.isfinite(heading).all()) or bool(self.fixed_request_mask.any()):
            raise ValueError("Finite heading and exclusive curriculum/source command ownership required")
        plan = deepcopy(self._course["provider"]())
        all_anchors, foundation = self._validate_plan(plan)  # Reject whole plan before RNG/state writes.
        c = self._course
        for role, selected in self._allocation(ids, plan["fractions"], generator).items():
            role_index = ROLES.index(role)
            c["category_counts"][role_index] += selected.numel()
            choices = []
            if role == "foundation":
                choices = [(anchor_ids, foundation[cid], cid) for cid, anchor_ids in
                           self._allocation(selected, plan["foundation_case_fractions"], generator).items()]
            elif role == "current":
                choices = [(anchor_ids, plan["node_id"], cid) for cid, anchor_ids in
                           self._allocation(selected, plan["current_case_fractions"], generator).items()]
            else:
                for nid, node_ids in self._allocation(selected, plan["previous_node_fractions"], generator).items():
                    weights = {cid: 1. / len(all_anchors[nid]) for cid in all_anchors[nid]}
                    choices += [(anchor_ids, nid, cid) for cid, anchor_ids in self._allocation(node_ids, weights, generator).items()]
            for anchor_ids, nid, cid in choices:
                ni, ci = c["node_ids"].index(nid), c["case_ids"].index(cid)
                for name, value in (("category_indices", role_index), ("sampling_node_indices", ni),
                                    ("episode_node_indices", -1 if role == "foundation" else ni), ("case_anchor_indices", ci)):
                    c["lanes"][name][anchor_ids] = value
                c["caps"][anchor_ids] = torch.tensor(all_anchors[nid][cid], dtype=torch.float32, device=self.device)
                c["anchor_counts"][ci] += anchor_ids.numel()
                c["node_counts"][ni] += anchor_ids.numel()
        self.episode_ids[ids] = next_episode_ids
        self._limit_cache(ids)  # Source first-standing retention cannot retain an old higher-cap command.
        self.episode_physics_ticks[ids] = 0
        self.command_counter[ids] = 0
        self.ground_birth[ids], self.holding[ids], self.ground_override_pending[ids] = ground, ground, ground
        self.ground_restore_command3[ids] = torch.nan
        self.ground_temporary_command3[ids] = torch.nan
        self._resample(ids, iteration, generator)
        self._sample_height(ids, generator)
        ground_ids = ids[ground]
        if ground_ids.numel():
            self.ground_restore_command3[ground_ids] = self.generator_command3[ground_ids]
            self.ground_temporary_command3[ground_ids, 0] = self._draw_limits(ground_ids, 0, -1., 1., generator)
            self.ground_temporary_command3[ground_ids, 1] = 0.
            self.ground_temporary_command3[ground_ids, 2] = self._draw_limits(ground_ids, 1, -1., 1., generator)
        self._publish(ids)

    def _draw_limits(self, ids, axis, low, high, generator):
        cap = self._course["caps"][ids, axis]
        lo, hi = torch.maximum(cap.new_tensor(low), -cap), torch.minimum(cap.new_tensor(high), cap)
        return lo + (hi - lo) * self._uniform(ids.numel(), 0., 1., generator)

    def _limit_cache(self, ids):
        cap = self._course["caps"][ids]
        self.generator_command3[ids, 0] = self.generator_command3[ids, 0].clamp(-cap[:, 0], cap[:, 0])
        self.generator_command3[ids, 1] = 0.
        self.generator_command3[ids, 2] = self.generator_command3[ids, 2].clamp(-cap[:, 1], cap[:, 1])

    @staticmethod
    def _ranges(ranges, cap):
        return [(max(lo, -cap), min(hi, cap)) for lo, hi in ranges
                if min(hi, cap) > max(lo, -cap) or (cap == 0 and lo <= 0 <= hi)]

    def _resample(self, ids, iteration, generator):
        n = ids.numel()
        if not n:
            return
        if bool((self.episode_ids[ids] < 0).any()):
            raise ValueError("Source sampling requires real-reset latched curriculum ownership")
        self.time_left[ids] = self._uniform(n, 5., 15., generator)
        self.command_counter[ids] += 1
        first = self._uniform(n, 0., 1., generator) <= .1
        self.standing_first_draw[ids], self.is_standing[ids] = first, first
        self.standing_second_draw[ids] = -1
        self.special_mode_id[ids] = -1
        nonstanding = ids[~first]
        ready = self.episode_physics_ticks[nonstanding] >= 1000
        candidate, delayed = nonstanding[ready], nonstanding[~ready]
        # Each anchor has one fixed cap; group only at command resampling, not
        # per physical tick. Preserve each lane's source probability mass and
        # random special-mode ordering; gated-out mass stays in normal sampling.
        eligible = candidate if self._course["special_eligibility"] else candidate[:0]
        for ci in self._course["lanes"]["case_anchor_indices"][eligible].unique().tolist():
            group = candidate[self._course["lanes"]["case_anchor_indices"][candidate] == ci]
            # An anchor may be shared across nodes with different ordinary caps.
            for cap in self._course["caps"][group].unique(dim=0):
                lanes = group[(self._course["caps"][group] == cap).all(dim=1)]
                vx_cap, yaw_cap = float(cap[0]), float(cap[1])
                active = [i for i, (_, start, _, vx, yaw) in enumerate(SPECIAL_MODES) if iteration >= start
                          and self._ranges(vx, vx_cap) and self._ranges(yaw, yaw_cap)]
                if active:
                    order = torch.randperm(len(active), device=self.device, generator=generator).tolist()
                    draw, cumulative = self._uniform(lanes.numel(), 0., 1., generator), 0.
                    for index in order:
                        mode = active[index]; upper = cumulative + SPECIAL_MODES[mode][2]
                        selected = lanes[(draw >= cumulative) & (draw < upper)]
                        self.special_mode_id[selected] = mode
                        if selected.numel():
                            self.generator_command3[selected, 0] = self._sample_ranges(self._ranges(SPECIAL_MODES[mode][3], vx_cap), selected.numel(), generator)
                            self.generator_command3[selected, 1] = 0.
                            self.generator_command3[selected, 2] = self._sample_ranges(self._ranges(SPECIAL_MODES[mode][4], yaw_cap), selected.numel(), generator)
                        cumulative = upper
        normal = torch.cat((delayed, candidate[self.special_mode_id[candidate] < 0]))
        if normal.numel():
            self.generator_command3[normal, 0] = self._draw_limits(normal, 0, -2.7, 2.7, generator)
            self.generator_command3[normal, 1] = self._uniform(normal.numel(), 0., 0., generator)
            self.generator_command3[normal, 2] = self._draw_limits(normal, 1, -2 * math.pi, 2 * math.pi, generator)
            self.heading_target[normal] = self._uniform(normal.numel(), -math.pi, math.pi, generator)
            self.is_heading[normal] = self._uniform(normal.numel(), 0., 1., generator) <= .5
            second = self._uniform(normal.numel(), 0., 1., generator) <= .1
            self.is_standing[normal], self.standing_second_draw[normal] = second, second.to(torch.int8)
        self.generator_after_resample3[ids] = self.generator_command3[ids]
        self.generator_cache_postprocessed[ids] = False

    def _sample_height(self, ids, generator):
        if ids.numel():
            limits = self._course["caps"][ids]
            self.height[ids] = limits[:, 2] + (limits[:, 3] - limits[:, 2]) * self._uniform(ids.numel(), 0., 1., generator)

    def _postprocess(self, heading):
        super()._postprocess(heading)
        self._limit_cache(torch.arange(self.num_envs, device=self.device))
        self.generator_after_postprocess3.copy_(self.generator_command3)

    def set_fixed_requests(self, *args, **kwargs):
        raise ValueError("Fixed evaluation requests cannot share a curriculum training command owner")

    def clear_fixed_requests(self, *args, **kwargs):
        raise ValueError("Fixed evaluation requests are not owned by this sampler")

    def diagnostics(self):
        result = super().diagnostics()
        result.update({"curriculum_" + k: v.clone() for k, v in self._course["lanes"].items()})
        result["curriculum_caps_vx_yaw_height_min_max"] = self._course["caps"].clone()
        result.update({"curriculum_reset_" + k: self._course[k].clone() for k in ("category_counts", "anchor_counts", "node_counts")})
        return result

    def summary(self):
        result = super().summary()
        if not hasattr(self, "_course"):
            return result
        result["curriculum_sampling"] = {"version": VERSION, "definition_sha256": self._course["definition_sha256"],
            "capabilities": deepcopy(self._course["capabilities"]), "node_ids": list(self._course["node_ids"]),
            "case_anchor_ids": list(self._course["case_ids"]), "categories": list(ROLES),
            "anchor_semantics": "training symmetric envelopes/kinds only; signed fixed-case targets, warmup/braking and evaluation credit are not reproduced",
            "quota_semantics": "randomized systematic allocation of actual reset lanes; not PPO transition fractions",
            "ground_and_retained_cache": "explicit capability bounds apply; hold/override clocks unchanged",
            "special_eligibility": self._course["special_eligibility"],
            "special_gate": ("source clocks retained; ability/exposure approval pending, all special modes disabled"
                             if not self._course["special_eligibility"] else
                             "legacy candidate source clocks AND command caps; not a V7 ability approval"),
            "provider_read": "only nonempty real reset; provider curriculum state is separately owned",
            "qualification": False}
        return result

    def state_dict(self, generator):
        self._restart_inventory()
        values = {}
        for name, (dtype, width) in _COMMAND_TENSORS.items():
            value = getattr(self, name)
            if name in _GROUND_SENTINELS:
                if not bool(torch.isnan(value[~self.ground_override_pending]).all()):
                    raise ValueError("Inactive ground command sentinel changed")
                value = torch.where(self.ground_override_pending[:, None], value, 0.)
            values[name] = value.detach().cpu().clone()
        source = {"schema": COMMAND_STATE_VERSION, "identity": self._restart_identity(), "tensors": values,
            "clocks": {n: getattr(self, n) for n in _COMMAND_CLOCKS}, "generator_state": self._generator(generator).get_state().clone()}
        extra = {**{k: v.detach().cpu().clone() for k, v in self._course["lanes"].items()},
                 **{k: self._course[k].detach().cpu().clone() for k in ("caps", "category_counts", "anchor_counts", "node_counts")}}
        state = {"schema": VERSION, "source": source, "curriculum": extra}
        self._prepare_restart(state, generator)
        return state

    def _prepare_restart(self, state, generator):
        if not isinstance(state, dict) or set(state) != {"schema", "source", "curriculum"} or state["schema"] != VERSION:
            raise ValueError("Complete curriculum command restart schema required")
        source = ScutFlatCommands._prepare_restart(self, state["source"], generator)
        if not self._course["special_eligibility"] and bool((source[0]["special_mode_id"] != -1).any()):
            raise ValueError("Measurement-only restart cannot restore unapproved special eligibility")
        c, raw = self._course, state["curriculum"]
        shapes = {**{k: (self.num_envs,) for k in _LANES}, "caps": (self.num_envs, 4),
            "category_counts": (3,), "anchor_counts": (len(c["case_ids"]),), "node_counts": (len(c["node_ids"]),)}
        for k, shape in shapes.items():
            target = c["lanes"][k] if k in _LANES else c[k]
            if (not isinstance(target, torch.Tensor) or tuple(target.shape) != shape
                    or target.dtype != (torch.float32 if k == "caps" else torch.long)
                    or target.device != self.device or target.layout != torch.strided
                    or target.requires_grad or not target.is_contiguous()):
                raise ValueError("Destination curriculum buffer shape/device/storage changed")
        if not isinstance(raw, dict) or set(raw) != set(shapes):
            raise ValueError("Complete curriculum tensor inventory required")
        checked = {k: _restart_tensor(raw[k], shape, torch.float32 if k == "caps" else torch.long, k)
                   for k, shape in shapes.items()}
        ids = checked["episode_ids"]
        if bool((ids < -1).any()):
            raise ValueError("Invalid episode identity")
        for k in ("category_counts", "anchor_counts", "node_counts"):
            if bool((checked[k] < 0).any()) or int(checked[k].sum()) != int((ids + 1).sum()):
                raise ValueError("Curriculum reset count reconciliation failed")
        for lane in range(self.num_envs):
            if int(ids[lane]) < 0:
                if any(int(checked[k][lane]) != -1 for k in _LANES) or bool((checked["caps"][lane] != 0).any()):
                    raise ValueError("Unassigned lane cannot contain curriculum ownership")
                continue
            ri, ni, ci = (int(checked[k][lane]) for k in ("category_indices", "sampling_node_indices", "case_anchor_indices"))
            if not 0 <= ri < 3 or not 0 <= ni < len(c["node_ids"]) or not 0 <= ci < len(c["case_ids"]):
                raise ValueError("Unknown restored curriculum category/node/anchor")
            anchors = self._node_anchors(c["node_ids"][ni])
            cid = c["case_ids"][ci]
            if cid not in anchors or int(checked["episode_node_indices"][lane]) != (-1 if ri == 0 else ni):
                raise ValueError("Restored anchor/exposure owner mismatch")
            if not torch.equal(checked["caps"][lane], torch.tensor(anchors[cid], dtype=torch.float32)):
                raise ValueError("Restored capabilities differ from construction-bound anchor")
            cap = checked["caps"][lane]
            for name in ("generator_command3", "command3", "ground_restore_command3", "ground_temporary_command3"):
                if name in _GROUND_SENTINELS and not bool(source[0]["ground_override_pending"][lane]):
                    continue
                command = source[0][name][lane].cpu()
                if command[1] != 0 or abs(float(command[0])) > float(cap[0]) or abs(float(command[2])) > float(cap[1]):
                    raise ValueError("Restored command exceeds latched capability")
            if not float(cap[2]) <= float(source[0]["height"][lane]) <= float(cap[3]):
                raise ValueError("Restored height exceeds latched capability")
        # Allocate all copies before touching the shared generator or any owner.
        return source, {k: v.to(self.device).clone() for k, v in checked.items()}

    def _commit_restart(self, prepared, generator):
        source, extra = prepared
        ScutFlatCommands._commit_restart(self, source, generator)
        for k, v in extra.items():
            (self._course["lanes"][k] if k in _LANES else self._course[k]).copy_(v)
