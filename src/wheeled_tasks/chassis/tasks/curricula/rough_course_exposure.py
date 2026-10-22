"""Attribute optimized rough transitions to pre-reset terrain cells.

This observer has no authority over sampling, rewards, PPO masks or promotion.
Visited-cell episodes may overlap between cells. Completed episodes are whole
native episodes grouped by birth cell, not successes confined to that tile.
"""
import hashlib
import json

import torch


VERSION = "rough-course-actor-exposure-v1"
CELL_FIELDS = (
    "current_actor_transitions", "birth_actor_transitions",
    "visited_distinct_episodes", "birth_distinct_episodes",
    "complete_actor_episodes",
)
TOTAL_FIELDS = (
    "actor_transitions", "unmeasured_actor_transitions",
    "outside_grid_actor_transitions", "unlocated_birth_actor_transitions",
)
MODES = ("stand", "pure_translate", "pure_spin", "combined")
RELEASE_VERSION = "rough-initial-release-exposure-v1"
RELEASE_FIELDS = ("actor_transitions_after_release", "complete_actor_windows_after_release",
                  "complete_actor_window_policy_intervals_after_release")
RELEASE_POLICY_INTERVALS = 10  # The pinned 0.2 s initial release at 50 Hz.


class RoughCourseExposure:
    """Stage one rollout and commit once after a successful actor update.

    Live episode cursors are deliberately not restored across cold workers.
    Those workers start new physical episodes. A window beginning after tick
    one, or after a critic-only/discarded window, cannot claim a complete
    actor episode. Historical unclassified counts never acquire terrain labels.
    """

    def __init__(self, source_mask, grid, steps):
        if (not isinstance(source_mask, torch.Tensor) or source_mask.ndim != 1
                or source_mask.dtype != torch.bool or not bool(source_mask.any())
                or type(steps) is not int or steps < 1):
            raise ValueError("Rough exposure requires actual rough lanes and rollout length")
        self.device = source_mask.device
        self.source_mask = source_mask.detach().clone()
        self.ids = source_mask.nonzero(as_tuple=False).flatten()
        definition = grid["definition"]
        self.row_count, self.column_count = definition["rows"], definition["columns"]
        self.family_names = tuple(definition["keys"])
        self.column_family = grid["column_key_ids"].detach().clone()
        self.start_xy = grid["start_xy"].detach().clone()
        self.tile_size = torch.tensor(definition["tile_size_m"], device=self.device)
        if (type(self.row_count) is not int or self.row_count < 1
                or type(self.column_count) is not int or self.column_count < 1
                or not self.family_names or len(set(self.family_names)) != len(self.family_names)
                or self.column_family.shape != (self.column_count,)
                or self.column_family.dtype != torch.long
                or self.column_family.device != self.device
                or bool(((self.column_family < 0) | (self.column_family >= len(self.family_names))).any())
                or self.start_xy.shape != (2,) or self.start_xy.device != self.device
                or self.tile_size.shape != (2,)
                or not bool(torch.isfinite(self.start_xy).all())
                or not bool((torch.isfinite(self.tile_size) & (self.tile_size > 0)).all())):
            raise ValueError("Rough exposure requires the actual complete terrain grid")
        self.steps, self.rows, self.rollout_serial = steps, [], 0
        self.cell_count = self.row_count * self.column_count
        self.counts = torch.zeros(len(CELL_FIELDS), self.cell_count, dtype=torch.long, device=self.device)
        self.mode_counts = torch.zeros(len(MODES), self.cell_count, dtype=torch.long, device=self.device)
        self.totals = torch.zeros(len(TOTAL_FIELDS), dtype=torch.long, device=self.device)
        self.release_counts = torch.zeros(len(RELEASE_FIELDS), self.cell_count, dtype=torch.long, device=self.device)
        self.release_unclassified = 0
        self.updates = self.historical_unclassified_actor_transitions = 0
        identity = dict(version=VERSION, rows=self.row_count, columns=self.column_count,
            family_names=self.family_names, column_family=self.column_family.cpu().tolist(),
            start_xy=self.start_xy.cpu().tolist(), tile_size_m=self.tile_size.cpu().tolist(),
            source_mask=self.source_mask.cpu().tolist())
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        n = len(self.ids)
        self.live = {
            "episode_id": torch.full((n,), -1, dtype=torch.long, device=self.device),
            "tick": torch.zeros(n, dtype=torch.long, device=self.device),
            "birth_cell": torch.full((n,), -1, dtype=torch.long, device=self.device),
            "visited": torch.zeros(n, self.cell_count, dtype=torch.bool, device=self.device),
            "birth_counted": torch.zeros(n, dtype=torch.bool, device=self.device),
            "complete": torch.zeros(n, dtype=torch.bool, device=self.device),
            "release_complete": torch.zeros(n, dtype=torch.bool, device=self.device),
        }

    def _vector(self, value, name, *, dtype=None, columns=None):
        shape = self.source_mask.shape if columns is None else (len(self.source_mask), columns)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != self.device or dtype is not None and value.dtype != dtype
                or dtype is None and not value.is_floating_point()):
            raise ValueError(f"Rough exposure needs pre-reset {name} with shape {shape}")
        return value[self.ids].detach().clone()

    def _cells(self, positions):
        finite = torch.isfinite(positions[:, :2]).all(-1)
        xy = torch.where(finite[:, None], positions[:, :2], self.start_xy)
        cell = ((xy - self.start_xy) / self.tile_size).floor().long()
        inside = (finite & (cell[:, 0] >= 0) & (cell[:, 0] < self.row_count)
            & (cell[:, 1] >= 0) & (cell[:, 1] < self.column_count))
        index = cell[:, 0].clamp(0, self.row_count - 1) * self.column_count
        index += cell[:, 1].clamp(0, self.column_count - 1)
        return index, inside, finite

    def observe(self, index, diagnostics):
        if index == 0:
            self.rows = []
            self.rollout_serial += 1
        if index != len(self.rows) or not 0 <= index < self.steps:
            raise ValueError("Rough exposure requires consecutive actual rollout rows")
        position = self._vector(diagnostics["root_link_position_world"], "world position", columns=3)
        origins = self._vector(diagnostics["rough_birth_origins_world"], "birth origins", columns=3)
        command = self._vector(diagnostics["commands"], "effective command", columns=3)
        cell, inside, finite = self._cells(position)
        birth, birth_inside, _ = self._cells(origins)
        valid = self._vector(diagnostics["measurement_valid_mask"], "measurement mask", dtype=torch.bool)
        valid &= finite & torch.isfinite(command).all(-1)
        moving, turning = command[:, 0] != 0., command[:, 1] != 0.
        mode = moving.long() + 2 * turning.long()
        row = dict(cell=cell, inside=inside, birth=torch.where(birth_inside, birth, -1),
                   valid=valid, mode=mode)
        for key in ("episode_id", "episode_ticks", "done", "timeout", "boundary", "terminated"):
            row[key] = self._vector(diagnostics[key], key,
                dtype=torch.long if key in ("episode_id", "episode_ticks") else torch.bool)
        owner = diagnostics.get("source_flat_episode", {})
        row["release_known"] = torch.zeros_like(valid)
        row["initial_ground"] = torch.zeros_like(valid)
        if "ground" in owner and "episode_physics_ticks" in owner:
            row["initial_ground"] = self._vector(owner["ground"], "initial ground", dtype=torch.bool)
            physical_ticks = self._vector(owner["episode_physics_ticks"], "source physical ticks", dtype=torch.long)
            row["release_known"] = physical_ticks == row["episode_ticks"] * 4
        self.rows.append(row)

    def prepare(self, loss_mask):
        if (len(self.rows) != self.steps or not isinstance(loss_mask, torch.Tensor)
                or loss_mask.shape != (self.steps, len(self.source_mask))
                or loss_mask.dtype != torch.bool or loss_mask.device != self.device):
            raise ValueError("Rough exposure requires the complete final PPO mask")
        counts, modes, totals = torch.zeros_like(self.counts), torch.zeros_like(self.mode_counts), torch.zeros_like(self.totals)
        release_counts = torch.zeros_like(self.release_counts)
        release_unclassified = torch.zeros((), dtype=torch.long, device=self.device)
        live = {key: value.clone() for key, value in self.live.items()}
        lane = torch.arange(len(self.ids), device=self.device)
        invalid_birth_change = torch.zeros((), dtype=torch.bool, device=self.device)
        for index, row in enumerate(self.rows):
            new = row["episode_id"] != live["episode_id"]
            contiguous = row["episode_ticks"] == live["tick"] + 1
            invalid_birth_change |= (~new & (row["birth"] != live["birth_cell"])).any()
            live["visited"][new] = False
            live["birth_counted"][new] = False
            live["complete"] = torch.where(new, row["episode_ticks"] == 1,
                                            live["complete"] & contiguous)
            paid = loss_mask[index, self.ids]
            measured = paid & row["valid"]
            current = measured & row["inside"]
            located_birth = measured & (row["birth"] >= 0)
            birth = row["birth"].clamp_min(0)
            totals += torch.stack((paid.sum(), (paid & ~row["valid"]).sum(),
                (measured & ~row["inside"]).sum(), (measured & ~located_birth).sum()))
            counts[0].scatter_add_(0, row["cell"], current.long())
            counts[1].scatter_add_(0, birth, located_birth.long())
            first_visit = current & ~live["visited"][lane, row["cell"]]
            counts[2].scatter_add_(0, row["cell"], first_visit.long())
            live["visited"][lane, row["cell"]] |= current
            first_birth = located_birth & ~live["birth_counted"]
            counts[3].scatter_add_(0, birth, first_birth.long())
            live["birth_counted"] |= first_birth
            live["complete"] &= current
            complete = (row["done"] & row["timeout"] & ~row["boundary"]
                & ~row["terminated"] & live["complete"] & (row["birth"] >= 0))
            counts[4].scatter_add_(0, birth, complete.long())
            # Only the known intentional prefix may be absent. Missing actor
            # samples after release, partial cold windows and gaps never restart
            # completeness at a convenient later interval.
            released = (row["release_known"] & row["initial_ground"]
                        & (row["episode_ticks"] > RELEASE_POLICY_INTERVALS))
            start = released & (row["episode_ticks"] == RELEASE_POLICY_INTERVALS + 1) & (new | contiguous)
            live["release_complete"] = (
                torch.where(new, start, (live["release_complete"] & contiguous) | start)
                & released & current)
            release_counts[0].scatter_add_(0, row["cell"], (released & current).long())
            release_finished = (row["done"] & row["timeout"] & ~row["boundary"]
                                & ~row["terminated"] & live["release_complete"] & (row["birth"] >= 0))
            release_counts[1].scatter_add_(0, birth, release_finished.long())
            release_counts[2].scatter_add_(0, birth,
                release_finished.long() * (row["episode_ticks"] - RELEASE_POLICY_INTERVALS).clamp_min(0))
            release_unclassified += (paid & ~row["release_known"]).sum()
            modes.view(-1).scatter_add_(0, row["mode"] * self.cell_count + row["cell"], current.long())
            live["episode_id"], live["tick"], live["birth_cell"] = (
                row["episode_id"].clone(), row["episode_ticks"].clone(), row["birth"].clone())
            # A terminal observation cannot later continue as the same episode.
            live["complete"] &= ~row["done"]
            live["release_complete"] &= ~row["done"]
        if bool(invalid_birth_change):
            raise ValueError("Rough birth cell changed inside a pre-reset native episode")
        return dict(generation=self.updates, rollout_serial=self.rollout_serial,
                    counts=counts, mode_counts=modes, totals=totals, live=live,
                    release_counts=release_counts, release_unclassified=release_unclassified)

    def commit(self, prepared):
        if (prepared["generation"] != self.updates or len(self.rows) != self.steps
                or prepared["rollout_serial"] != self.rollout_serial):
            raise ValueError("Rough exposure may commit a prepared actor window only once")
        self.counts += prepared["counts"]
        self.mode_counts += prepared["mode_counts"]
        self.totals += prepared["totals"]
        self.release_counts += prepared["release_counts"]
        self.release_unclassified += int(prepared["release_unclassified"])
        self.live, self.rows = prepared["live"], []
        self.updates += 1
        by_family = torch.zeros(len(CELL_FIELDS), self.row_count, len(self.family_names),
                               dtype=torch.long, device=self.device)
        by_family.index_add_(2, self.column_family,
                            prepared["counts"].view(len(CELL_FIELDS), self.row_count, self.column_count))
        values = by_family.cpu().tolist()
        result = {f"rough/{field}/row{row}/{name}": values[i][row][j]
            for i, field in enumerate(CELL_FIELDS) for row in range(self.row_count)
            for j, name in enumerate(self.family_names)}
        result.update({"rough/" + key: value for key, value in
                       zip(TOTAL_FIELDS, prepared["totals"].cpu().tolist())})
        result.update({"rough/mode/" + key: value for key, value in
                       zip(MODES, prepared["mode_counts"].sum(-1).cpu().tolist())})
        release_family = torch.zeros(len(RELEASE_FIELDS), self.row_count, len(self.family_names),
                                     dtype=torch.long, device=self.device)
        release_family.index_add_(2, self.column_family,
            prepared["release_counts"].view(len(RELEASE_FIELDS), self.row_count, self.column_count))
        values = release_family.cpu().tolist()
        result.update({f"rough/release/{field}/row{row}/{name}": values[i][row][j]
            for i, field in enumerate(RELEASE_FIELDS) for row in range(self.row_count)
            for j, name in enumerate(self.family_names)})
        result["rough/release/unclassified_actor_transitions"] = int(prepared["release_unclassified"])
        return result

    def state_dict(self):
        return dict(version=VERSION, grid_identity=self.identity,
            cell_fields=list(CELL_FIELDS), total_fields=list(TOTAL_FIELDS), mode_names=list(MODES),
            rows=self.row_count, columns=self.column_count, family_names=list(self.family_names),
            column_family=self.column_family.cpu().tolist(),
            successful_actor_updates=self.updates,
            historical_unclassified_actor_transitions=self.historical_unclassified_actor_transitions,
            cell_counts=self.counts.view(len(CELL_FIELDS), self.row_count, self.column_count).cpu().tolist(),
            mode_counts=self.mode_counts.view(len(MODES), self.row_count, self.column_count).cpu().tolist(),
            totals=self.totals.cpu().tolist(),
            initial_release_exposure=dict(version=RELEASE_VERSION,
                release_policy_intervals=RELEASE_POLICY_INTERVALS, fields=list(RELEASE_FIELDS),
                cell_counts=self.release_counts.view(len(RELEASE_FIELDS), self.row_count, self.column_count).cpu().tolist(),
                unclassified_actor_transitions=self.release_unclassified,
                scope="separate ground-release tail windows; no success or whole-episode claim; live cursors reset on cold load"),
            scope="unique successful actor-loss samples; visited episodes overlap cells; complete episodes are whole episodes grouped by birth, not tracking successes; cold workers reset live cursors")

    def load_state_dict(self, state):
        fresh = self.state_dict()
        mutable = {"successful_actor_updates", "historical_unclassified_actor_transitions", "cell_counts",
                   "mode_counts", "totals", "initial_release_exposure"}
        if any(state.get(key) != value for key, value in fresh.items() if key not in mutable):
            raise ValueError("Rough exposure checkpoint differs from the actual grid/population")
        for key in ("successful_actor_updates", "historical_unclassified_actor_transitions"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError("Rough exposure checkpoint has invalid counters")
        tensors = {}
        for key, shape in (("cell_counts", (len(CELL_FIELDS), self.row_count, self.column_count)),
                           ("mode_counts", (len(MODES), self.row_count, self.column_count)),
                           ("totals", (len(TOTAL_FIELDS),))):
            value = state.get(key)
            def integers(item):
                return all(integers(v) for v in item) if isinstance(item, list) else type(item) is int and item >= 0
            if not integers(value):
                raise ValueError("Rough exposure checkpoint requires nonnegative integer tables")
            tensor = torch.tensor(value, dtype=torch.long, device=self.device)
            if tensor.shape != shape:
                raise ValueError("Rough exposure checkpoint has invalid table dimensions")
            tensors[key] = tensor
        counts, modes, totals = (tensors[key] for key in ("cell_counts", "mode_counts", "totals"))
        if (not torch.equal(modes.sum(0), counts[0])
                or int(counts[0].sum() + totals[1] + totals[2]) != int(totals[0])
                or int(counts[1].sum() + totals[1] + totals[3]) != int(totals[0])
                or bool((counts[2] > counts[0]).any())
                or bool((counts[3] > counts[1]).any())
                or bool((counts[4] > counts[3]).any())):
            raise ValueError("Rough exposure checkpoint has inconsistent cell accounting")
        release = state.get("initial_release_exposure")
        release_counts = torch.zeros_like(self.release_counts)
        release_unclassified = int(totals[0])
        if release is not None:
            expected = fresh["initial_release_exposure"]
            if (not isinstance(release, dict) or set(release) != set(expected)
                    or any(release[key] != expected[key] for key in expected
                           if key not in ("cell_counts", "unclassified_actor_transitions"))
                    or type(release.get("unclassified_actor_transitions")) is not int
                    or release["unclassified_actor_transitions"] < 0
                    or not integers(release.get("cell_counts"))):
                raise ValueError("Rough initial release checkpoint has invalid identity or counters")
            release_counts = torch.tensor(release["cell_counts"], dtype=torch.long, device=self.device)
            if release_counts.shape != (len(RELEASE_FIELDS), self.row_count, self.column_count):
                raise ValueError("Rough initial release checkpoint has invalid dimensions")
            release_unclassified = release["unclassified_actor_transitions"]
            if (bool((release_counts[0] > counts[0]).any())
                    or bool((release_counts[1] > counts[3]).any())
                    or bool((release_counts[2] < release_counts[1]).any())
                    or int(release_counts[2].sum()) > int(release_counts[0].sum())
                    or int(release_counts[0].sum()) + release_unclassified > int(totals[0])):
                raise ValueError("Rough initial release checkpoint has inconsistent accounting")
            release_counts = release_counts.flatten(1)
        self.counts, self.mode_counts = counts.flatten(1), modes.flatten(1)
        self.release_counts, self.release_unclassified = release_counts, release_unclassified
        self.totals, self.updates = totals, state["successful_actor_updates"]
        self.historical_unclassified_actor_transitions = state["historical_unclassified_actor_transitions"]
