"""Per-terrain command overrides (per-terrain command envelopes).

Different terrain patches demand different command envelopes: stairs want
reduced forward speed, dash terrain wants the full range. Vectorized: envs are
grouped by matched profile with boolean masks — no per-env Python loop, so
4096-env resamples stay on-device.
"""
import torch


class TerrainCommandOverride:
    def __init__(self, profiles: dict[str, dict], default: dict | None = None):
        """profiles: patch-name-keyword -> command ranges, e.g.
            {"stair": {"vx": (-1.0, 1.0), "yaw_rate": (-1.0, 1.0)},
             "slope": {"vx": (-0.8, 0.8)},
             "dash_pad": {"vx": [(2.0, 3.0)]}}
        default: ranges for patches matching no profile.
        """
        self.profiles = {k.lower(): v for k, v in profiles.items()}
        self.default = default or {}

    @staticmethod
    def _sample(spec, mask: torch.Tensor, device) -> torch.Tensor:
        """Uniform sample for the `mask.sum()` envs; returns full-size tensor."""
        n = mask.shape[0]
        out = torch.zeros(n, device=device)
        m = int(mask.sum())
        if m == 0:
            return out
        if isinstance(spec[0], (int, float)):
            out[mask] = (spec[1] - spec[0]) * torch.rand(m, device=device) + spec[0]
        else:
            widths = torch.tensor([hi - lo for lo, hi in spec], device=device, dtype=torch.float)
            pick = torch.multinomial(widths, m, replacement=True)
            for i, (lo, hi) in enumerate(spec):
                sel = mask & (pick == i)
                c = int(sel.sum())
                if c:
                    out[sel] = (hi - lo) * torch.rand(c, device=device) + lo
        return out

    def apply(self, patch_names: list[str], env_patch_idx: torch.Tensor,
              command: torch.Tensor) -> torch.Tensor:
        """Rewrite command[:, 0](vx) and [:, 2](yaw_rate) per env by its patch.

        Vectorized over envs: one mask pass per profile, remaining envs get the
        default envelope. command is modified in place and returned.
        """
        n, device = command.shape[0], command.device
        names = [str(p).lower() for p in patch_names]
        # patch index -> matched profile index (-1 = default)
        patch_profile = torch.full((len(names),), -1, dtype=torch.long)
        for pi, name in enumerate(names):
            for ki, keyword in enumerate(self.profiles):
                if keyword in name:
                    patch_profile[pi] = ki
                    break
        env_profile = patch_profile.to(device)[env_patch_idx]  # (N,)

        claimed = torch.zeros(n, dtype=torch.bool, device=device)
        for ki, ranges in enumerate(self.profiles.values()):
            mask = env_profile == ki
            claimed |= mask
            if "vx" in ranges:
                command[:, 0] = torch.where(mask, self._sample(ranges["vx"], mask, device),
                                            command[:, 0])
            if "yaw_rate" in ranges:
                command[:, 2] = torch.where(mask, self._sample(ranges["yaw_rate"], mask, device),
                                            command[:, 2])
        rest = ~claimed
        if rest.any():
            if "vx" in self.default:
                command[rest, 0] = self._sample(self.default["vx"], rest, device)[rest]
            if "yaw_rate" in self.default:
                command[rest, 2] = self._sample(self.default["yaw_rate"], rest, device)[rest]
        return command
