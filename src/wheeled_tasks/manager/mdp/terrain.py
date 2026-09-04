"""Rough-terrain support (height-field terrain set).

Two parts:
1. Pure-torch flag mapping (testable without Isaac Sim): from a TerrainImporter's
   per-patch sub-terrain names and per-env patch assignment, produce the
   stair / slope booleans consumed by StairStateMachine (7D mode flags).
2. `make_rm_rough_terrain_cfg()` — an Isaac Lab TerrainImporterCfg factory
   with a height-field generator covering flat / pyramid stairs / inverted
   (down-stair) / slopes / inverted slopes, RM-field flavored proportions.
   Isaac Lab is imported lazily so the mdp package stays unit-testable.

TerrainImporter fields consumed by the env (Isaac Lab 2.x):
    terrain.terrain_type_names : list[str]   sub-terrain name per patch (row-major)
    terrain.terrain_types      : Tensor(N,)  patch index each env stands on
"""
import torch

DEFAULT_STAIR_KEYWORDS = ("stair",)
DEFAULT_SLOPE_KEYWORDS = ("slope",)


def terrain_flags_from_types(
    patch_names: list[str],
    terrain_types: torch.Tensor,
    num_envs: int,
    device: str = "cpu",
    stair_keywords: tuple[str, ...] = DEFAULT_STAIR_KEYWORDS,
    slope_keywords: tuple[str, ...] = DEFAULT_SLOPE_KEYWORDS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each env's terrain patch to (stair_flag, slope_flag) booleans."""
    stair = torch.zeros(num_envs, dtype=torch.bool, device=device)
    slope = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for patch_idx, name in enumerate(patch_names):
        low = str(name).lower()
        ids = (terrain_types == patch_idx).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        if any(k in low for k in stair_keywords):
            stair[ids] = True
        if any(k in low for k in slope_keywords):
            slope[ids] = True
    return stair, slope


def make_rm_rough_terrain_cfg(
    size: float = 8.0,
    num_rows: int = 6,
    num_cols: int = 6,
    step_height: float = 0.125,
    slope_angle: float = 0.15,
    seed: int | None = None,
):
    """RM-flavored height-field terrain: 台阶 / 反向台阶 / 坡面 / 反向坡.

    Returns an isaaclab.terrains.TerrainImporterCfg (lazy import). Proportions
    mirror the proven rough task family: flat ~20%, stairs/slopes (up + inverse)
    share the rest so the stair state machine gets continuous exposure.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import (
        HfFlatTerrainCfg,
        HfInvertedPyramidStairsTerrainCfg,
        HfPyramidSlopesTerrainCfg,
        HfPyramidStairsTerrainCfg,
        TerrainGeneratorCfg,
        TerrainImporterCfg,
    )

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            size=(size, size),
            num_rows=num_rows,
            num_cols=num_cols,
            sub_terrains={
                "flat": HfFlatTerrainCfg(proportion=0.2),
                # 台阶(上)
                "pyramid_stairs": HfPyramidStairsTerrainCfg(
                    proportion=0.2, step_height=step_height, step_width=0.3, platform_len=2.0),
                # 反向台阶(下, cliff-like)
                "pyramid_stairs_inv": HfInvertedPyramidStairsTerrainCfg(
                    proportion=0.2, step_height=step_height, step_width=0.3, platform_len=2.0),
                # 坡面(上)
                "slopes": HfPyramidSlopesTerrainCfg(
                    proportion=0.2, slope_angle=slope_angle),
                # 反向坡(下)
                "slopes_inv": HfPyramidSlopesTerrainCfg(
                    proportion=0.2, slope_angle=-slope_angle),
            },
        ),
        max_init_terrain_level=num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
