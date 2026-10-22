"""Shared CPU fixtures for current runtime tests."""
from types import SimpleNamespace as NS
import pytest
import torch

class MaterialView:

    def __init__(self, names, shapes):
        self.count, self.max_shapes, self.is_homogeneous = (3, sum(shapes), True)
        self.link_paths = [[f'/World/env_{i}/Robot/{n}' for n in names] for i in range(3)]
        self.materials = torch.ones(self.count, self.max_shapes, 3)
        self.materials[..., 2] = 0.0
        self.reads, self.writes = (0, 0)

    def get_material_properties(self):
        self.reads += 1
        return self.materials

    def set_material_properties(self, materials, ids):
        import warp as wp
        materials, ids = (wp.to_torch(materials), wp.to_torch(ids).long())
        assert materials.shape == self.materials.shape
        assert materials.device.type == ids.device.type == 'cpu'
        self.materials[ids] = materials[ids]
        self.writes += 1

def fixture(shapes=(1, 0, 2, 1, 3)):
    names = ('base_link', 'empty', 'R_link3', 'leg', 'L_link3')
    root = MaterialView(names, shapes)
    counts = dict(zip(root.link_paths[0], shapes))
    queried = []

    def create(path):
        queried.append(path)
        return NS(count=1, prim_paths=[path], max_shapes=counts[path])
    return (NS(root_view=root, body_names=names), NS(create_rigid_body_view=create), queried)

@pytest.fixture
def warp_runtime():
    import warp as wp
    wp.init()
