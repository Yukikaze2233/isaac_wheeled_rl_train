"""Capture a conservative, local USD dependency closure using OpenUSD on CPU.

Sdf visits *authored* specs (including inactive/unselected variants), while
UsdUtils and a LoadAll stage independently check the captured tree. This does
not validate collisions, physics, mechanical limits, or native admission.
Only the stock file resolver, USD layers and self-contained image assets are
supported. Unknown schemes, packages, expressions and importer formats fail
closed; their dependency semantics are not guessed from strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import MappingProxyType
from collections.abc import Mapping


class DependencyClosureError(ValueError):
    pass


_LAYERS = frozenset({".usd", ".usda", ".usdc"})
_OPAQUE = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".tga", ".bmp", ".exr", ".hdr"})
_LIST_ITEMS = ("explicitItems", "addedItems", "prependedItems", "appendedItems", "deletedItems", "orderedItems")


def _digest(data):
    return sha256(data).hexdigest()


def _relative_path(root, containing_layer, authored):
    # Reject anything whose meaning could depend on a URI/package resolver,
    # environment expansion, a template, Windows path handling or a symlink.
    if (not isinstance(authored, str) or not authored or authored.startswith("~")
            or any(c in authored for c in ("\0", "\\", ":", "[", "]", "<", ">", "$", "{", "}", "*", "?", "#", "%"))
            or PurePosixPath(authored).is_absolute()):
        raise DependencyClosureError(f"Unsupported external/expressed asset path: {authored!r}")
    candidate = root / PurePosixPath(containing_layer).parent / authored
    # Normalize .. lexically before checking every path component for aliases.
    import os
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_relative_to(root):
        raise DependencyClosureError(f"Asset dependency escapes its root: {authored!r}")
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise DependencyClosureError(f"Symlink dependency is not a frozen local path: {current}")
    if not candidate.is_file():
        raise DependencyClosureError(f"Missing local asset dependency: {candidate}")
    return candidate.relative_to(root).as_posix()


@dataclass(frozen=True)
class DependencyFile:
    path: Path
    data: bytes
    digest: str
    kind: str


class UsdDependencyClosure:
    """Captured bytes plus the exact relative paths needed to reproduce them."""

    def __init__(self, root, root_layer, files, edges, statistics):
        self.asset_directory = root
        self.root_layer = root_layer
        self.files = MappingProxyType(dict(files))
        self._edges = tuple(sorted(set(edges)))
        self._statistics = dict(statistics)

    def receipt(self):
        tree = {"root_layer": self.root_layer,
                "files": {n: {"sha256": f.digest, "bytes": len(f.data), "kind": f.kind}
                          for n, f in sorted(self.files.items())},
                "edges": [{"layer": l, "spec": s, "field": k, "authored": a, "target": t}
                          for l, s, k, a, t in self._edges]}
        return {"schema": "local_usd_dependency_closure_v1", "asset_directory": str(self.asset_directory),
                **tree, "closure_sha256": _digest(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()),
                "inspection": dict(self._statistics), "native_qualification": False,
                "scope": "all supported authored USD file dependencies including unused variants; not collision or importer/native qualification"}

    def assert_unchanged(self):
        # Child-path checks alone cannot see replacement of the root itself or
        # one of its ancestors by an alias, even if all content hashes match.
        # Reestablish the same canonical-root invariant required at capture.
        root = self.asset_directory
        if not root.is_absolute() or not root.is_dir() or root.resolve() != root:
            raise DependencyClosureError("Captured asset root is no longer a canonical absolute local directory")
        for name, item in self.files.items():
            current = _relative_path(self.asset_directory, ".", name)
            if current != name or _digest(item.path.read_bytes()) != item.digest:
                raise DependencyClosureError(f"Captured USD dependency changed: {name}")


def capture_usd_dependency_closure(asset_directory, root_layer, *, declared_sha256=None):
    """Capture a fixed local tree, optionally requiring every dependency in a manifest.

    ``root_layer`` and manifest keys are asset-directory-relative POSIX paths.
    All source bytes are copied into a private temporary tree *before* stage
    composition, avoiding global Sdf caches of previously opened source layers.
    Caller must archive ``files`` at the same relative paths, retain the manifest
    separately and recheck bytes before use. This is not a filesystem freeze.
    """
    from pxr import Ar, Sdf, Usd, UsdUtils

    root = Path(asset_directory)
    if not root.is_absolute() or not root.is_dir() or root.resolve() != root:
        raise DependencyClosureError("Asset root must be a canonical absolute local directory")
    if type(Ar.GetUnderlyingResolver()) is not Ar.DefaultResolver or not Ar.GetResolver().GetCurrentContext().IsEmpty():
        raise DependencyClosureError("Only the unbound stock DefaultResolver is supported")
    root_name = _relative_path(root, ".", root_layer)
    if Path(root_name).suffix.lower() not in _LAYERS:
        raise DependencyClosureError("Root must be a supported USD layer")
    if declared_sha256 is not None:
        if not isinstance(declared_sha256, Mapping) or any(
            not isinstance(n, str) or not isinstance(h, str) or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h) for n, h in declared_sha256.items()):
            raise DependencyClosureError("Manifest dependency hashes must be explicit lowercase SHA-256")
    files, edges, layers = {}, [], {}
    counts = {"usd_version": list(Usd.GetVersion()), "resolver": "DefaultResolver", "authored_specs": 0,
              "variant_specs": 0, "asset_path_values": 0, "composition_arcs": 0}
    pending = [root_name]
    with TemporaryDirectory(prefix="usd-closure-") as directory:
        frozen = Path(directory)

        def add_edge(layer_name, spec, field, authored, *, composition=False):
            if not authored:  # Empty asset values and internal references need no file.
                return
            target = _relative_path(root, layer_name, authored)
            suffix = Path(target).suffix.lower()
            if suffix not in _LAYERS | _OPAQUE or (composition and suffix not in _LAYERS):
                raise DependencyClosureError(f"Unsupported dependency importer: {target}")
            edges.append((layer_name, str(spec), str(field), authored, target))
            if target not in files:
                pending.append(target)

        def inspect_value(value, layer_name, spec, field):
            if isinstance(value, Sdf.AssetPath):
                counts["asset_path_values"] += 1
                add_edge(layer_name, spec, field, value.path)
            elif isinstance(value, (Sdf.Reference, Sdf.Payload)):
                counts["composition_arcs"] += 1
                add_edge(layer_name, spec, field, value.assetPath, composition=True)
                if isinstance(value, Sdf.Reference):
                    inspect_value(value.customData, layer_name, spec, field + "/customData")
            elif isinstance(value, (Sdf.ReferenceListOp, Sdf.PayloadListOp)):
                for slot in _LIST_ITEMS:
                    for child in getattr(value, slot):
                        inspect_value(child, layer_name, spec, field + "/" + slot)
            elif isinstance(value, Mapping):
                for key, child in value.items():
                    if str(key) == "templateAssetPath" and child:
                        raise DependencyClosureError("USD clip template dependencies are unsupported")
                    inspect_value(child, layer_name, spec, field + "/" + str(key))
            elif isinstance(value, (Sdf.AssetPathArray, list, tuple)):
                for child in value:
                    inspect_value(child, layer_name, spec, field)

        while pending:
            name = pending.pop()
            if name in files:
                continue
            path = root / name
            raw = path.read_bytes()
            digest = _digest(raw)
            if declared_sha256 is not None and declared_sha256.get(name) != digest:
                raise DependencyClosureError(f"Dependency absent from manifest or SHA differs: {name}")
            is_layer = path.suffix.lower() in _LAYERS
            files[name] = DependencyFile(path, raw, digest, "usd_layer" if is_layer else "self_contained_image")
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            if not is_layer:
                continue
            try:
                layer = Sdf.Layer.FindOrOpen(str(destination))
                if layer is None or layer.GetFileFormat().formatId not in ("usd", "usda", "usdc"):
                    raise DependencyClosureError(f"Unsupported USD file format: {name}")
                layers[name] = layer
                for authored in layer.subLayerPaths:
                    add_edge(name, "/", "subLayers", authored, composition=True)
                paths = []
                layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
                for spec_path in paths:
                    counts["authored_specs"] += 1
                    counts["variant_specs"] += int(spec_path.ContainsPrimVariantSelection())
                    spec = layer.GetObjectAtPath(spec_path)
                    if spec is None and spec_path.IsTargetPath():
                        # Sdf traverses relationship/connection target paths,
                        # but USD exposes no object for these path-only entries.
                        # Their owning property's target list is inspected, as
                        # are any traversed child attribute specs. A scene path
                        # is not an external asset path.
                        continue
                    if spec is None:
                        raise DependencyClosureError(f"Uninspectable Sdf spec: {name}:{spec_path}")
                    for key in spec.ListInfoKeys():
                        if key in ("subLayers", "subLayerOffsets"):
                            continue  # Offsets are numerical, with no Python converter in some SDKs.
                        inspect_value(spec.GetInfo(key), name, spec_path, key)
                # File-format dependency APIs complement authored asset-valued fields.
                for authored in layer.GetExternalReferences():
                    add_edge(name, "/", "externalReferences", authored, composition=True)
                for authored in layer.GetExternalAssetDependencies():
                    add_edge(name, "/", "externalAssetDependencies", authored)
            except DependencyClosureError:
                raise
            except Exception as error:
                raise DependencyClosureError(f"Cannot fully inspect USD layer {name}: {error}") from error
        try:
            resolved_layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(frozen / root_name)))
            if unresolved:
                raise DependencyClosureError(f"Unresolved USD dependencies: {list(unresolved)!r}")
            resolved = [layer.realPath for layer in resolved_layers] + list(assets)
            for path in resolved:
                p = Path(path).resolve()
                if not p.is_relative_to(frozen) or p.relative_to(frozen).as_posix() not in files:
                    raise DependencyClosureError(f"Resolver discovered an uncaptured dependency: {path}")
            stage = Usd.Stage.Open(str(frozen / root_name), load=Usd.Stage.LoadAll)
            if stage is None or stage.GetCompositionErrors():
                raise DependencyClosureError("Captured USD tree has composition errors")
            prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
            counts.update(layers=len(layers), files=len(files), composed_prims=len(list(stage.TraverseAll())),
                          instances=sum(p.IsInstance() for p in prims), instance_proxies=sum(p.IsInstanceProxy() for p in prims),
                          usdutils_layers=len(resolved_layers), usdutils_assets=len(assets), unresolved=0)
        except DependencyClosureError:
            raise
        except Exception as error:
            raise DependencyClosureError(f"Captured tree composition could not be verified: {error}") from error
        result = UsdDependencyClosure(root, root_name, files, edges, counts)
        result.assert_unchanged()
        return result
