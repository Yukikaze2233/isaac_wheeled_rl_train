"""Stdlib-only identity gate for the byte-pinned official Sim6 Grid ground."""
import hashlib
from pathlib import Path


GROUND_ASSET_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
    "Environments/Grid/default_environment.usd"
)
GROUND_ASSET_SHA256 = "78e9a1e72a8838a13d0f65c49cd487ab92e89233cd128b057730b5b5b4ca2164"


def verify_cached_ground(path: Path) -> dict:
    """Accept only the official cached asset, never an arbitrary USD override."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GROUND_ASSET_SHA256:
        raise ValueError("cached ground must match the pinned official Sim6 grid USD SHA256")
    return {"mode": "local_verified_official_asset", "path": str(path.resolve()),
            "source_url": GROUND_ASSET_URL, "sha256": digest, "size": len(raw)}
