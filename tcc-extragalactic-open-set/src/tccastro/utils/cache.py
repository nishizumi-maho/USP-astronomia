from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheManager:
    root: Path

    def path_for(self, survey: str, ra: float, dec: float, band: str, size: int) -> Path:
        ra_str = f"{ra:.6f}".replace(".", "p")
        dec_str = f"{dec:.6f}".replace(".", "p").replace("-", "m")
        filename = f"{survey}_{band}_{size}_{ra_str}_{dec_str}.npz"
        return self.root / survey / band / filename

    def exists(self, path: Path) -> bool:
        return path.exists()

    def ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
