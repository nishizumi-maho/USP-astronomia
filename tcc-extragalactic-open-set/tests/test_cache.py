from pathlib import Path

from tccastro.utils.cache import CacheManager


def test_cache_path_deterministic(tmp_path: Path) -> None:
    cache = CacheManager(tmp_path)
    path1 = cache.path_for("legacy", 10.1234567, -5.1234567, "grz", 64)
    path2 = cache.path_for("legacy", 10.1234567, -5.1234567, "grz", 64)
    assert path1 == path2
    assert "legacy" in str(path1)
