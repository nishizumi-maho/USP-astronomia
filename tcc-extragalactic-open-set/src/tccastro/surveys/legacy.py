from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from tccastro.surveys.base import CutoutResult, SurveyClient
from tccastro.utils.http import HttpClient


@dataclass
class LegacySurveyClient(SurveyClient):
    name: str = "legacy"
    layer: str = "ls-dr10"
    fmt: str = "jpg"
    http_client: HttpClient | None = None

    def __post_init__(self) -> None:
        if self.http_client is None:
            self.http_client = HttpClient()

    def build_url(self, ra: float, dec: float, size: int, band: str) -> str:
        return (
            "https://www.legacysurvey.org/viewer/cutout"
            f"?ra={ra}&dec={dec}&size={size}&layer={self.layer}"
            f"&bands={band}&format={self.fmt}"
        )

    def fetch_cutout(self, ra: float, dec: float, size: int, band: str) -> CutoutResult:
        url = self.build_url(ra, dec, size, band)
        assert self.http_client is not None
        payload = self.http_client.get(url)
        array = self.parse_to_array(payload)
        return CutoutResult(
            array=array,
            meta={"ra": ra, "dec": dec, "size": size, "band": band, "url": url},
        )

    def parse_to_array(self, payload: bytes) -> np.ndarray:
        image = Image.open(BytesIO(payload)).convert("RGB")
        arr = np.array(image)
        return np.transpose(arr, (2, 0, 1))
