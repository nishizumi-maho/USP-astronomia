from __future__ import annotations

import numpy as np

from tccastro.surveys.base import CutoutResult, SurveyClient


class PS1SurveyClient(SurveyClient):
    name: str = "ps1"

    def build_url(self, ra: float, dec: float, size: int, band: str) -> str:
        raise NotImplementedError("TODO: Implement PS1 cutout URL builder")

    def fetch_cutout(self, ra: float, dec: float, size: int, band: str) -> CutoutResult:
        raise NotImplementedError("TODO: Implement PS1 cutout fetch")

    def parse_to_array(self, payload: bytes) -> np.ndarray:
        raise NotImplementedError("TODO: Implement PS1 parse")
