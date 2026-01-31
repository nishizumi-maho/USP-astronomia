from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CutoutResult:
    array: np.ndarray
    meta: dict[str, Any]


class SurveyClient(ABC):
    name: str

    @abstractmethod
    def build_url(self, ra: float, dec: float, size: int, band: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch_cutout(self, ra: float, dec: float, size: int, band: str) -> CutoutResult:
        raise NotImplementedError

    @abstractmethod
    def parse_to_array(self, payload: bytes) -> np.ndarray:
        raise NotImplementedError
