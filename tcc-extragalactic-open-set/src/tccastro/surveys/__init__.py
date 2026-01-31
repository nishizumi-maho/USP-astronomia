from __future__ import annotations

from tccastro.surveys.base import SurveyClient
from tccastro.surveys.legacy import LegacySurveyClient
from tccastro.surveys.ps1 import PS1SurveyClient
from tccastro.surveys.sdss import SDSSSurveyClient


def get_client(name: str, **kwargs) -> SurveyClient:
    key = name.lower()
    if key == "legacy":
        return LegacySurveyClient(**kwargs)
    if key == "sdss":
        return SDSSSurveyClient(**kwargs)
    if key == "ps1":
        return PS1SurveyClient(**kwargs)
    raise ValueError(f"Unknown survey: {name}")
