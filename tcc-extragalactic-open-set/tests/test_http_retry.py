from unittest.mock import Mock

import requests

from tccastro.utils.http import HttpClient


def test_http_retry(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise requests.RequestException("boom")
        response = Mock()
        response.status_code = 200
        response.content = b"ok"
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    client = HttpClient(retries=2, backoff_s=0.0, rate_limit_rps=0)
    payload = client.get("http://example.com")
    assert payload == b"ok"
    assert calls["count"] == 2
