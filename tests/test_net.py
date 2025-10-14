from datetime import UTC, datetime, timedelta

import httpx

import net


def test_parse_retry_after_seconds() -> None:
    response = httpx.Response(429, headers={"Retry-After": "2"})
    wait = net.parse_retry_after(response)
    assert wait is not None
    assert 1.9 <= wait <= 2.1


def test_parse_retry_after_http_date() -> None:
    future = datetime.now(UTC) + timedelta(seconds=5)
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response = httpx.Response(429, headers={"Retry-After": header})
    wait = net.parse_retry_after(response)
    assert wait is not None
    assert wait >= 0.0
    assert wait <= 5.0
