# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
import requests

from qai_hub_models.utils.asset_loaders import download_file


class FakeResponse:
    def __init__(
        self, status_code: int, body: bytes = b"", content_type: str = ""
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {
            "content-length": str(len(body)),
            "content-type": content_type,
        }
        self.closed = False

    def iter_content(self, block_size: int) -> Iterator[bytes]:
        for i in range(0, len(self._body), block_size):
            yield self._body[i : i + block_size]

    def close(self) -> None:
        self.closed = True


def drive_download(
    responses: list[FakeResponse],
    num_retries: int = 4,
) -> tuple[bytes | None, int]:
    """Drive download_file with a scripted sequence of responses.

    Returns (bytes written to destination or None if download raised,
    number of requests.get calls made).
    """
    calls = {"n": 0}

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        idx = calls["n"]
        calls["n"] += 1
        return responses[idx]

    with TemporaryDirectory() as tmpdir:
        dst = Path(tmpdir) / "model.pt"
        with mock.patch(
            "qai_hub_models.utils.asset_loaders.requests.get", side_effect=fake_get
        ):
            download_file(
                "https://example.com/model.pt", str(dst), num_retries=num_retries
            )
        return dst.read_bytes(), calls["n"]


def test_retries_on_503_then_succeeds() -> None:
    body = b"payload"
    written, num_calls = drive_download(
        [
            FakeResponse(503),
            FakeResponse(503),
            FakeResponse(200, body=body),
        ]
    )
    assert written == body
    assert num_calls == 3


def test_retries_on_502_504_500_429() -> None:
    body = b"payload"
    for code in (502, 504, 500, 429):
        written, num_calls = drive_download(
            [FakeResponse(code), FakeResponse(200, body=body)]
        )
        assert written == body, f"expected recovery for status {code}"
        assert num_calls == 2, f"expected retry for status {code}"


def test_raises_after_exhausting_retries_on_persistent_5xx() -> None:
    responses = [FakeResponse(503) for _ in range(5)]
    with pytest.raises(ValueError, match="status 503"):
        drive_download(responses, num_retries=4)


def test_non_retryable_4xx_raises_immediately() -> None:
    with pytest.raises(ValueError, match="status 404"):
        drive_download([FakeResponse(404)])


def test_no_retry_on_404_even_with_retries_available() -> None:
    calls = {"n": 0}

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        calls["n"] += 1
        return FakeResponse(404)

    with (
        TemporaryDirectory() as tmpdir,
        mock.patch(
            "qai_hub_models.utils.asset_loaders.requests.get", side_effect=fake_get
        ),
    ):
        dst = Path(tmpdir) / "model.pt"
        with pytest.raises(ValueError, match="status 404"):
            download_file("https://example.com/model.pt", str(dst), num_retries=4)
    assert calls["n"] == 1


def test_retries_on_connection_error_still_work() -> None:
    """Existing retry behavior for connection errors must be preserved."""
    body = b"payload"
    calls = {"n": 0}

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("boom")
        return FakeResponse(200, body=body)

    with TemporaryDirectory() as tmpdir:
        dst = Path(tmpdir) / "model.pt"
        with mock.patch(
            "qai_hub_models.utils.asset_loaders.requests.get", side_effect=fake_get
        ):
            download_file("https://example.com/model.pt", str(dst), num_retries=4)
        assert dst.read_bytes() == body
    assert calls["n"] == 2
