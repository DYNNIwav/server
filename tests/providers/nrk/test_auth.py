"""Tests for the NRK login flow and password hashing."""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from music_assistant_models.errors import LoginFailed, ResourceTemporarilyUnavailable

from music_assistant.providers.nrk.auth import NrkAuth, hash_password
from music_assistant.providers.nrk.models import NrkApiError

from .conftest import FakeResponse, FakeSession, redirect_to

CALLBACK = "https://innlogging.nrk.no/connect/authorize/callback?client_id=radio&state=abc"
RECIPE = {"algorithm": "cscrypt:17:8:1:32", "salt": "jrS73PwX03muCGR+grDkvw=="}


def _token(user: str = "user-1", token: str = "token-1") -> dict[str, Any]:  # noqa: S107
    return {
        "session": {
            "accessToken": token,
            "user": {"sub": user},
            "serverEpochExpiry": time.time() + 3600,
        }
    }


def _routes(logginn_status: int = 200) -> dict[str, Any]:
    return {
        "GET /auth/web/login": FakeResponse(history=redirect_to(CALLBACK)),
        "POST /getHashingInstructions": FakeResponse({"current": RECIPE}),
        "POST /logginn": FakeResponse(status=logginn_status),
        "GET /connect/authorize/callback": FakeResponse(),
        "POST /auth/csrf_init": FakeResponse(),
        "GET /auth/contextinfo": FakeResponse({}),
        "POST /auth/session/tokenforsub/_": FakeResponse(_token()),
    }


def _auth(email: str = "user@example.com", password: str = "pw") -> NrkAuth:  # noqa: S107
    return NrkAuth(MagicMock(), logging.getLogger("test"), email, password)


def _patch_session(session: FakeSession) -> Any:
    return patch("music_assistant.providers.nrk.auth.create_clientsession", return_value=session)


def test_hash_password_matches_nrk() -> None:
    """The scrypt hash matches the value NRK's login page computes for the same recipe."""
    assert hash_password("pw", RECIPE) == (
        "2ae428184cf4834545424031610f43df0487dc5a2bca2f245c5346fe234d01a7"
    )


def test_hash_password_rejects_unknown_recipe() -> None:
    """A hashing scheme NRK has not used before fails loudly instead of sending garbage."""
    with pytest.raises(NrkApiError):
        hash_password("pw", {"algorithm": "bcrypt:10", "salt": "x"})


async def test_login_flow() -> None:
    """A full login sends the hashed password and returns the user id and token."""
    session = FakeSession(_routes())
    with (
        _patch_session(session),
        patch("music_assistant.providers.nrk.auth.hash_password", return_value="hashed"),
    ):
        assert await _auth().get_token() == ("user-1", "token-1")
    login = next(call for call in session.calls if call["url"].endswith("/logginn"))
    assert login["json"]["hashedPassword"]["current"] == {"recipe": RECIPE, "hash": "hashed"}
    callback = next(call for call in session.calls if "callback" in call["url"])
    assert callback["params"] == {"client_id": "radio", "state": "abc"}


async def test_rejected_password_fails_login() -> None:
    """Wrong credentials surface as LoginFailed, which MA does not retry."""
    with (
        _patch_session(FakeSession(_routes(logginn_status=401))),
        patch("music_assistant.providers.nrk.auth.hash_password", return_value="hashed"),
        pytest.raises(LoginFailed),
    ):
        await _auth().get_token()


async def test_outage_is_temporary() -> None:
    """A network or server failure is temporary, never a credentials error."""
    session = FakeSession({})
    session.request = MagicMock(side_effect=aiohttp.ClientConnectionError("down"))  # type: ignore[method-assign]
    with _patch_session(session), pytest.raises(ResourceTemporarilyUnavailable) as err:
        await _auth().get_token()
    assert not isinstance(err.value, LoginFailed)


async def test_token_is_reused_until_expiry() -> None:
    """A valid token is returned without another login."""
    session = FakeSession(_routes())
    auth = _auth()
    with (
        _patch_session(session),
        patch("music_assistant.providers.nrk.auth.hash_password", return_value="hashed"),
    ):
        await auth.get_token()
        calls = len(session.calls)
        await auth.get_token()
    assert len(session.calls) == calls


async def test_refresh_does_not_send_the_password_again() -> None:
    """An expired token is refreshed on the NRK session cookies."""
    auth = _auth()
    auth._store(_token(token="old"))
    auth._expiry = 0
    session = FakeSession(_routes())
    session.routes["POST /auth/session/tokenforsub/_"] = FakeResponse(_token(token="new"))
    with _patch_session(session):
        assert await auth.get_token() == ("user-1", "new")
    assert not any(call["url"].endswith("/logginn") for call in session.calls)


async def test_no_credentials() -> None:
    """Without an account, asking for a token is a login failure."""
    auth = _auth(email="", password="")
    assert not auth.enabled
    with pytest.raises(LoginFailed):
        await auth.get_token()


async def test_captcha_login_is_reported_and_not_retried() -> None:
    """NRK's captcha wall is named in the error, and the login is not hammered again."""
    session = FakeSession(_routes())
    session.routes["POST /logginn"] = FakeResponse(
        {"type": "captcha-validation-error", "title": "Captcha validation failed."}, status=400
    )
    auth = _auth()
    with (
        _patch_session(session),
        patch("music_assistant.providers.nrk.auth.hash_password", return_value="hashed"),
    ):
        with pytest.raises(LoginFailed, match="captcha"):
            await auth.get_token()
        calls = len(session.calls)
        with pytest.raises(LoginFailed):
            await auth.get_token()
    assert len(session.calls) == calls
