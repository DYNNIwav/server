"""Tests for logging in to NRK with a saved radio.nrk.no session instead of a password."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from music_assistant_models.enums import EventType, ProviderFeature
from music_assistant_models.errors import LoginFailed, ResourceTemporarilyUnavailable
from yarl import URL

from music_assistant.providers.nrk import auth as auth_module
from music_assistant.providers.nrk import provider as provider_module
from music_assistant.providers.nrk.auth import NrkAuth
from music_assistant.providers.nrk.models import NrkAuthError
from music_assistant.providers.nrk.provider import CONF_SESSION, NrkProvider

from .conftest import FakeResponse, FakeSession
from .test_progress_sync import _feed_episode, _link_library, _played

SESSION = "nrk-session=abc; nrk-login-context-info=xyz"


def _token(user: str = "user-1", token: str = "token-1") -> dict[str, Any]:  # noqa: S107
    return {
        "session": {
            "accessToken": token,
            "user": {"sub": user},
            "serverEpochExpiry": time.time() + 3600,
        }
    }


def _routes(token: FakeResponse | None = None) -> dict[str, Any]:
    return {
        "POST /auth/csrf_init": FakeResponse(),
        "GET /auth/contextinfo": FakeResponse({}),
        "POST /auth/session/tokenforsub/_": token or FakeResponse(_token()),
    }


def _auth(session: str = SESSION, save: Mock | None = None) -> NrkAuth:
    return NrkAuth(
        MagicMock(), logging.getLogger("test"), "", "", session=session, save_session=save
    )


def _patch_session(session: FakeSession) -> Any:
    return patch("music_assistant.providers.nrk.auth.create_clientsession", return_value=session)


def _jar_cookies(auth: NrkAuth) -> dict[str, str]:
    return {
        key: morsel.value
        for key, morsel in auth._jar.filter_cookies(URL("https://radio.nrk.no/")).items()
    }


class RotatingResponse(FakeResponse):
    """A token response that also sets a new session cookie, as NRK does on refresh."""

    def __init__(self, auth: NrkAuth, payload: Any) -> None:
        """Initialize the response."""
        super().__init__(payload)
        self.auth = auth

    async def __aenter__(self) -> Self:
        """Rotate the session cookie on the way in."""
        self.auth._jar.update_cookies({"nrk-session": "rotated"}, URL("https://radio.nrk.no/"))
        return self


async def test_saved_session_gets_a_token_without_a_password() -> None:
    """The saved cookies go to NRK's token step, and no password login is attempted."""
    auth = _auth()
    session = FakeSession(_routes())
    assert auth.enabled
    with _patch_session(session):
        assert await auth.get_token() == ("user-1", "token-1")
    assert _jar_cookies(auth) == {"nrk-session": "abc", "nrk-login-context-info": "xyz"}
    assert not any("/logginn" in call["url"] for call in session.calls)


@pytest.mark.parametrize(
    "response",
    [FakeResponse(status=401), FakeResponse(status=403), FakeResponse({"session": None})],
)
async def test_expired_session_asks_for_a_new_login(response: FakeResponse) -> None:
    """A session NRK no longer accepts is named as a login the user has to redo."""
    with (
        _patch_session(FakeSession(_routes(response))),
        pytest.raises(LoginFailed, match="NRK login needed"),
    ):
        await _auth().get_token()


async def test_outage_with_a_session_is_temporary() -> None:
    """A server error on the token step keeps the session and is retried."""
    with (
        _patch_session(FakeSession(_routes(FakeResponse(status=503)))),
        pytest.raises(ResourceTemporarilyUnavailable) as err,
    ):
        await _auth().get_token()
    assert not isinstance(err.value, LoginFailed)


async def test_rotated_cookies_are_saved() -> None:
    """When NRK renews the session cookie, the new one is saved so a restart keeps it."""
    save = Mock()
    auth = _auth(save=save)
    session = FakeSession(_routes(RotatingResponse(auth, _token())))
    with _patch_session(session):
        await auth.get_token()
    save.assert_called_once()
    assert "nrk-session=rotated" in save.call_args.args[0]
    assert "nrk-login-context-info=xyz" in save.call_args.args[0]


async def test_unchanged_cookies_are_not_saved_again() -> None:
    """A refresh that keeps the same cookies writes nothing."""
    save = Mock()
    with _patch_session(FakeSession(_routes())):
        await _auth(save=save).get_token()
    save.assert_not_called()


async def test_refused_login_is_retried_after_an_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused login is not hammered, but is tried again later instead of never."""
    now = [1000.0]
    monkeypatch.setattr(auth_module, "monotonic", lambda: now[0])
    auth = _auth()
    session = FakeSession(_routes(FakeResponse(status=401)))
    with _patch_session(session):
        with pytest.raises(LoginFailed):
            await auth.get_token()
        calls = len(session.calls)
        with pytest.raises(LoginFailed):
            await auth.get_token()
        assert len(session.calls) == calls
        now[0] += 3601
        session.routes["POST /auth/session/tokenforsub/_"] = FakeResponse(_token())
        assert await auth.get_token() == ("user-1", "token-1")


def _with_session(provider: NrkProvider) -> None:
    values = {CONF_SESSION: SESSION}
    provider.config.get_value.side_effect = (  # type: ignore[attr-defined]
        lambda key, default=None: values.get(key, default)
    )


def test_a_saved_session_counts_as_an_account(provider: NrkProvider) -> None:
    """With only a session saved, followed podcasts are synced to the library."""
    _with_session(provider)
    assert provider.has_account
    assert ProviderFeature.LIBRARY_PODCASTS in provider.supported_features


async def test_failed_login_still_follows_playback(
    provider: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A failed login no longer turns progress sync off until a restart.

    Setup subscribes anyway, so sync starts as soon as a later login works, and the log
    says plainly that NRK needs a new login.
    """
    _with_session(provider)
    subscribe = MagicMock()
    provider.mass.subscribe = subscribe  # type: ignore[method-assign]
    auth = MagicMock(
        enabled=True, get_token=AsyncMock(side_effect=NrkAuthError("NRK login needed: expired"))
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(provider_module, "NrkAuth", MagicMock(return_value=auth))
        await provider.handle_async_init()
    events = [c.args[1] for c in subscribe.call_args_list]
    assert events == [EventType.QUEUE_UPDATED, EventType.MEDIA_ITEM_PLAYED]
    assert "NRK login needed" in caplog.text


async def test_renewed_session_is_saved_encrypted(provider: NrkProvider) -> None:
    """The session NRK renews is stored as the provider's secure session value."""
    _with_session(provider)
    make_auth = MagicMock(return_value=MagicMock(enabled=False))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(provider_module, "NrkAuth", make_auth)
        await provider.handle_async_init()
    kwargs = make_auth.call_args.kwargs
    assert kwargs["session"] == SESSION
    kwargs["save_session"]("nrk-session=new")
    cast("MagicMock", provider.mass.config.set_raw_provider_config_value).assert_called_once_with(
        "nrk--test", CONF_SESSION, "nrk-session=new", encrypted=True, immediate=True
    )


async def test_progress_is_written_to_nrk_with_a_saved_session(provider: NrkProvider) -> None:
    """Playing the feed episode writes its progress to NRK on the session's token."""
    _with_session(provider)
    provider.mass.subscribe = MagicMock()  # type: ignore[method-assign]
    with _patch_session(FakeSession(_routes())):
        await provider.handle_async_init()
        provider._client.put_progress = AsyncMock()  # type: ignore[method-assign]
        _link_library(provider, _feed_episode())
        await provider._on_media_item_played(_played(600, is_playing=False))
    put = provider._client.put_progress
    put.assert_awaited_once()
    assert put.await_args is not None
    assert put.await_args.args[:5] == ("user-1", "token-1", "baade", "l_abc", 600)


def _last_errors(provider: NrkProvider) -> list[str | None]:
    calls = provider.mass.config.update_provider_last_error.call_args_list  # type: ignore[attr-defined]
    assert all(c.args[0] == "nrk--test" for c in calls)
    return [c.args[1].message if c.args[1] else None for c in calls]


async def _start(provider: NrkProvider, token: FakeResponse | None = None) -> None:
    _with_session(provider)
    provider.mass.subscribe = MagicMock()  # type: ignore[method-assign]
    with _patch_session(FakeSession(_routes(token))):
        await provider.handle_async_init()


@pytest.mark.parametrize(
    ("token", "said"),
    [
        (FakeResponse(status=401), "NRK login needed"),
        (FakeResponse(status=503), "Could not reach NRK"),
    ],
)
async def test_a_failed_login_is_shown_as_the_provider_error(
    provider: NrkProvider, token: FakeResponse, said: str
) -> None:
    """The login state is shown as the provider's last error."""
    await _start(provider, token)
    errors = _last_errors(provider)
    assert errors
    assert errors[-1] is not None
    assert errors[-1].startswith(said)


async def test_the_error_survives_the_end_of_loading(provider: NrkProvider) -> None:
    """MA clears last_error once a provider loads, so the provider writes it again after."""
    await _start(provider, FakeResponse(status=401))
    provider.mass.config.update_provider_last_error.reset_mock()  # type: ignore[attr-defined]
    await provider.loaded_in_mass()
    assert _last_errors(provider)[-1].startswith("NRK login needed")  # type: ignore[union-attr]


async def test_the_error_survives_mas_eager_post_load_step(provider: NrkProvider) -> None:
    """MA starts loaded_in_mass eagerly and clears last_error right after, so it must yield first."""
    await _start(provider, FakeResponse(status=401))
    clear = provider.mass.config.update_provider_last_error
    task = asyncio.Task(provider.loaded_in_mass(), eager_start=True)
    clear(provider.instance_id, None)
    await task
    assert _last_errors(provider)[-1].startswith("NRK login needed")  # type: ignore[union-attr]


async def test_a_working_login_clears_the_error(provider: NrkProvider) -> None:
    """Once NRK lets the login through, the error is gone."""
    await _start(provider, FakeResponse(status=503))
    with _patch_session(FakeSession(_routes())):
        await provider._auth.get_token()
    assert _last_errors(provider)[-1] is None


async def test_without_an_account_there_is_no_error(provider: NrkProvider) -> None:
    """Radio and podcasts need no NRK login, so none set up is not an error."""
    provider.mass.subscribe = MagicMock()  # type: ignore[method-assign]
    await provider.handle_async_init()
    await provider.loaded_in_mass()
    assert _last_errors(provider) == []
