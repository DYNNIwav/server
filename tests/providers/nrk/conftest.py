"""Shared fixtures for NRK Radio provider tests."""

from __future__ import annotations

import json
import pathlib
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import Any, Self, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import pytest
from music_assistant_models.enums import ProviderType
from yarl import URL

from music_assistant.constants import CONF_PASSWORD, CONF_USERNAME
from music_assistant.providers.nrk.provider import NrkProvider
from tests.common import use_real_create_task

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_T = TypeVar("_T")


def load_fixture(name: str) -> Any:
    """
    Return a recorded NRK response from the fixtures folder.

    :param name: The fixture file name.
    """
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def provider() -> NrkProvider:
    """Return an NrkProvider with a mocked client/auth and a neutralized cache."""
    mass = MagicMock()
    # Force every @use_cache lookup to miss so the real method body always runs.
    mass.cache.get_with_freshness = AsyncMock(return_value=(None, False, False))
    mass.cache.set = AsyncMock()
    use_real_create_task(mass)
    manifest = Mock()
    manifest.domain = "nrk"
    manifest.type = ProviderType.MUSIC
    config = Mock()
    config.instance_id = "nrk--test"
    config.name = "NRK Radio"
    config.enabled = True
    config.get_value.side_effect = lambda key, default=None: {}.get(key, default)
    prov = NrkProvider(mass, manifest, config)
    prov._client = MagicMock()
    prov._auth = Mock()
    prov._auth.enabled = False
    return prov


def set_credentials(provider: NrkProvider) -> None:
    """Configure NRK account credentials on a test provider."""
    values = {CONF_USERNAME: "user@example.com", CONF_PASSWORD: "secret"}
    provider.config.get_value.side_effect = (  # type: ignore[attr-defined]
        lambda key, default=None: values.get(key, default)
    )
    provider._auth.enabled = True  # type: ignore[misc]
    provider._auth.get_token = AsyncMock(return_value=("user-1", "token-1"))  # type: ignore[method-assign]


def async_gen(items: Iterable[_T]) -> Callable[..., AsyncGenerator[_T]]:
    """Return an async-generator function yielding `items`, ignoring call args."""

    async def _gen(*_args: Any, **_kwargs: Any) -> AsyncGenerator[_T]:
        for item in items:
            yield item

    return _gen


class FakeResponse:
    """Stand-in for an aiohttp response used as an async context manager."""

    def __init__(
        self,
        payload: Any = None,
        status: int = 200,
        headers: dict[str, str] | None = None,
        history: tuple[Any, ...] = (),
    ) -> None:
        """Initialize the response."""
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self.history = history

    async def json(self, **_kwargs: Any) -> Any:
        """Return the canned payload."""
        return self.payload

    def raise_for_status(self) -> None:
        """Mimic aiohttp for 5xx statuses."""
        if self.status >= 500:
            raise aiohttp.ClientResponseError(
                MagicMock(), (), status=self.status, message="server error"
            )

    async def __aenter__(self) -> Self:
        """Enter the context."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Leave the context."""


class FakeSession:
    """
    Session that answers from a route table and records every call.

    Routes map "METHOD url-suffix" to a response, or to a list of responses served in turn.
    """

    def __init__(self, routes: dict[str, FakeResponse | list[FakeResponse]]) -> None:
        """Initialize the session."""
        self.routes = routes
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        """Answer a request from the route table."""
        self.calls.append({"method": method, "url": url, **kwargs})
        for route, response in self.routes.items():
            route_method, suffix = route.split(" ", 1)
            if route_method == method and str(url).split("?")[0].endswith(suffix):
                if isinstance(response, list):
                    return response.pop(0) if len(response) > 1 else response[0]
                return response
        raise AssertionError(f"Unexpected request: {method} {url}")

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Answer a GET."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        """Answer a POST."""
        return self.request("POST", url, **kwargs)

    async def __aenter__(self) -> Self:
        """Enter the context."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Leave the context."""


def redirect_to(url: str) -> tuple[Any, ...]:
    """
    Return a response history ending at `url`, as aiohttp records a followed redirect.

    :param url: The URL the last redirect pointed at.
    """
    step = Mock()
    step.url = URL(url)
    return (step,)
