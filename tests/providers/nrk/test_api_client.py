"""Tests for the NRK API client: paging, error mapping and the progress requests."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from music_assistant_models.errors import RetriesExhausted

from music_assistant.providers.nrk.api_client import NrkClient
from music_assistant.providers.nrk.models import NrkNotFoundError, NrkRequestError

from .conftest import FakeResponse, FakeSession, load_fixture

PODCAST_ID = "baade_erlend_og_steinar_"


def _client(routes: dict[str, Any]) -> tuple[NrkClient, FakeSession]:
    session = FakeSession(routes)
    return NrkClient(session, logging.getLogger("test")), session  # type: ignore[arg-type]


def _last_page(page: dict[str, Any]) -> dict[str, Any]:
    return {"_links": {}, "_embedded": page["_embedded"]}


async def test_iter_episodes_follows_pages() -> None:
    """Every page is fetched, 50 at a time, until NRK stops sending a next link."""
    page = load_fixture("episodes.json")
    client, session = _client(
        {
            f"GET /radio/catalog/podcast/{PODCAST_ID}/episodes": [
                FakeResponse(page),
                FakeResponse(_last_page(page)),
            ]
        }
    )
    episodes = [ep async for ep in client.iter_episodes(PODCAST_ID)]
    assert len(episodes) == 6
    assert [call["params"]["page"] for call in session.calls] == [1, 2]
    assert all(call["params"]["pageSize"] == 50 for call in session.calls)


async def test_public_requests_send_the_api_version_and_no_token() -> None:
    """Catalog calls are anonymous and ask for the API version the parsers expect."""
    client, session = _client({"GET /radio/live": FakeResponse(load_fixture("live.json"))})
    await client.get_channels()
    headers = session.calls[0]["headers"]
    assert headers["Accept"] == "application/json;api-version=3.5"
    assert "Authorization" not in headers


async def test_not_found_maps_to_media_not_found() -> None:
    """A 404 is a missing item, not a temporary failure."""
    client, _ = _client({"GET /radio/catalog/podcast/nope": FakeResponse(status=404)})
    with pytest.raises(NrkNotFoundError):
        await client.get_podcast("nope")


async def test_client_error_is_not_retried() -> None:
    """A 4xx refusal is raised at once instead of being retried."""
    client, session = _client({"GET /radio/catalog/podcast/x": FakeResponse(status=400)})
    with pytest.raises(NrkRequestError):
        await client.get_podcast("x")
    assert len(session.calls) == 1


async def test_rate_limit_backs_off_and_retries() -> None:
    """HTTP 429 is retried after a back-off, then succeeds."""
    client, session = _client(
        {
            "GET /radio/catalog/podcast/x": [
                FakeResponse(status=429, headers={"Retry-After": "1"}),
                FakeResponse(load_fixture("podcast.json")),
            ]
        }
    )
    with patch("music_assistant.helpers.throttle_retry.asyncio.sleep") as sleep:
        podcast = await client.get_podcast("x")
    assert podcast.id == PODCAST_ID
    assert len(session.calls) == 2
    assert sleep.await_count == 1
    assert sleep.await_args is not None
    assert sleep.await_args.args[0] >= 1


async def test_server_errors_exhaust_retries() -> None:
    """A persistent 5xx gives up after the retry budget instead of looping."""
    client, session = _client({"GET /radio/live": FakeResponse(status=503)})
    with (
        patch("music_assistant.helpers.throttle_retry.asyncio.sleep"),
        pytest.raises(RetriesExhausted),
    ):
        await client.get_channels()
    assert len(session.calls) == 3


async def test_followed_podcasts_follow_the_paging_key() -> None:
    """The favourites next link (with its paging key) is requested as-is."""
    first = load_fixture("favourites.json")
    second = {"_links": {}, "favourites": first["favourites"][:1]}
    client, session = _client({"GET /favourites": [FakeResponse(first), FakeResponse(second)]})
    ids = [pid async for pid in client.iter_followed_podcast_ids("user-1", "token-1")]
    assert ids == ["abels_taarn", PODCAST_ID, "abels_taarn"]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer token-1"
    assert session.calls[0]["url"].endswith("/radio/userdata/user-1/favourites")
    assert "key=" in session.calls[1]["url"]
    assert session.calls[1]["params"] is None


async def test_progress_uses_nrk_content_id() -> None:
    """Progress is read and written at NRK's `podcastepisode/{series}|{episode}` path."""
    client, session = _client(
        {
            "GET |ep-1": FakeResponse(load_fixture("progress_episode.json")),
            "PUT |ep-1": FakeResponse(status=204),
        }
    )
    progress = await client.get_progress("user-1", "token-1", PODCAST_ID, "ep-1")
    assert progress.position == 1392
    await client.put_progress("user-1", "token-1", PODCAST_ID, "ep-1", 1450, 1392)
    expected = f"/radio/userdata/user-1/progress/podcastepisode/{PODCAST_ID}|ep-1"
    assert all(call["url"].endswith(expected) for call in session.calls)
    assert session.calls[1]["method"] == "PUT"
    assert session.calls[1]["json"] == {"position": "PT1450S", "startPlaybackPosition": "PT1392S"}
    assert session.calls[1]["headers"]["Authorization"] == "Bearer token-1"


async def test_podcast_progress_asks_50_episodes_at_a_time() -> None:
    """NRK takes at most 50 episode ids per batch lookup."""
    client, session = _client(
        {
            f"GET /progress/podcast/{PODCAST_ID}": [
                FakeResponse(load_fixture("progress_podcast.json")),
                FakeResponse([]),
            ]
        }
    )
    ids = [f"l_{n}" for n in range(60)]
    progress = await client.get_podcast_progress("user-1", "token-1", PODCAST_ID, ids)
    assert len(progress) == 2
    assert [call["params"]["contentIds"].count(",") + 1 for call in session.calls] == [50, 10]
    assert session.calls[0]["url"].endswith(f"/radio/userdata/user-1/progress/podcast/{PODCAST_ID}")


async def test_delete_progress_follows_nrk_link() -> None:
    """Marking unplayed deletes through NRK's own link, which answers 202 with no body."""
    client, session = _client({"DELETE /progress/abc": FakeResponse(status=202)})
    await client.delete_progress("token-1", "/radio/userdata/user-1/progress/abc")
    assert session.calls[0]["url"] == "https://psapi.nrk.no/radio/userdata/user-1/progress/abc"
