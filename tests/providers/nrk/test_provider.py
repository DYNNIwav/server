"""Tests for the NRK Radio provider glue: browse, search, streams, library and resume."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import MediaNotFoundError, UnplayableMediaError

from music_assistant.providers.nrk.models import (
    NrkApiError,
    NrkAuthError,
    NrkChannel,
    NrkEpisode,
    NrkNotFoundError,
    NrkPodcast,
    NrkProgress,
    NrkRequestError,
    NrkStream,
)
from music_assistant.providers.nrk.provider import NrkProvider

from .conftest import async_gen, set_credentials

EPISODE_ID = "baade|ep-1"
CHANNELS = [NrkChannel(id="p1", name="NRK P1"), NrkChannel(id="p2", name="NRK P2")]
PODCAST = NrkPodcast(id="baade", title="Baade Erlend og Steinar", total_episodes=2)


def _episode(episode_id: str, day: int) -> NrkEpisode:
    return NrkEpisode(
        podcast_id="baade",
        episode_id=episode_id,
        title=episode_id,
        duration=600,
        published=datetime(2026, 9, day, tzinfo=UTC),
    )


def test_features_without_account(provider: NrkProvider) -> None:
    """Without an account there is no library to sync (and none to prune)."""
    assert ProviderFeature.LIBRARY_PODCASTS not in provider.supported_features
    assert ProviderFeature.SEARCH in provider.supported_features


def test_features_with_account(provider: NrkProvider) -> None:
    """With an account the followed podcasts are synced to the library."""
    set_credentials(provider)
    assert ProviderFeature.LIBRARY_PODCASTS in provider.supported_features


async def test_browse_root(provider: NrkProvider) -> None:
    """The root lists live radio and podcasts, plus followed podcasts with an account."""
    assert [f.item_id for f in await provider.browse("nrk--test://")] == ["radio", "podcasts"]
    set_credentials(provider)
    root = await provider.browse("nrk--test://")
    assert [f.item_id for f in root] == ["radio", "podcasts", "followed"]


async def test_browse_radio(provider: NrkProvider) -> None:
    """The radio folder lists every live channel."""
    provider._client.get_channels = AsyncMock(return_value=CHANNELS)  # type: ignore[method-assign]
    items = await provider.browse("nrk--test://radio")
    assert [item.name for item in items] == ["NRK P1", "NRK P2"]


async def test_search(provider: NrkProvider) -> None:
    """Channels match by name; podcasts come from NRK search."""
    provider._client.get_channels = AsyncMock(return_value=CHANNELS)  # type: ignore[method-assign]
    provider._client.search = AsyncMock(return_value=[PODCAST])  # type: ignore[method-assign]
    results = await provider.search("p2", [MediaType.RADIO, MediaType.PODCAST])
    assert [radio.item_id for radio in results.radio] == ["p2"]
    assert [podcast.item_id for podcast in results.podcasts] == ["baade"]


async def test_search_survives_nrk_outage(provider: NrkProvider) -> None:
    """A failing NRK search returns no podcasts instead of breaking MA's global search."""
    provider._client.search = AsyncMock(side_effect=NrkApiError("down"))  # type: ignore[method-assign]
    results = await provider.search("x", [MediaType.PODCAST])
    assert results.podcasts == []


async def test_get_radio_not_found(provider: NrkProvider) -> None:
    """An unknown channel id is a missing item."""
    provider._client.get_channels = AsyncMock(return_value=CHANNELS)  # type: ignore[method-assign]
    with pytest.raises(MediaNotFoundError):
        await provider.get_radio("p13")


async def test_episodes_are_numbered_oldest_first(provider: NrkProvider) -> None:
    """NRK lists newest first; MA numbers episodes from the oldest."""
    provider._client.get_podcast = AsyncMock(return_value=PODCAST)  # type: ignore[method-assign]
    provider._client.iter_episodes = async_gen([_episode("new", 21), _episode("old", 14)])  # type: ignore[method-assign]
    episodes = [ep async for ep in provider.get_podcast_episodes("baade")]
    assert [(ep.item_id, ep.position) for ep in episodes] == [("baade|new", 2), ("baade|old", 1)]
    assert episodes[0].podcast.name == "Baade Erlend og Steinar"


async def test_rejected_login_keeps_radio_and_podcasts(provider: NrkProvider) -> None:
    """A login NRK refuses leaves the provider loaded, so podcasts still play."""
    set_credentials(provider)
    auth = MagicMock(enabled=True)
    auth.get_token = AsyncMock(side_effect=NrkAuthError("NRK asks for a captcha"))
    with patch("music_assistant.providers.nrk.provider.NrkAuth", return_value=auth):
        await provider.handle_async_init()
    provider._client = MagicMock()
    provider._client.get_podcast = AsyncMock(return_value=PODCAST)
    provider._client.iter_episodes = async_gen([_episode("new", 21)])
    episodes = [ep async for ep in provider.get_podcast_episodes("baade")]
    assert [ep.item_id for ep in episodes] == ["baade|new"]


async def test_bad_episode_id(provider: NrkProvider) -> None:
    """An episode id without its podcast part is a missing item."""
    with pytest.raises(MediaNotFoundError):
        await provider.get_podcast_episode("ep-1")


async def test_radio_stream_is_live_hls(provider: NrkProvider) -> None:
    """Live channels play as HLS and cannot be seeked."""
    stream = NrkStream(url="https://nrk/p1.m3u8", content_type="aac", is_live=True)
    provider._client.get_channel_stream = AsyncMock(return_value=stream)  # type: ignore[method-assign]
    details = await provider.get_stream_details("p1", MediaType.RADIO)
    assert details.stream_type == StreamType.HLS
    assert not details.can_seek
    assert details.duration is None


async def test_episode_stream_is_seekable_http(provider: NrkProvider) -> None:
    """Episodes play as a direct MP3 that MA can seek in."""
    stream = NrkStream(url="https://nrk/ep.mp3", content_type="mp3", is_live=False, duration=600)
    provider._client.get_episode_stream = AsyncMock(return_value=stream)  # type: ignore[method-assign]
    details = await provider.get_stream_details(EPISODE_ID, MediaType.PODCAST_EPISODE)
    provider._client.get_episode_stream.assert_awaited_once_with("ep-1")
    assert details.stream_type == StreamType.HTTP
    assert details.path == "https://nrk/ep.mp3"
    assert details.can_seek
    assert details.duration == 600


async def test_unplayable_stream(provider: NrkProvider) -> None:
    """NRK saying "not playable" (e.g. expired rights) is surfaced as unplayable."""
    provider._client.get_episode_stream = AsyncMock(return_value=None)  # type: ignore[method-assign]
    with pytest.raises(UnplayableMediaError):
        await provider.get_stream_details(EPISODE_ID, MediaType.PODCAST_EPISODE)


async def test_library_sync(provider: NrkProvider) -> None:
    """Followed podcasts are synced; one NRK has removed is skipped."""
    set_credentials(provider)
    provider._client.iter_followed_podcast_ids = async_gen(["baade", "gone"])  # type: ignore[method-assign]
    provider._client.get_podcast = AsyncMock(  # type: ignore[method-assign]
        side_effect=[PODCAST, NrkNotFoundError("gone")]
    )
    podcasts = [podcast async for podcast in provider.get_library_podcasts()]
    assert [podcast.item_id for podcast in podcasts] == ["baade"]


async def test_library_sync_failure_aborts(provider: NrkProvider) -> None:
    """A failing sync raises, so MA never prunes the library on an empty result."""
    set_credentials(provider)
    provider._auth.get_token = AsyncMock(side_effect=NrkApiError("down"))  # type: ignore[method-assign]
    with pytest.raises(NrkApiError):
        _ = [podcast async for podcast in provider.get_library_podcasts()]


async def test_resume_position_from_nrk(provider: NrkProvider) -> None:
    """NRK progress is returned in milliseconds."""
    set_credentials(provider)
    provider._client.get_progress = AsyncMock(  # type: ignore[method-assign]
        return_value=NrkProgress(fully_played=False, position=90)
    )
    assert await provider.get_resume_position(EPISODE_ID, MediaType.PODCAST_EPISODE) == (
        False,
        90_000,
        None,
    )
    provider._client.get_progress.assert_awaited_once_with("user-1", "token-1", "baade", "ep-1")


async def test_resume_falls_back_and_warns_once(
    provider: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """When NRK progress fails, MA's own resume takes over with a single warning."""
    set_credentials(provider)
    provider._client.get_progress = AsyncMock(side_effect=NrkRequestError("400"))  # type: ignore[method-assign]
    with caplog.at_level(logging.DEBUG):
        for _ in range(2):
            with pytest.raises(NotImplementedError):
                await provider.get_resume_position(EPISODE_ID, MediaType.PODCAST_EPISODE)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "own resume position" in warnings[0].getMessage()


async def test_resume_without_stored_progress_is_quiet(
    provider: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """No progress yet (404) is the normal case for a new episode and is not warned about."""
    set_credentials(provider)
    provider._client.get_progress = AsyncMock(side_effect=NrkNotFoundError("none"))  # type: ignore[method-assign]
    with pytest.raises(NotImplementedError):
        await provider.get_resume_position(EPISODE_ID, MediaType.PODCAST_EPISODE)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_resume_without_account(provider: NrkProvider) -> None:
    """Without an account MA always uses its own resume position."""
    with pytest.raises(NotImplementedError):
        await provider.get_resume_position(EPISODE_ID, MediaType.PODCAST_EPISODE)


async def test_episodes_carry_nrk_progress(provider: NrkProvider) -> None:
    """Listed episodes show NRK's resume point and played mark; others use MA's play log."""
    set_credentials(provider)
    provider._client.get_podcast = AsyncMock(return_value=PODCAST)  # type: ignore[method-assign]
    provider._client.iter_episodes = async_gen(  # type: ignore[method-assign]
        [_episode("mid", 21), _episode("done", 14), _episode("new", 7)]
    )
    provider._client.get_podcast_progress = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "mid": NrkProgress(fully_played=False, position=90),
            "done": NrkProgress(fully_played=True, position=0),
            "new": NrkProgress(fully_played=False, position=0),
        }
    )
    episodes = {ep.item_id: ep async for ep in provider.get_podcast_episodes("baade")}
    provider._client.get_podcast_progress.assert_awaited_once_with(
        "user-1", "token-1", "baade", ["mid", "done", "new"]
    )
    assert (episodes["baade|mid"].fully_played, episodes["baade|mid"].resume_position_ms) == (
        False,
        90_000,
    )
    assert episodes["baade|done"].fully_played is True
    assert episodes["baade|new"].fully_played is None


async def test_episodes_survive_progress_outage(provider: NrkProvider) -> None:
    """When NRK progress fails the episodes still list, without NRK progress."""
    set_credentials(provider)
    provider._client.get_podcast = AsyncMock(return_value=PODCAST)  # type: ignore[method-assign]
    provider._client.iter_episodes = async_gen([_episode("mid", 21)])  # type: ignore[method-assign]
    provider._client.get_podcast_progress = AsyncMock(side_effect=NrkApiError("down"))  # type: ignore[method-assign]
    episodes = [ep async for ep in provider.get_podcast_episodes("baade")]
    assert episodes[0].fully_played is None


async def test_single_episode_carries_nrk_progress(provider: NrkProvider) -> None:
    """A single episode shows NRK's resume point too."""
    set_credentials(provider)
    provider._client.get_podcast = AsyncMock(return_value=PODCAST)  # type: ignore[method-assign]
    provider._client.get_episode = AsyncMock(return_value=_episode("ep-1", 21))  # type: ignore[method-assign]
    provider._client.get_progress = AsyncMock(  # type: ignore[method-assign]
        return_value=NrkProgress(fully_played=False, position=90)
    )
    episode = await provider.get_podcast_episode(EPISODE_ID)
    assert episode.resume_position_ms == 90_000


async def test_on_played_writes_progress(provider: NrkProvider) -> None:
    """Progress is written with the position playback started from."""
    set_credentials(provider)
    provider._client.get_progress = AsyncMock(  # type: ignore[method-assign]
        return_value=NrkProgress(fully_played=False, position=60)
    )
    provider._client.put_progress = AsyncMock()  # type: ignore[method-assign]
    await provider.get_resume_position(EPISODE_ID, MediaType.PODCAST_EPISODE)
    await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, False, 90, MagicMock(), True)
    await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, False, 120, MagicMock())
    await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, False, 150, MagicMock())
    calls = [c.args for c in provider._client.put_progress.await_args_list]
    assert calls == [
        ("user-1", "token-1", "baade", "ep-1", 90, 60),
        ("user-1", "token-1", "baade", "ep-1", 120, 60),
        # stopped at 120, so the next playback starts where it reports first
        ("user-1", "token-1", "baade", "ep-1", 150, 150),
    ]


async def test_on_played_finished_writes_full_length(provider: NrkProvider) -> None:
    """A finished episode is stored at its full length, which NRK marks as finished."""
    set_credentials(provider)
    provider._client.put_progress = AsyncMock()  # type: ignore[method-assign]
    item = MagicMock(duration=600)
    await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, True, 590, item)
    provider._client.put_progress.assert_awaited_once_with(
        "user-1", "token-1", "baade", "ep-1", 600, 590
    )


async def test_on_played_unplayed_deletes_nrk_progress(provider: NrkProvider) -> None:
    """Marking unplayed in MA deletes the progress through NRK's link."""
    set_credentials(provider)
    provider._client.get_progress = AsyncMock(  # type: ignore[method-assign]
        return_value=NrkProgress(fully_played=True, position=0, delete_path="/del/1")
    )
    provider._client.delete_progress = AsyncMock()  # type: ignore[method-assign]
    provider._client.put_progress = AsyncMock()  # type: ignore[method-assign]
    await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, False, 0, MagicMock())
    provider._client.delete_progress.assert_awaited_once_with("token-1", "/del/1")
    provider._client.put_progress.assert_not_awaited()


async def test_on_played_warns_on_every_refusal(
    provider: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """A refused write is logged as a warning every time, never dropped silently."""
    set_credentials(provider)
    provider._client.put_progress = AsyncMock(side_effect=NrkRequestError("400"))  # type: ignore[method-assign]
    for _ in range(3):
        await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, False, 90, MagicMock())
    assert provider._client.put_progress.await_count == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3
    assert "baade|ep-1" in warnings[0].getMessage()


async def test_on_played_keeps_trying_after_outage(provider: NrkProvider) -> None:
    """A temporary failure does not switch progress sync off."""
    set_credentials(provider)
    provider._client.put_progress = AsyncMock(side_effect=NrkApiError("down"))  # type: ignore[method-assign]
    for _ in range(2):
        await provider.on_played(MediaType.PODCAST_EPISODE, EPISODE_ID, False, 90, MagicMock())
    assert provider._client.put_progress.await_count == 2
