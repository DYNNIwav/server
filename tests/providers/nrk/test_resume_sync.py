"""Tests for matching NRK listening progress when playback starts, resumes and pauses."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import MediaType, PlaybackState

from music_assistant.providers.nrk import provider as provider_module
from music_assistant.providers.nrk.api_client import NrkClient
from music_assistant.providers.nrk.models import NrkProgress
from music_assistant.providers.nrk.provider import NrkProvider

from .conftest import FakeResponse, FakeSession, load_fixture, set_credentials
from .test_progress_sync import FEED, _feed_episode, _link_library, _played

EPISODE_ID = "baade|ep-1"
LATER = datetime.now(UTC) + timedelta(hours=1)
EARLIER = datetime(2020, 1, 1, tzinfo=UTC)


class Clock:
    """Monotonic clock the tests move by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Replace the provider's monotonic clock."""
    fake = Clock()
    monkeypatch.setattr(provider_module, "monotonic", fake)
    return fake


@pytest.fixture
def nrk(provider: NrkProvider) -> NrkProvider:
    """Return a provider with an account, a recording seek and a recording write."""
    set_credentials(provider)
    provider.mass.player_queues.seek = AsyncMock()  # type: ignore[method-assign]
    provider._client.put_progress = AsyncMock()  # type: ignore[method-assign]
    return provider


def _event(
    state: PlaybackState = PlaybackState.PLAYING,
    elapsed: float = 0,
    seek: int = 0,
    provider: str = "nrk--test",
    item_id: str = EPISODE_ID,
) -> Any:
    streamdetails = SimpleNamespace(
        provider=provider,
        media_type=MediaType.PODCAST_EPISODE,
        item_id=item_id,
        seek_position=seek,
    )
    queue = SimpleNamespace(
        queue_id="q1",
        state=state,
        elapsed_time=elapsed,
        current_item=SimpleNamespace(streamdetails=streamdetails),
    )
    return SimpleNamespace(data=queue)


def _nrk_progress(provider: NrkProvider, position: int, updated: datetime | None) -> None:
    provider._client.get_progress = AsyncMock(  # type: ignore[method-assign]
        return_value=NrkProgress(fully_played=False, position=position, updated=updated)
    )


async def _report(provider: NrkProvider, position: int, is_playing: bool) -> None:
    await provider.on_played(
        MediaType.PODCAST_EPISODE, EPISODE_ID, False, position, MagicMock(), is_playing
    )


def _seek(provider: NrkProvider) -> AsyncMock:
    return cast("AsyncMock", provider.mass.player_queues.seek)


def _put(provider: NrkProvider) -> AsyncMock:
    return cast("AsyncMock", provider._client.put_progress)


def _writes(provider: NrkProvider) -> list[tuple[int, int]]:
    return [c.args[4:] for c in _put(provider).await_args_list]


async def test_nrk_newer_than_ma_seeks_to_nrk(nrk: NrkProvider, clock: Clock) -> None:
    """Paused at home at 10:00, listened on the phone to 30:00: play continues at 30:00."""
    await _report(nrk, 600, is_playing=False)
    _nrk_progress(nrk, 1800, LATER)
    await nrk._on_queue_updated(_event(elapsed=600))
    _seek(nrk).assert_awaited_once_with("q1", 1800)
    # the next report continues from NRK's position, it does not write 10:00 back
    clock.now = 30
    await _report(nrk, 1830, is_playing=True)
    assert _writes(nrk)[-1] == (1830, 1800)


async def test_ma_newer_than_nrk_keeps_ma_position(nrk: NrkProvider) -> None:
    """NRK's progress is older than MA's last report, so MA's position stays."""
    await _report(nrk, 1800, is_playing=False)
    _nrk_progress(nrk, 600, EARLIER)
    await nrk._on_queue_updated(_event(elapsed=1800))
    _seek(nrk).assert_not_awaited()


async def test_agreeing_positions_resume_without_seek(nrk: NrkProvider) -> None:
    """MA already resumed where NRK is, so there is no reload and no jump."""
    _nrk_progress(nrk, 605, LATER)
    await nrk._on_queue_updated(_event(elapsed=600))
    _seek(nrk).assert_not_awaited()


async def test_nrk_holding_ma_own_write_is_not_newer(nrk: NrkProvider) -> None:
    """NRK's timestamp on MA's own write can run ahead of MA's clock; it is not news."""
    await _report(nrk, 2640, is_playing=False)
    _nrk_progress(nrk, 2640, LATER)
    await nrk._on_queue_updated(_event(elapsed=900))
    _seek(nrk).assert_not_awaited()


async def test_finished_or_missing_nrk_progress_never_seeks(nrk: NrkProvider) -> None:
    """Nothing to continue from on NRK, so MA's resume stands."""
    nrk._client.get_progress = AsyncMock(  # type: ignore[method-assign]
        return_value=NrkProgress(fully_played=True, position=0, updated=LATER)
    )
    await nrk._on_queue_updated(_event(elapsed=600))
    _seek(nrk).assert_not_awaited()


async def test_pause_reports_live_position_not_stream_time(nrk: NrkProvider, clock: Clock) -> None:
    """
    Resumed at 29:00, paused at 44:00, but MA reports 15:00 (the stream time only).

    The live 44:00 is written to NRK, and pressing play again resumes at once with no
    seek, even while MA still shows the stale 15:00.
    """
    _nrk_progress(nrk, 1740, EARLIER)
    await nrk._on_queue_updated(_event(elapsed=1740, seek=1740))
    _seek(nrk).assert_not_awaited()
    clock.now = 600
    await _report(nrk, 2340, is_playing=True)
    clock.now = 900
    await nrk._on_queue_updated(_event(state=PlaybackState.PAUSED, elapsed=900, seek=1740))
    await _report(nrk, 900, is_playing=False)
    assert _writes(nrk)[-1] == (2640, 1740)

    _nrk_progress(nrk, 2640, LATER)
    clock.now = 1200
    await nrk._on_queue_updated(_event(elapsed=900, seek=1740))
    _seek(nrk).assert_not_awaited()
    # a second pause later in the same stream is corrected the same way
    clock.now = 1260
    await nrk._on_queue_updated(_event(state=PlaybackState.PAUSED, elapsed=960, seek=1740))
    await _report(nrk, 960, is_playing=False)
    assert _writes(nrk)[-1] == (2700, 2640)


async def test_real_backward_seek_is_not_corrected(nrk: NrkProvider, clock: Clock) -> None:
    """A seek back to 20:00 reloads the stream from there, and 20:10 is reported as is."""
    _nrk_progress(nrk, 1740, EARLIER)
    await nrk._on_queue_updated(_event(elapsed=1740, seek=1740))
    clock.now = 900
    await nrk._on_queue_updated(_event(elapsed=1200, seek=1200))
    clock.now = 910
    await _report(nrk, 1210, is_playing=False)
    assert _writes(nrk)[-1] == (1210, 1210)


async def test_no_writes_while_matching_nrk(nrk: NrkProvider) -> None:
    """MA's stale position is not written while NRK's newer progress is being read."""
    release = asyncio.Event()

    async def slow_progress(*_args: Any) -> NrkProgress:
        await release.wait()
        return NrkProgress(fully_played=False, position=1800, updated=LATER)

    nrk._client.get_progress = slow_progress  # type: ignore[method-assign,assignment]
    task = asyncio.create_task(nrk._on_queue_updated(_event(elapsed=600)))
    await asyncio.sleep(0)
    await _report(nrk, 600, is_playing=True)
    release.set()
    await task
    _put(nrk).assert_not_awaited()
    _seek(nrk).assert_awaited_once_with("q1", 1800)


async def test_other_queue_updates_do_nothing(nrk: NrkProvider) -> None:
    """Updates while the same episode keeps playing, or for other podcasts, are ignored."""
    _link_library(nrk, _feed_episode(), with_nrk=False)
    _nrk_progress(nrk, 1800, LATER)
    await nrk._on_queue_updated(_event(elapsed=1800))
    await nrk._on_queue_updated(_event(elapsed=1830))
    other = _event(elapsed=0)
    other.data.current_item.streamdetails.provider = "spotify"
    await nrk._on_queue_updated(other)
    assert cast("AsyncMock", nrk._client.get_progress).await_count == 1


async def test_unload_stops_following_queues(nrk: NrkProvider) -> None:
    """The queue and playback subscriptions end with the provider."""
    unsubscribers = [Mock(), Mock()]
    nrk._unsubscribers = list(unsubscribers)
    await nrk.unload()
    for unsubscribe in unsubscribers:
        unsubscribe.assert_called_once_with()


async def test_both_directions_over_http(nrk: NrkProvider, clock: Clock) -> None:
    """NRK's stored 23:12 is read and seeked to, and progress from there is written back."""
    session = FakeSession(
        {
            "GET |ep-1": FakeResponse(load_fixture("progress_episode.json")),
            "PUT |ep-1": FakeResponse(status=204),
        }
    )
    nrk._client = NrkClient(session, logging.getLogger("test"))  # type: ignore[arg-type]
    await nrk._on_queue_updated(_event(elapsed=300))
    _seek(nrk).assert_awaited_once_with("q1", 1392)
    clock.now = 30
    await _report(nrk, 1422, is_playing=True)
    put = session.calls[-1]
    assert put["method"] == "PUT"
    assert put["url"].endswith("/radio/userdata/user-1/progress/podcastepisode/baade|ep-1")
    assert put["json"] == {"position": "PT1422S", "startPlaybackPosition": "PT1392S"}


async def test_rss_feed_episode_resumes_from_nrk(nrk: NrkProvider) -> None:
    """
    The library podcast streams from its RSS feed, and the phone got further on NRK.

    The feed episode is linked to its NRK episode, so playback continues from the phone.
    """
    _link_library(nrk, _feed_episode())
    _nrk_progress(nrk, 1800, LATER)
    await nrk._on_queue_updated(_event(elapsed=600, provider=FEED, item_id="l_abc"))
    get_progress = cast("AsyncMock", nrk._client.get_progress)
    get_progress.assert_awaited_once()
    assert get_progress.await_args is not None
    assert get_progress.await_args.args[2:] == ("baade", "l_abc")
    _seek(nrk).assert_awaited_once_with("q1", 1800)


async def test_rss_feed_episode_last_played_on_speaker_stays(nrk: NrkProvider) -> None:
    """NRK holds what the speaker wrote from the feed itself, so there is no jump."""
    _link_library(nrk, _feed_episode())
    await nrk._on_media_item_played(_played(1800, is_playing=False))
    _nrk_progress(nrk, 1800, LATER)
    await nrk._on_queue_updated(_event(elapsed=900, provider=FEED, item_id="l_abc"))
    _seek(nrk).assert_not_awaited()


async def test_rss_feed_report_held_back_while_matching(nrk: NrkProvider) -> None:
    """The stale position the feed resumed from does not overwrite NRK's newer one."""
    _link_library(nrk, _feed_episode())
    nrk._reconciling.add("baade|l_abc")
    await nrk._on_media_item_played(_played(600))
    _put(nrk).assert_not_awaited()


async def test_rss_podcast_without_nrk_mapping_never_seeks(nrk: NrkProvider) -> None:
    """Other feed podcasts are not matched against NRK at all."""
    _link_library(nrk, _feed_episode(), with_nrk=False)
    nrk._client.get_progress = AsyncMock()  # type: ignore[method-assign]
    await nrk._on_queue_updated(_event(elapsed=600, provider=FEED, item_id="l_abc"))
    nrk._client.get_progress.assert_not_awaited()
    _seek(nrk).assert_not_awaited()


async def test_rss_pause_reporting_stream_start_keeps_live_position(
    nrk: NrkProvider, clock: Clock
) -> None:
    """
    Resumed from the feed at 3:14 and paused at 6:00, but MA reports 3:16 on pause.

    That is where the stream started, not where it was paused, so the live 6:00 is
    written to NRK.
    """
    _link_library(nrk, _feed_episode())
    _nrk_progress(nrk, 194, EARLIER)
    await nrk._on_queue_updated(_event(elapsed=194, seek=194, provider=FEED, item_id="l_abc"))
    clock.now = 136
    await nrk._on_media_item_played(_played(330))
    clock.now = 166
    await nrk._on_media_item_played(_played(196, is_playing=False))
    assert _writes(nrk)[-1] == (360, 194)


async def test_rss_seek_back_near_stream_start_is_kept(nrk: NrkProvider, clock: Clock) -> None:
    """A seek back to 1:00 starts a new stream there, and pausing at 1:05 is kept."""
    _link_library(nrk, _feed_episode())
    _nrk_progress(nrk, 194, EARLIER)
    await nrk._on_queue_updated(_event(elapsed=194, seek=194, provider=FEED, item_id="l_abc"))
    clock.now = 136
    await nrk._on_media_item_played(_played(330))
    await nrk._on_queue_updated(_event(elapsed=60, seek=60, provider=FEED, item_id="l_abc"))
    clock.now = 141
    await nrk._on_media_item_played(_played(65, is_playing=False))
    assert _writes(nrk)[-1][0] == 65
