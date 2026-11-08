"""Tests for writing listening progress to NRK: throttling, finishing and RSS-fed episodes."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import EventType, MediaType
from music_assistant_models.media_items import (
    ItemMapping,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
)

from music_assistant.providers.nrk import provider as provider_module
from music_assistant.providers.nrk.models import NrkApiError
from music_assistant.providers.nrk.provider import NrkProvider

from .conftest import set_credentials

EPISODE_ID = "baade|ep-1"
FEED = "podcastfeed--feed"
FEED_EPISODE_URI = f"{FEED}://podcast_episode/l_abc"


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
    """Return a provider with an account and a recording progress write."""
    set_credentials(provider)
    provider._client.put_progress = AsyncMock()  # type: ignore[method-assign]
    return provider


def _put(provider: NrkProvider) -> AsyncMock:
    return cast("AsyncMock", provider._client.put_progress)


def _writes(provider: NrkProvider) -> list[tuple[Any, ...]]:
    """Return (item, position, start) of every write."""
    return [(f"{c.args[2]}|{c.args[3]}", *c.args[4:]) for c in _put(provider).await_args_list]


async def _report(
    provider: NrkProvider, position: int, is_playing: bool, duration: int = 3600
) -> None:
    item: Any = SimpleNamespace(duration=duration)
    await provider.on_played(
        MediaType.PODCAST_EPISODE, EPISODE_ID, False, position, item, is_playing
    )


def _feed_episode(item_id: str = "l_abc", name: str = "Teslaen innen aktiv dødshjelp") -> Any:
    """Return an episode as the RSS feed provider serves it."""
    return PodcastEpisode(
        item_id=item_id,
        provider=FEED,
        name=name,
        position=1,
        duration=3844,
        podcast=ItemMapping(
            media_type=MediaType.PODCAST, item_id="feed-1", provider=FEED, name="Baade"
        ),
        provider_mappings={
            ProviderMapping(item_id=item_id, provider_domain="podcastfeed", provider_instance=FEED)
        },
    )


def _library_podcast(with_nrk: bool = True) -> Podcast:
    mappings = {
        ProviderMapping(item_id="feed-1", provider_domain="podcastfeed", provider_instance=FEED)
    }
    if with_nrk:
        mappings.add(
            ProviderMapping(item_id="baade", provider_domain="nrk", provider_instance="nrk--test")
        )
    return Podcast(item_id="1", provider="library", name="Baade", provider_mappings=mappings)


def _link_library(provider: NrkProvider, episode: Any, with_nrk: bool = True) -> None:
    """Make MA resolve the feed episode and find its podcast in the library."""
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=episode)  # type: ignore[method-assign]
    provider.mass.music.get_library_item_by_prov_id = AsyncMock(  # type: ignore[method-assign]
        return_value=_library_podcast(with_nrk)
    )
    nrk_episodes = [
        SimpleNamespace(item_id="baade|l_other", name="Traumepåføring hos barn", duration=3940),
        SimpleNamespace(item_id="baade|l_abc", name="Teslaen innen aktiv dødshjelp", duration=3844),
    ]
    provider._fetch_episodes = AsyncMock(return_value=nrk_episodes)  # type: ignore[method-assign]


def _played(
    seconds: int, is_playing: bool = True, fully_played: bool = False, uri: str = FEED_EPISODE_URI
) -> Any:
    report = SimpleNamespace(
        uri=uri,
        media_type=MediaType.PODCAST_EPISODE,
        duration=3844,
        seconds_played=seconds,
        fully_played=fully_played,
        is_playing=is_playing,
    )
    return SimpleNamespace(data=report)


async def test_writes_at_most_every_30_seconds_while_playing(
    nrk: NrkProvider, clock: Clock
) -> None:
    """Reports closer together than MA's 30 s interval are not written twice."""
    await _report(nrk, 100, is_playing=True)
    clock.now = 10
    await _report(nrk, 110, is_playing=True)
    clock.now = 29
    await _report(nrk, 129, is_playing=True)
    assert [w[1] for w in _writes(nrk)] == [100, 129]


async def test_pause_and_stop_are_always_written(nrk: NrkProvider, clock: Clock) -> None:
    """A pause right after a write is still saved, so the phone resumes at the right place."""
    await _report(nrk, 100, is_playing=True)
    clock.now = 3
    await _report(nrk, 103, is_playing=False)
    assert [w[1] for w in _writes(nrk)] == [100, 103]


async def test_past_95_percent_marks_finished(nrk: NrkProvider) -> None:
    """At 95% the episode is written at its full length, which NRK shows as listened."""
    await _report(nrk, 3420, is_playing=True, duration=3600)
    assert _writes(nrk)[-1][1] == 3600


async def test_last_minute_marks_finished(nrk: NrkProvider) -> None:
    """Within 60 s of the end counts as finished even under 95% (long outros)."""
    await _report(nrk, 945, is_playing=False, duration=1000)
    assert _writes(nrk)[-1][1] == 1000


async def test_short_episode_is_not_finished_at_start(nrk: NrkProvider) -> None:
    """The 60 s rule does not finish an episode shorter than two minutes at once."""
    await _report(nrk, 20, is_playing=False, duration=90)
    assert _writes(nrk)[-1][1] == 20


async def test_every_write_is_logged_without_token(
    nrk: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """A saved write is logged at INFO, and the bearer token never shows up in the log."""
    with caplog.at_level(logging.INFO):
        await _report(nrk, 100, is_playing=True)
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("Saved NRK progress of baade|ep-1: 100s" in m for m in infos)
    assert "token-1" not in caplog.text


async def test_failed_write_is_a_warning(
    nrk: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """An NRK outage is logged as a warning on every write, and the next write retries."""
    _put(nrk).side_effect = NrkApiError("down")
    await _report(nrk, 100, is_playing=True)
    await _report(nrk, 110, is_playing=True)
    # a failed write does not count for throttling, so the retry is not held back
    assert _put(nrk).await_count == 2
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "Could not save NRK progress of baade|ep-1" in warnings[0]


async def test_rss_feed_episode_is_written_to_nrk(nrk: NrkProvider, clock: Clock) -> None:
    """
    The library podcast plays from its RSS feed, so MA only reports to the feed provider.

    The report is linked to the NRK episode with the same id and written to NRK, with
    NRK's own length deciding when it is finished.
    """
    _link_library(nrk, _feed_episode())
    await nrk._on_media_item_played(_played(364))
    clock.now = 30
    await nrk._on_media_item_played(_played(394))
    clock.now = 40
    await nrk._on_media_item_played(_played(404, is_playing=False))
    assert _writes(nrk) == [
        ("baade|l_abc", 364, 364),
        ("baade|l_abc", 394, 364),
        ("baade|l_abc", 404, 364),
    ]
    # linked once, not resolved again on every report
    assert cast("AsyncMock", nrk.mass.music.get_item_by_uri).await_count == 1
    clock.now = 100
    await nrk._on_media_item_played(_played(3800))
    assert _writes(nrk)[-1] == ("baade|l_abc", 3844, 3800)


async def test_rss_feed_episode_matched_by_title(nrk: NrkProvider) -> None:
    """A feed with its own guids is matched to NRK's episode by title."""
    _link_library(nrk, _feed_episode(item_id="guid-7", name="traumepåføring  hos barn"))
    await nrk._on_media_item_played(_played(600, is_playing=False))
    assert _writes(nrk) == [("baade|l_other", 600, 600)]


async def test_podcast_without_nrk_mapping_is_left_alone(nrk: NrkProvider) -> None:
    """Other podcasts are not written to NRK, and are only looked up once."""
    _link_library(nrk, _feed_episode(), with_nrk=False)
    await nrk._on_media_item_played(_played(600))
    await nrk._on_media_item_played(_played(630))
    _put(nrk).assert_not_awaited()
    assert cast("AsyncMock", nrk.mass.music.get_item_by_uri).await_count == 1


async def test_nrk_own_reports_go_through_on_played_only(nrk: NrkProvider) -> None:
    """NRK's own episodes are written by on_played, so the playback report is skipped."""
    get_item = AsyncMock()
    nrk.mass.music.get_item_by_uri = get_item  # type: ignore[method-assign]
    await nrk._on_media_item_played(_played(600, uri="nrk--test://podcast_episode/baade|ep-1"))
    get_item.assert_not_awaited()
    _put(nrk).assert_not_awaited()


async def test_login_subscribes_to_playback_reports(provider: NrkProvider) -> None:
    """With an account, setup follows both queue updates and playback reports."""
    set_credentials(provider)
    subscribe = MagicMock()
    provider.mass.subscribe = subscribe  # type: ignore[method-assign]
    auth = MagicMock(enabled=True, get_token=AsyncMock(return_value=("user-1", "token-1")))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(provider_module, "NrkAuth", MagicMock(return_value=auth))
        await provider.handle_async_init()
    events = [c.args[1] for c in subscribe.call_args_list]
    assert events == [EventType.QUEUE_UPDATED, EventType.MEDIA_ITEM_PLAYED]


async def test_position_past_the_end_is_not_written(
    nrk: NrkProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A report far past the end of the episode is bogus, so it does not finish it in NRK.

    Restarting Music Assistant mid-episode reported 4257 s of a 2606 s episode as played.
    """
    _link_library(nrk, _feed_episode())
    await nrk._on_media_item_played(_played(4257, is_playing=False, fully_played=True))
    _put(nrk).assert_not_awaited()
    assert "past the end" in caplog.text
