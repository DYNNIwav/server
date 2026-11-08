"""Tests for parsing recorded NRK responses."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from music_assistant.providers.nrk.models import NrkApiError
from music_assistant.providers.nrk.parsers import (
    format_duration,
    parse_channels,
    parse_curated_podcasts,
    parse_duration,
    parse_episode,
    parse_episodes,
    parse_favourites,
    parse_manifest,
    parse_podcast,
    parse_podcast_progress,
    parse_progress,
    parse_search,
)

from .conftest import load_fixture

PODCAST_ID = "baade_erlend_og_steinar_"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("PT1H6M40S", 4000), ("PT45S", 45), ("PT2M", 120), ("P1D", 86400), ("", 0), (None, 0)],
)
def test_parse_duration(value: str | None, expected: int) -> None:
    """ISO 8601 durations become whole seconds, anything unreadable becomes 0."""
    assert parse_duration(value) == expected


def test_format_duration_round_trips() -> None:
    """A position written back to NRK reads back as the same number of seconds."""
    assert parse_duration(format_duration(754)) == 754


def test_parse_channels() -> None:
    """Live channels carry their id, display name and the widest square poster."""
    channels = parse_channels(load_fixture("live.json"))
    assert [c.id for c in channels] == ["p1", "p1pluss", "p2"]
    assert channels[0].name == "NRK P1"
    assert channels[0].image_url is not None
    assert channels[0].image_url.startswith("https://gfx.nrk.no/")


def test_parse_channels_rejects_unexpected_shape() -> None:
    """A changed response raises rather than returning no channels."""
    with pytest.raises(NrkApiError):
        parse_channels({"unexpected": True})


def test_parse_search_folds_episodes_into_podcasts() -> None:
    """Episode hits come back as their podcast, without duplicating a podcast hit."""
    podcasts = parse_search(load_fixture("search.json"))
    ids = [p.id for p in podcasts]
    assert ids[0] == PODCAST_ID
    assert len(ids) == len(set(ids))
    assert podcasts[0].title == "Både Erlend og Steinar"


def test_parse_podcast() -> None:
    """A catalog podcast carries its title, subtitle and square image."""
    podcast = parse_podcast(load_fixture("podcast.json"))
    assert podcast.id == PODCAST_ID
    assert podcast.description == "En tenketank for folk med lav intelligens."
    assert podcast.image_url is not None


def test_parse_curated_podcasts_dedupes() -> None:
    """A podcast featured in two sections is listed once."""
    podcasts = parse_curated_podcasts(load_fixture("page_podcast.json"))
    ids = [p.id for p in podcasts]
    assert ids == ["utro_haap_og_kjaerlighet", "krimrommet"]
    assert podcasts[0].total_episodes


def test_parse_episodes_page() -> None:
    """An episode page yields its episodes, newest first, and says a next page exists."""
    episodes, has_next = parse_episodes(load_fixture("episodes.json"), PODCAST_ID)
    assert has_next
    assert len(episodes) == 3
    first = episodes[0]
    assert first.podcast_id == PODCAST_ID
    assert first.episode_id == "l_9b4b0949-cb93-4824-8b09-49cb930824d4"
    assert first.duration == 4000
    assert first.published == datetime(2026, 9, 21, 4, 0, tzinfo=UTC)
    assert first.available
    assert episodes[0].published is not None
    assert episodes[1].published is not None
    assert episodes[0].published > episodes[1].published


def test_parse_single_episode_duration_object() -> None:
    """The single-episode endpoint nests the duration in an object; it still reads."""
    episode = parse_episode(load_fixture("episode_single.json"), PODCAST_ID)
    assert episode is not None
    assert episode.duration == 4000


def test_parse_episode_manifest() -> None:
    """A podcast manifest gives the MP3 asset and its duration."""
    stream = parse_manifest(load_fixture("manifest_episode.json"))
    assert stream is not None
    assert stream.content_type == "mp3"
    assert not stream.is_live
    assert stream.duration == 4000
    assert stream.url.endswith(".mp3")


def test_parse_channel_manifest() -> None:
    """A channel manifest gives the live AAC HLS stream."""
    stream = parse_manifest(load_fixture("manifest_channel.json"))
    assert stream is not None
    assert stream.is_live
    assert stream.content_type == "aac"
    assert ".m3u8" in stream.url


def test_parse_manifest_not_playable() -> None:
    """A manifest NRK marks as not playable yields no stream."""
    manifest = load_fixture("manifest_episode.json")
    manifest["playability"] = "nonPlayable"
    assert parse_manifest(manifest) is None


def test_parse_favourites_keeps_podcasts_only() -> None:
    """Followed radio series are skipped (not supported yet) and the next page is kept."""
    data = load_fixture("favourites.json")
    data["favourites"].append(
        {
            "favouriteContentType": "series",
            "_links": {"series": {"href": "/radio/catalog/series/nitimen"}},
        }
    )
    ids, next_href = parse_favourites(data)
    assert ids == ["abels_taarn", PODCAST_ID]
    assert next_href is not None
    assert "key=" in next_href


def test_parse_progress_in_progress() -> None:
    """A part-played episode gives its position, timestamp and delete link."""
    progress = parse_progress(load_fixture("progress_episode.json"))
    assert progress.fully_played is False
    assert progress.position == 23 * 60 + 12
    assert progress.updated == datetime(2026, 9, 26, 14, 3, 20, 342000, tzinfo=UTC)
    assert progress.delete_path == (
        "/radio/userdata/user-1/progress/1f2cdefa-7746-4c5c-ab5d-ce18b599d45d"
    )


@pytest.mark.parametrize(
    ("payload", "fully_played", "position"),
    [
        ({"progress": "finished", "finished": True}, True, 0),
        ({"progress": "notStarted"}, False, 0),
    ],
)
def test_parse_progress_states(
    payload: dict[str, object], fully_played: bool, position: int
) -> None:
    """Finished and not started episodes carry no position."""
    progress = parse_progress(payload)
    assert progress.fully_played is fully_played
    assert progress.position == position


def test_parse_podcast_progress_keys_by_episode_id() -> None:
    """Batch progress is keyed by the bare episode id, as the catalog lists it."""
    progress = parse_podcast_progress(load_fixture("progress_podcast.json"))
    assert set(progress) == {
        "l_9b4b0949-cb93-4824-8b09-49cb930824d4",
        "l_beb99ebc-3e8b-4dab-b99e-bc3e8b9dabf1",
    }
    assert progress["l_9b4b0949-cb93-4824-8b09-49cb930824d4"].position == 1392
    assert progress["l_beb99ebc-3e8b-4dab-b99e-bc3e8b9dabf1"].fully_played is True


def test_parse_podcast_progress_falls_back_to_the_id() -> None:
    """Without links, the `series|episode` id names the episode; bad entries are skipped."""
    data = [
        {"id": f"{PODCAST_ID}|l_x", "progress": "finished"},
        {"id": f"{PODCAST_ID}|l_y", "progress": "somethingNew"},
        "junk",
    ]
    assert list(parse_podcast_progress(data)) == ["l_x"]


def test_parse_podcast_progress_rejects_non_list() -> None:
    """An unexpected batch response raises so the provider falls back to MA's play log."""
    with pytest.raises(NrkApiError):
        parse_podcast_progress({"progresses": []})


def test_parse_progress_unknown_shape_raises() -> None:
    """An unrecognised progress response raises so the provider falls back to local resume."""
    with pytest.raises(NrkApiError):
        parse_progress({"somethingElse": 1})
