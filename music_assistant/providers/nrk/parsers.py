"""Parsers that turn NRK psapi JSON into typed models."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from music_assistant.helpers.datetime import from_iso_string

from .models import NrkApiError, NrkChannel, NrkEpisode, NrkPodcast, NrkProgress, NrkStream

if TYPE_CHECKING:
    from datetime import datetime

_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def parse_duration(value: Any) -> int:
    """
    Return an ISO 8601 duration (e.g. "PT1H6M40S") in whole seconds, 0 if unreadable.

    :param value: The duration string from NRK.
    """
    if not isinstance(value, str):
        return 0
    match = _ISO_DURATION.match(value)
    if not match:
        return 0
    parts = {key: float(val) for key, val in match.groupdict().items() if val}
    return int(
        parts.get("days", 0) * 86400
        + parts.get("hours", 0) * 3600
        + parts.get("minutes", 0) * 60
        + parts.get("seconds", 0)
    )


def format_duration(seconds: int) -> str:
    """
    Return seconds as an ISO 8601 duration, the format NRK uses for positions.

    :param seconds: The position in seconds.
    """
    return f"PT{max(int(seconds), 0)}S"


def parse_channels(data: Any) -> list[NrkChannel]:
    """
    Parse the live channel list from `radio/live`.

    :param data: The JSON array returned by NRK.
    """
    if not isinstance(data, list):
        raise NrkApiError("Unexpected live channel list")
    channels: list[NrkChannel] = []
    for entry in data:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        playback = (entry.get("_embedded") or {}).get("playback") or {}
        posters = playback.get("squarePosters") or playback.get("posters") or []
        image_url = None
        if posters and isinstance(posters[0], dict):
            poster = posters[0].get("image", posters[0])
            image_url = _largest(poster.get("items"), "url", "pixelWidth")
        channels.append(
            NrkChannel(
                id=entry["id"],
                name=playback.get("title") or entry["id"],
                description=playback.get("description") or None,
                image_url=image_url,
            )
        )
    return channels


def parse_search(data: Any) -> list[NrkPodcast]:
    """
    Parse `radio/search/search` into podcasts, folding episode hits into their podcast.

    :param data: The JSON object returned by NRK.
    """
    results = (data or {}).get("results") or {}
    podcasts: list[NrkPodcast] = []
    seen: set[str] = set()
    for hit in ((results.get("series") or {}).get("results")) or []:
        if hit.get("type") != "podcast" or not hit.get("seriesId") or hit["seriesId"] in seen:
            continue
        seen.add(hit["seriesId"])
        podcasts.append(
            NrkPodcast(
                id=hit["seriesId"],
                title=(hit.get("title") or hit["seriesId"]).strip(),
                description=hit.get("description") or None,
                image_url=_largest(hit.get("images_1_1") or hit.get("images"), "uri"),
            )
        )
    for hit in ((results.get("episodes") or {}).get("results")) or []:
        series_id = hit.get("seriesId")
        if hit.get("type") != "podcastEpisode" or not series_id or series_id in seen:
            continue
        seen.add(series_id)
        podcasts.append(
            NrkPodcast(
                id=series_id,
                title=(hit.get("seriesTitle") or series_id).strip(),
                image_url=_largest(hit.get("images_1_1"), "uri"),
            )
        )
    return podcasts


def parse_podcast(data: Any) -> NrkPodcast:
    """
    Parse `radio/catalog/podcast/{id}`.

    :param data: The JSON object returned by NRK.
    """
    series = (data or {}).get("series")
    if not isinstance(series, dict) or not series.get("id"):
        raise NrkApiError("Unexpected podcast response")
    titles = series.get("titles") or {}
    return NrkPodcast(
        id=series["id"],
        title=(titles.get("title") or series["id"]).strip(),
        description=titles.get("subtitle") or None,
        image_url=_largest(series.get("squareImage") or series.get("image"), "url"),
    )


def parse_curated_podcasts(data: Any) -> list[NrkPodcast]:
    """
    Parse the podcasts NRK features on `radio/pages/podcast`, without duplicates.

    :param data: The JSON object returned by NRK.
    """
    podcasts: list[NrkPodcast] = []
    seen: set[str] = set()
    for section in (data or {}).get("sections") or []:
        for plug in (section.get("included") or {}).get("plugs") or []:
            info = plug.get("podcast") if plug.get("type") == "podcast" else None
            if not isinstance(info, dict) or not info.get("podcastId"):
                continue
            if info["podcastId"] in seen:
                continue
            seen.add(info["podcastId"])
            podcasts.append(
                NrkPodcast(
                    id=info["podcastId"],
                    title=(info.get("podcastTitle") or plug.get("title") or "").strip(),
                    description=plug.get("description") or None,
                    image_url=info.get("imageUrl"),
                    total_episodes=info.get("numberOfEpisodes"),
                )
            )
    return podcasts


def parse_episodes(data: Any, podcast_id: str) -> tuple[list[NrkEpisode], bool]:
    """
    Parse one page of `radio/catalog/podcast/{id}/episodes`.

    Returns the episodes and whether NRK has a next page.

    :param data: The JSON object returned by NRK.
    :param podcast_id: The podcast the episodes belong to.
    """
    if not isinstance(data, dict):
        raise NrkApiError("Unexpected episode list")
    embedded = data.get("_embedded") or {}
    episodes = [parse_episode(ep, podcast_id) for ep in embedded.get("episodes") or []]
    has_next = bool((data.get("_links") or {}).get("next"))
    return [ep for ep in episodes if ep is not None], has_next


def parse_episode(data: Any, podcast_id: str) -> NrkEpisode | None:
    """
    Parse a single catalog episode, None if it has no id.

    :param data: The episode JSON object.
    :param podcast_id: The podcast the episode belongs to.
    """
    if not isinstance(data, dict) or not data.get("episodeId"):
        return None
    titles = data.get("titles") or {}
    return NrkEpisode(
        podcast_id=podcast_id,
        episode_id=data["episodeId"],
        title=(titles.get("title") or data["episodeId"]).strip(),
        description=titles.get("subtitle") or None,
        image_url=_largest(data.get("squareImage") or data.get("image"), "url"),
        duration=_episode_duration(data),
        published=_iso(data.get("date")),
        available=(data.get("availability") or {}).get("status", "available") == "available",
    )


def parse_manifest(data: Any) -> NrkStream | None:
    """
    Parse a playback manifest, None when NRK says it is not playable.

    :param data: The JSON object from `playback/manifest/...`.
    """
    if not isinstance(data, dict):
        raise NrkApiError("Unexpected playback manifest")
    if data.get("playability") != "playable":
        return None
    playable = data.get("playable") or {}
    assets = playable.get("assets") or []
    if not assets or not isinstance(assets[0], dict) or not assets[0].get("url"):
        return None
    asset = assets[0]
    is_live = data.get("streamingMode") == "live"
    fmt = str(asset.get("format") or "").lower()
    return NrkStream(
        url=asset["url"],
        content_type="mp3" if fmt == "mp3" else "aac",
        is_live=is_live,
        duration=0 if is_live else parse_duration(playable.get("duration")),
    )


def parse_favourites(data: Any) -> tuple[list[str], str | None]:
    """
    Parse one page of the user's favourites into podcast ids and the next page path.

    :param data: The JSON object from `radio/userdata/{userId}/favourites`.
    """
    if not isinstance(data, dict):
        raise NrkApiError("Unexpected favourites response")
    ids: list[str] = []
    for fav in data.get("favourites") or []:
        if fav.get("favouriteContentType") != "podcast":
            continue
        href = ((fav.get("_links") or {}).get("podcasts") or {}).get("href") or ""
        podcast_id = href.rstrip("/").rsplit("/", 1)[-1]
        if podcast_id:
            ids.append(podcast_id)
    next_href = ((data.get("_links") or {}).get("next") or {}).get("href")
    return ids, next_href or None


def parse_progress(data: Any) -> NrkProgress:
    """
    Parse the user's progress on one episode.

    Shape from NRK's Personalization API (GetProgressForContent, `ProgramProgress`):
    `progress` is notStarted, inProgress or finished, `inProgress.time` the position.
    https://psapi.nrk.no/documentation/redoc/personlig/

    :param data: The JSON object from `radio/userdata/{userId}/progress/podcastepisode/...`.
    """
    if not isinstance(data, dict):
        raise NrkApiError("Unexpected progress response")
    state = data.get("progress")
    if state not in ("notStarted", "inProgress", "finished"):
        raise NrkApiError(f"Unrecognised progress response: {sorted(data)}")
    in_progress = data.get("inProgress") or {}
    links = data.get("_links") or {}
    return NrkProgress(
        fully_played=state == "finished" or data.get("finished") is True,
        position=parse_duration(in_progress.get("time")) if state == "inProgress" else 0,
        updated=_iso(data.get("registeredAt")),
        delete_path=(links.get("deleteProgress") or {}).get("href"),
    )


def parse_podcast_progress(data: Any) -> dict[str, NrkProgress]:
    """
    Parse the progress of a batch of episodes, keyed by NRK episode id.

    :param data: The JSON array from `radio/userdata/{userId}/progress/podcast/{seriesId}`.
    """
    if not isinstance(data, list):
        raise NrkApiError("Unexpected podcast progress response")
    result: dict[str, NrkProgress] = {}
    for entry in data:
        if not isinstance(entry, dict) or not (episode_id := _progress_episode_id(entry)):
            continue
        try:
            result[episode_id] = parse_progress(entry)
        except NrkApiError:
            continue
    return result


def _progress_episode_id(entry: dict[str, Any]) -> str | None:
    """Return the episode id a progress entry is about, from its links or its id."""
    links = entry.get("_links") or {}
    for key in ("podcastEpisode", "self"):
        href = (links.get(key) or {}).get("href") or ""
        # The href ends in the episode id, or in the series id and episode id joined by a pipe.
        tail = href.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1].rsplit("|", 1)[-1]
        if tail.startswith("l_"):
            return tail
    content_id = entry.get("contentId") or entry.get("id")
    if isinstance(content_id, str) and content_id:
        return content_id.rsplit("|", 1)[-1]
    return None


def _episode_duration(data: dict[str, Any]) -> int:
    """Return an episode's duration in seconds; lists and single episodes differ in shape."""
    if isinstance(data.get("durationInSeconds"), int):
        return int(data["durationInSeconds"])
    duration = data.get("duration")
    if isinstance(duration, dict):
        return int(duration.get("seconds") or 0)
    return parse_duration(duration)


def _iso(value: Any) -> datetime | None:
    """Return an ISO timestamp as datetime, None if absent or unreadable."""
    if not isinstance(value, str):
        return None
    try:
        return from_iso_string(value)
    except ValueError:
        return None


def _largest(images: Any, url_key: str, width_key: str = "width") -> str | None:
    """Return the url of the widest image in an NRK image list."""
    if not isinstance(images, list):
        return None
    candidates = [img for img in images if isinstance(img, dict) and img.get(url_key)]
    if not candidates:
        return None
    best = max(candidates, key=lambda img: img.get(width_key) or 0)
    return str(best[url_key])
