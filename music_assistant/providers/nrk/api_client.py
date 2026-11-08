"""Async client for NRK's public radio API (psapi.nrk.no)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp
from music_assistant_models.errors import RateLimited

from music_assistant.helpers.throttle_retry import (
    ThrottlerManager,
    parse_retry_after,
    throttle_with_retries,
)

from .constants import (
    API_HEADERS,
    API_URL,
    EPISODE_PAGE_SIZE,
    FAVOURITES_PAGE_SIZE,
    PROGRESS_BATCH_SIZE,
    REQUEST_RATE_LIMIT,
    REQUEST_TIMEOUT,
)
from .models import (
    NrkApiError,
    NrkChannel,
    NrkEpisode,
    NrkNotFoundError,
    NrkPodcast,
    NrkProgress,
    NrkRequestError,
    NrkStream,
)
from .parsers import (
    format_duration,
    parse_channels,
    parse_curated_podcasts,
    parse_episode,
    parse_episodes,
    parse_favourites,
    parse_manifest,
    parse_podcast,
    parse_podcast_progress,
    parse_progress,
    parse_search,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import AsyncGenerator

    from aiohttp import ClientSession


class NrkClient:
    """
    Thin async client for the NRK radio API.

    Keeps endpoints and parsing apart from the MA provider glue so it can be patched
    in one place when NRK changes its API. Every request is throttled, and HTTP 429 or
    a transient failure is retried with backoff.
    """

    def __init__(self, session: ClientSession, logger: logging.Logger) -> None:
        """
        Initialize the client.

        :param session: Shared aiohttp session (use the MA http session).
        :param logger: Logger for diagnostics.
        """
        self._session = session
        self.logger = logger
        self.throttler = ThrottlerManager(
            rate_limit=REQUEST_RATE_LIMIT, period=1.0, retry_attempts=3, initial_backoff=2
        )

    async def search(self, query: str, take: int) -> list[NrkPodcast]:
        """
        Search podcasts; episode hits are returned as their podcast.

        :param query: The search text.
        :param take: How many hits NRK returns per category.
        """
        data = await self._request("GET", "radio/search/search", params={"q": query, "take": take})
        return parse_search(data)

    async def get_channels(self) -> list[NrkChannel]:
        """Return all live NRK radio channels."""
        return parse_channels(await self._request("GET", "radio/live"))

    async def get_curated_podcasts(self) -> list[NrkPodcast]:
        """Return the podcasts NRK features on its podcast front page."""
        return parse_curated_podcasts(await self._request("GET", "radio/pages/podcast"))

    async def get_podcast(self, podcast_id: str) -> NrkPodcast:
        """
        Return a podcast by id.

        :param podcast_id: The NRK series id of the podcast.
        """
        return parse_podcast(await self._request("GET", f"radio/catalog/podcast/{podcast_id}"))

    async def iter_episodes(self, podcast_id: str) -> AsyncGenerator[NrkEpisode]:
        """
        Yield every episode of a podcast, newest first.

        :param podcast_id: The NRK series id of the podcast.
        """
        page = 1
        while True:
            data = await self._request(
                "GET",
                f"radio/catalog/podcast/{podcast_id}/episodes",
                params={"page": page, "pageSize": EPISODE_PAGE_SIZE},
            )
            episodes, has_next = parse_episodes(data, podcast_id)
            for episode in episodes:
                yield episode
            if not has_next or not episodes:
                return
            page += 1

    async def get_episode(self, podcast_id: str, episode_id: str) -> NrkEpisode:
        """
        Return a single episode.

        :param podcast_id: The NRK series id of the podcast.
        :param episode_id: The NRK episode id.
        """
        data = await self._request(
            "GET", f"radio/catalog/podcast/{podcast_id}/episodes/{episode_id}"
        )
        episode = parse_episode(data, podcast_id)
        if episode is None:
            raise NrkNotFoundError(f"Episode {episode_id} not found")
        return episode

    async def get_episode_stream(self, episode_id: str) -> NrkStream | None:
        """
        Return the playable asset of an episode, None when NRK says it is not playable.

        :param episode_id: The NRK episode id.
        """
        return parse_manifest(await self._request("GET", f"playback/manifest/podcast/{episode_id}"))

    async def get_channel_stream(self, channel_id: str) -> NrkStream | None:
        """
        Return the live stream of a channel, None when NRK says it is not playable.

        :param channel_id: The NRK channel id, e.g. "p1".
        """
        return parse_manifest(await self._request("GET", f"playback/manifest/channel/{channel_id}"))

    async def iter_followed_podcast_ids(self, user_id: str, token: str) -> AsyncGenerator[str]:
        """
        Yield the ids of the podcasts the user follows in NRK Radio.

        :param user_id: The NRK user id.
        :param token: A valid bearer token.
        """
        path: str | None = f"radio/userdata/{user_id}/favourites"
        params: dict[str, Any] | None = {
            "favouriteType": "any",
            "sortOrder": "descending",
            "pageSize": FAVOURITES_PAGE_SIZE,
        }
        while path:
            data = await self._request("GET", path, params=params, token=token)
            ids, path = parse_favourites(data)
            # The next link carries its own query (including the paging key).
            params = None
            for podcast_id in ids:
                yield podcast_id

    async def get_progress(
        self, user_id: str, token: str, podcast_id: str, episode_id: str
    ) -> NrkProgress:
        """
        Return the user's progress on an episode (NRK's GetProgressForContent).

        :param user_id: The NRK user id.
        :param token: A valid bearer token.
        :param podcast_id: The NRK series id of the podcast.
        :param episode_id: The NRK episode id.
        """
        data = await self._request(
            "GET", _progress_path(user_id, podcast_id, episode_id), token=token
        )
        return parse_progress(data)

    async def get_podcast_progress(
        self, user_id: str, token: str, podcast_id: str, episode_ids: list[str]
    ) -> dict[str, NrkProgress]:
        """
        Return the user's progress on many episodes of a podcast, keyed by episode id.

        Uses NRK's GetProgressForPodcast, which takes at most 50 episode ids per call.
        Episodes without progress are left out.

        :param user_id: The NRK user id.
        :param token: A valid bearer token.
        :param podcast_id: The NRK series id of the podcast.
        :param episode_ids: The NRK episode ids.
        """
        result: dict[str, NrkProgress] = {}
        for start in range(0, len(episode_ids), PROGRESS_BATCH_SIZE):
            batch = episode_ids[start : start + PROGRESS_BATCH_SIZE]
            data = await self._request(
                "GET",
                f"radio/userdata/{user_id}/progress/podcast/{podcast_id}",
                params={"contentIds": ",".join(batch)},
                token=token,
            )
            result.update(parse_podcast_progress(data))
        return result

    async def put_progress(
        self,
        user_id: str,
        token: str,
        podcast_id: str,
        episode_id: str,
        position: int,
        start_position: int,
    ) -> None:
        """
        Store the user's position in an episode (NRK's PutProgress).

        NRK marks the episode finished itself when the position reaches its end.

        :param user_id: The NRK user id.
        :param token: A valid bearer token.
        :param podcast_id: The NRK series id of the podcast.
        :param episode_id: The NRK episode id.
        :param position: The current position in seconds.
        :param start_position: Where this playback started, in seconds.
        """
        body = {
            "position": format_duration(position),
            "startPlaybackPosition": format_duration(start_position),
        }
        await self._request(
            "PUT", _progress_path(user_id, podcast_id, episode_id), token=token, json=body
        )

    async def delete_progress(self, token: str, path: str) -> None:
        """
        Delete stored progress, which marks the episode unplayed in NRK Radio.

        :param token: A valid bearer token.
        :param path: The `deleteProgress` link from the episode's progress.
        """
        await self._request("DELETE", path, token=token)

    @throttle_with_retries
    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        token: str | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Call the NRK API and return the decoded JSON body (None when empty)."""
        headers = API_HEADERS
        if token:
            headers = {**API_HEADERS, "Authorization": f"Bearer {token}"}
        url = f"{API_URL}/{path.lstrip('/')}"
        try:
            async with self._session.request(
                method, url, params=params, headers=headers, json=json, timeout=REQUEST_TIMEOUT
            ) as resp:
                if resp.status == 429:
                    raise RateLimited(
                        "NRK rate limit reached",
                        backoff_time=parse_retry_after(resp.headers.get("Retry-After")),
                    )
                if resp.status == 404:
                    raise NrkNotFoundError(f"Not found on NRK: {path}")
                if 400 <= resp.status < 500:
                    raise NrkRequestError(f"NRK returned HTTP {resp.status} for {path}")
                resp.raise_for_status()
                if resp.status in (202, 204):
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            raise NrkApiError(f"NRK request failed: {err}") from err


def _progress_path(user_id: str, podcast_id: str, episode_id: str) -> str:
    """Return the progress endpoint of an episode, as in NRK's own `progress` link."""
    return f"radio/userdata/{user_id}/progress/podcastepisode/{podcast_id}|{episode_id}"
