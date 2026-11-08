"""NRK Radio music provider implementation."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Callable, Sequence
from datetime import datetime, timedelta
from time import monotonic
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ProviderError
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    EventType,
    ImageType,
    MediaType,
    PlaybackState,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    MediaNotFoundError,
    MusicAssistantError,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    MediaItemType,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
    Radio,
    SearchResults,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import (
    CONF_ENTRY_UNOFFICIAL_PROVIDER,
    CONF_PASSWORD,
    CONF_USERNAME,
)
from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.datetime import utc
from music_assistant.helpers.podcast_parsers import rank_episodes_by_date
from music_assistant.models.music_provider import MusicProvider

from .api_client import NrkClient
from .auth import NrkAuth
from .constants import (
    BROWSE_FOLLOWED,
    BROWSE_PODCASTS,
    BROWSE_RADIOS,
    EPISODE_ID_SEPARATOR,
    SEARCH_TIMEOUT,
)
from .models import (
    NrkApiError,
    NrkAuthError,
    NrkChannel,
    NrkEpisode,
    NrkNotFoundError,
    NrkPodcast,
    NrkProgress,
    NrkRequestError,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.playback_progress_report import MediaItemPlaybackProgressReport
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
}
# Only meaningful with an NRK account: the podcasts the user follows in NRK Radio.
ACCOUNT_FEATURES = {ProviderFeature.LIBRARY_PODCASTS}
# The radio.nrk.no cookies of a browser login, as a Cookie header. NRK's password login
# often asks for a captcha, so a session copied from a browser is the dependable way in.
CONF_SESSION = "nrk_session"

# Errors that mean NRK's progress sync did not work, so MA's own resume takes over.
_PROGRESS_ERRORS = (NrkApiError, NrkAuthError, NrkRequestError, NrkNotFoundError)

# Two positions this close (seconds) are the same place, so no seek and no correction.
_SAME_POSITION = 10
# Clock difference allowed between NRK's progress timestamp and MA's own report.
_CLOCK_SKEW = timedelta(seconds=10)
# MA reports playback every 30 seconds of elapsed time, so on the wall clock two reports
# can land a little under 30 s apart. A gap below this is a duplicate and is not written.
_MIN_WRITE_GAP = 25
# An episode counts as finished past this share of its length, or this close (seconds)
# to its end.
_FINISHED_SHARE = 0.95
_FINISHED_TAIL = 60


class NrkProvider(MusicProvider):
    """NRK Radio provider."""

    _client: NrkClient
    _auth: NrkAuth

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize the NRK Radio provider."""
        super().__init__(mass, manifest, config, SUPPORTED_FEATURES)
        self._progress_warned = False
        # When progress of an episode was last written to NRK (monotonic), for throttling.
        self._last_writes: dict[str, float] = {}
        # Episodes of other providers (by uri) linked to their NRK episode id and length,
        # None when they are not an NRK podcast.
        self._links: dict[str, tuple[str, int] | None] = {}
        # Where the current playback of an episode started, in seconds, for NRK's
        # startPlaybackPosition. Set right before playback, dropped when it stops.
        self._playback_starts: dict[str, int] = {}
        # Episode playing in each queue, to notice when playback (re)starts.
        self._queue_episodes: dict[str, str | None] = {}
        # Last position this provider reported for an episode, and when (wall clock).
        self._last_reports: dict[str, tuple[int, datetime]] = {}
        # Live position of an episode: (seconds, monotonic time), time None while paused.
        self._live: dict[str, tuple[int, float | None]] = {}
        # Where the current stream of an episode was started from (MA's seek_position).
        self._stream_offsets: dict[str, int] = {}
        # Episodes being matched to NRK's progress right now; their reports wait.
        self._reconciling: set[str] = set()
        self._unsubscribers: list[Callable[[], None]] = []
        # How the last NRK login went, shown as the provider's error; None when it worked.
        self._login_error: ProviderError | None = None

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return provider configuration entries (optional NRK account login)."""
        return (
            CONF_ENTRY_UNOFFICIAL_PROVIDER,
            ConfigEntry(key=CONF_USERNAME, type=ConfigEntryType.STRING, required=False),
            ConfigEntry(key=CONF_PASSWORD, type=ConfigEntryType.SECURE_STRING, required=False),
            ConfigEntry(
                key=CONF_SESSION,
                type=ConfigEntryType.SECURE_STRING,
                required=False,
                advanced=True,
            ),
        )

    @property
    def has_account(self) -> bool:
        """Return True when an NRK session or account credentials are configured."""
        # Read from config: MA resolves the features before handle_async_init runs.
        return bool(self.config.get_value(CONF_SESSION)) or bool(
            self.config.get_value(CONF_USERNAME) and self.config.get_value(CONF_PASSWORD)
        )

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Return the features supported by this provider."""
        # Without an account there is nothing to sync, and a declared library feature
        # would let the deletion pass prune podcasts synced earlier.
        if self.has_account:
            return {*SUPPORTED_FEATURES, *ACCOUNT_FEATURES}
        return {*SUPPORTED_FEATURES}

    @property
    def supported_media_types(self) -> set[MediaType]:
        """Return the media types this provider can serve."""
        return {MediaType.RADIO, MediaType.PODCAST, MediaType.PODCAST_EPISODE}

    async def handle_async_init(self) -> None:
        """Set up the NRK client and, with a session or credentials, log in."""
        self._client = NrkClient(self.mass.http_session, self.logger)
        email = str(self.config.get_value(CONF_USERNAME) or "")
        password = str(self.config.get_value(CONF_PASSWORD) or "")
        session = str(self.config.get_value(CONF_SESSION) or "")
        self._auth = NrkAuth(
            self.mass,
            self.logger,
            email,
            password,
            session=session,
            save_session=self._save_session,
            report_login=self._report_login,
        )
        if not self._auth.enabled:
            self.logger.info("No NRK account configured, so listening progress is not synced")
            return
        # Follow playback even when the login fails now: every sync asks for a token
        # again, so sync starts by itself once NRK lets the login through.
        self._unsubscribers = [
            self.mass.subscribe(self._on_queue_updated, EventType.QUEUE_UPDATED),
            self.mass.subscribe(self._on_media_item_played, EventType.MEDIA_ITEM_PLAYED),
        ]
        try:
            await self._auth.get_token()
        except NrkAuthError as err:
            # Radio and podcasts need no account, so a refused login must not take
            # them down with it.
            self.logger.warning(
                "%s. Live radio and podcasts still work, but followed podcasts and "
                "listening progress are not synced until the login works.",
                err if "NRK login needed" in str(err) else f"NRK login needed ({err})",
            )
            return
        except NrkApiError as err:
            self.logger.warning("Could not reach NRK to log in, trying again later: %s", err)
            return
        self.logger.info("Logged in to NRK, listening progress sync is on")

    async def loaded_in_mass(self) -> None:
        """Show a failed login again, since MA clears the provider's error once it has loaded."""
        # MA starts this step eagerly and clears last_error right after, so let that run first.
        await asyncio.sleep(0)
        if self._login_error:
            self.mass.config.update_provider_last_error(self.instance_id, self._login_error)

    async def unload(self, is_removed: bool = False) -> None:
        """Stop following queue updates and playback reports."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers = []

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Browse live radio, featured podcasts and (with an account) followed podcasts.

        :param path: The path to browse (e.g. provider_id://radio).
        """
        subpath = path.split("://", 1)[1] if "://" in path else ""
        if subpath == BROWSE_RADIOS:
            return await self._get_radios()
        if subpath == BROWSE_PODCASTS:
            return await self._get_curated_podcasts()
        if subpath == BROWSE_FOLLOWED and self._auth.enabled:
            return [podcast async for podcast in self.get_library_podcasts()]
        return self._browse_root()

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """
        Search live channels and podcasts (episode hits are returned as their podcast).

        :param search_query: The search text.
        :param media_types: The media types to search for.
        :param limit: Maximum number of results per media type.
        """
        results = SearchResults()
        query = search_query.strip()
        if not query:
            return results
        if MediaType.RADIO in media_types:
            needle = query.lower()
            radios = await self._get_radios()
            results.radio = [radio for radio in radios if needle in radio.name.lower()][:limit]
        if MediaType.PODCAST in media_types:
            try:
                async with asyncio.timeout(SEARCH_TIMEOUT):
                    podcasts = await self._client.search(query, limit)
            except (TimeoutError, MusicAssistantError) as err:
                self.logger.debug("NRK search for %r failed: %s", query, err)
                podcasts = []
            results.podcasts = [self._podcast_item(podcast) for podcast in podcasts[:limit]]
        return results

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """
        Get a live channel by id.

        :param prov_radio_id: The NRK channel id, e.g. "p1".
        """
        for radio in await self._get_radios():
            if radio.item_id == prov_radio_id:
                return radio
        raise MediaNotFoundError(f"NRK channel {prov_radio_id} not found")

    @use_cache(3600 * 6, base_class=Podcast)
    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """
        Get a podcast by id.

        :param prov_podcast_id: The NRK series id of the podcast.
        """
        return self._podcast_item(await self._client.get_podcast(prov_podcast_id))

    async def get_podcast_episodes(self, prov_podcast_id: str) -> AsyncGenerator[PodcastEpisode]:
        """
        Get all episodes of a podcast, with the user's NRK listening progress.

        :param prov_podcast_id: The NRK series id of the podcast.
        """
        episodes = await self._fetch_episodes(prov_podcast_id)
        progress = await self._read_podcast_progress(prov_podcast_id, episodes)
        for episode in episodes:
            yield _with_progress(episode, progress.get(_split_episode_id(episode.item_id)[1]))

    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """
        Get a single episode, with the user's NRK listening progress.

        :param prov_episode_id: The episode id, "<series id>|<episode id>".
        """
        episode = await self._fetch_episode(prov_episode_id)
        return _with_progress(episode, await self._read_progress(prov_episode_id))

    async def get_library_podcasts(self) -> AsyncGenerator[Podcast]:
        """Sync the podcasts the user follows in NRK Radio to the library."""
        if not self._auth.enabled:
            return
        # A failure here must abort the sync. Yielding nothing would tell MA the user
        # unfollowed everything, and the deletion pass would prune the whole library.
        user_id, token = await self._auth.get_token()
        async for podcast_id in self._client.iter_followed_podcast_ids(user_id, token):
            try:
                yield await self.get_podcast(podcast_id)
            except MediaNotFoundError:
                self.logger.debug("Skipping followed podcast NRK no longer has: %s", podcast_id)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """
        Resolve a playable stream.

        :param item_id: The channel id or episode id.
        :param media_type: RADIO or PODCAST_EPISODE.
        """
        if media_type == MediaType.RADIO:
            stream = await self._client.get_channel_stream(item_id)
        elif media_type == MediaType.PODCAST_EPISODE:
            stream = await self._client.get_episode_stream(_split_episode_id(item_id)[1])
        else:
            raise UnplayableMediaError(f"Unsupported media type: {media_type}")
        if stream is None:
            raise UnplayableMediaError("NRK says this is not playable right now")
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            media_type=media_type,
            stream_type=StreamType.HLS if stream.is_live else StreamType.HTTP,
            path=stream.url,
            audio_format=AudioFormat(content_type=ContentType.try_parse(stream.content_type)),
            duration=stream.duration or None,
            can_seek=not stream.is_live,
            allow_seek=not stream.is_live,
        )

    async def get_resume_position(
        self, item_id: str, media_type: MediaType
    ) -> tuple[bool, int, datetime | None]:
        """
        Return the NRK listening progress of an episode (fully_played, position_ms, updated).

        Raises NotImplementedError when NRK progress is unavailable, so MA falls back to
        its own resume position.

        :param item_id: The episode id, "<series id>|<episode id>".
        :param media_type: Only PODCAST_EPISODE has progress.
        """
        if media_type != MediaType.PODCAST_EPISODE or not self._auth.enabled:
            raise NotImplementedError
        progress = await self._read_progress(item_id)
        # Called right before playback starts, so this is where the playback starts.
        self._playback_starts[item_id] = progress.position if progress else 0
        if progress is None:
            raise NotImplementedError
        return progress.fully_played, progress.position * 1000, progress.updated

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """
        Write the listening progress of an episode back to NRK.

        MA calls this every 30 seconds while playing and on pause, stop and end. A
        finished episode is stored at its full length, which NRK marks as finished, and
        an episode marked unplayed in MA has its NRK progress deleted.

        :param media_type: Only PODCAST_EPISODE is written.
        :param prov_item_id: The episode id, "<series id>|<episode id>".
        :param fully_played: True when the episode was played to the end.
        :param position: The position in seconds.
        :param media_item: The played item.
        :param is_playing: True while playback continues.
        """
        if media_type != MediaType.PODCAST_EPISODE or not self._auth.enabled:
            return
        if prov_item_id in self._reconciling:
            # A report while matching NRK's progress may be the stale position MA
            # resumed from, which would overwrite the newer progress from the phone.
            self.logger.debug("Holding back NRK progress of %s while matching NRK's", prov_item_id)
            return
        position = self._correct_position(prov_item_id, position)
        self._live[prov_item_id] = (position, monotonic() if is_playing else None)
        self._last_reports[prov_item_id] = (position, utc())
        start = self._playback_starts.get(prov_item_id, position)
        if not is_playing:
            self._playback_starts.pop(prov_item_id, None)
        if position == 0 and not fully_played and not is_playing:
            await self._mark_unplayed(prov_item_id)
            return
        duration = getattr(media_item, "duration", None)
        await self._write_progress(
            prov_item_id,
            position,
            duration if isinstance(duration, int) else 0,
            is_playing=is_playing,
            fully_played=fully_played,
            start=start,
            source=self.instance_id,
        )

    def _report_login(self, err: NrkAuthError | NrkApiError | None) -> None:
        """Show how the last NRK login went as the provider's error."""
        if err is None:
            error = None
        elif isinstance(err, NrkAuthError):
            text = str(err)
            message = text if text.startswith("NRK login needed") else f"NRK login needed ({text})"
            error = ProviderError(error_code=err.error_code, message=message)
        else:
            error = ProviderError(error_code=err.error_code, message=f"Could not reach NRK ({err})")
        if error == self._login_error:
            return
        self._login_error = error
        self.mass.config.update_provider_last_error(self.instance_id, error)

    def _save_session(self, session: str) -> None:
        """Store the NRK session cookies NRK renewed, so a restart keeps the login."""
        self.mass.config.set_raw_provider_config_value(
            self.instance_id, CONF_SESSION, session, encrypted=True, immediate=True
        )

    async def _on_media_item_played(self, event: MassEvent) -> None:
        """
        Write NRK progress for an NRK episode that another provider played.

        A library podcast can be mapped to both NRK and, say, an RSS feed of it, and MA
        may stream its episodes from the feed. MA then reports the playback only to the
        feed provider, so this follows MA's playback reports and writes the progress of
        such episodes to the matching NRK episode.
        """
        report: MediaItemPlaybackProgressReport = event.data
        if report is None or report.media_type != MediaType.PODCAST_EPISODE:
            return
        source = report.uri.split("://", 1)[0]
        if source in (self.instance_id, self.domain):
            return  # NRK's own episodes are reported to on_played
        link = await self._link_episode(report.uri)
        if link is None:
            return
        item_id, duration = link
        if item_id in self._reconciling:
            self.logger.debug("Holding back NRK progress of %s while matching NRK's", item_id)
            return
        position = self._correct_position(item_id, int(report.seconds_played or 0))
        self._live[item_id] = (position, monotonic() if report.is_playing else None)
        self._last_reports[item_id] = (position, utc())
        if report.is_playing:
            start = self._playback_starts.setdefault(item_id, position)
        else:
            start = self._playback_starts.pop(item_id, position)
        await self._write_progress(
            item_id,
            position,
            duration or int(report.duration or 0),
            is_playing=report.is_playing,
            fully_played=report.fully_played,
            start=start,
            source=source,
        )

    async def _write_progress(
        self,
        item_id: str,
        position: int,
        duration: int,
        *,
        is_playing: bool,
        fully_played: bool,
        start: int,
        source: str,
    ) -> None:
        """
        Write the position in an episode to NRK, at most every 30 s while it plays.

        A finished episode is stored at its full length, which NRK marks as finished.
        Every write and every failure is logged.

        :param item_id: The NRK episode id, "<series id>|<episode id>".
        :param position: The position in seconds.
        :param duration: The length of the episode in seconds, 0 when unknown.
        :param is_playing: True while playback continues.
        :param fully_played: True when MA says the episode was played to the end.
        :param start: Where this playback started, in seconds.
        :param source: The provider instance that played the episode, for the log.
        """
        if duration > 0 and position > duration + _FINISHED_TAIL:
            # MA reports such positions when a stream breaks off, like on a restart.
            self.logger.warning(
                "Not saving NRK progress of %s: %ss is past the end (%ss), played by %s",
                item_id,
                position,
                duration,
                source,
            )
            return
        finished = fully_played or _is_finished(position, duration)
        now = monotonic()
        last = self._last_writes.get(item_id)
        if is_playing and not finished and last is not None and now - last < _MIN_WRITE_GAP:
            self.logger.debug("NRK progress of %s written %.0fs ago, skipping", item_id, now - last)
            return
        if finished:
            position = max(position, duration)
        state = "finished" if finished else "playing" if is_playing else "paused or stopped"
        podcast_id, episode_id = _split_episode_id(item_id)
        try:
            user_id, token = await self._auth.get_token()
            await self._client.put_progress(
                user_id, token, podcast_id, episode_id, position, min(start, position)
            )
        except _PROGRESS_ERRORS as err:
            self.logger.warning(
                "Could not save NRK progress of %s at %ss (%s, played by %s): %s",
                item_id,
                position,
                state,
                source,
                err,
            )
            return
        self._last_writes[item_id] = now
        self.logger.info(
            "Saved NRK progress of %s: %ss of %ss, %s (played by %s)",
            item_id,
            position,
            duration or "?",
            state,
            source,
        )

    async def _link_episode(self, uri: str) -> tuple[str, int] | None:
        """
        Return the NRK episode id and length for another provider's episode, else None.

        :param uri: The uri of the played episode.
        """
        if uri in self._links:
            return self._links[uri]
        try:
            episode = await self.mass.music.get_item_by_uri(uri)
            link = (
                await self._match_nrk_episode(episode)
                if isinstance(episode, PodcastEpisode)
                else None
            )
        except MusicAssistantError as err:
            self.logger.warning("Could not link %s to an NRK episode: %s", uri, err)
            return None
        self._links[uri] = link
        if link:
            self.logger.info("Syncing progress of %s to NRK episode %s", uri, link[0])
        return link

    async def _match_nrk_episode(self, episode: PodcastEpisode) -> tuple[str, int] | None:
        """
        Find the NRK episode of an episode another provider serves, through the library.

        The podcast must be in the library with an NRK mapping. The episode is matched on
        its id (NRK's RSS feeds use NRK's episode id as guid), then on its title.

        :param episode: The played episode.
        """
        podcast = episode.podcast
        library: MediaItemType | BrowseFolder | None
        if podcast.provider == "library":
            library = await self.mass.music.get_item(MediaType.PODCAST, podcast.item_id, "library")
        else:
            library = await self.mass.music.get_library_item_by_prov_id(
                MediaType.PODCAST, podcast.item_id, podcast.provider
            )
        podcast_id = next(
            (
                mapping.item_id
                for mapping in getattr(library, "provider_mappings", ())
                if mapping.provider_instance == self.instance_id
            ),
            None,
        )
        if podcast_id is None:
            self.logger.debug("%s is not an NRK podcast in the library", podcast.name)
            return None
        match = _find_nrk_episode(await self._fetch_episodes(podcast_id), episode)
        if match is None:
            self.logger.warning(
                "No NRK episode of %s matches %r, so its progress is not synced to NRK",
                podcast_id,
                episode.name,
            )
            return None
        return match.item_id, match.duration or 0

    def _correct_position(self, item_id: str, position: int) -> int:
        """
        Return the live position when MA reports one without its stream offset.

        On pause some players report the time played in the current stream only, so an
        episode resumed at 29:00 and paused at 44:00 is reported at 15:00. That value
        would be saved to NRK and the next resume would start 29 minutes behind. A
        report that is off from the live position by exactly the stream offset is that
        case. Other players report where the stream started, so an episode resumed from
        the feed at 3:14 and paused at 6:00 is reported at 3:16; a report at the stream
        start while the live position is well past it is that case. Any other
        difference is a real seek and is kept.

        :param item_id: The episode id.
        :param position: The reported position in seconds.
        """
        offset = self._stream_offsets.get(item_id, 0)
        if not offset or item_id not in self._live:
            return position
        live_position, since = self._live[item_id]
        expected = live_position + (monotonic() - since if since is not None else 0)
        if abs(position - expected) <= _SAME_POSITION:
            return position
        if abs(position + offset - expected) <= _SAME_POSITION:
            return position + offset
        if abs(position - offset) <= _SAME_POSITION:
            return round(expected)
        return position

    async def _on_queue_updated(self, event: MassEvent) -> None:
        """Match NRK's progress when an NRK episode starts or resumes playing."""
        queue: PlayerQueue = event.data
        item_id = await self._playing_episode(queue)
        previous = self._queue_episodes.get(queue.queue_id)
        self._queue_episodes[queue.queue_id] = item_id
        if item_id is None or queue.current_item is None:
            return
        streamdetails = queue.current_item.streamdetails
        offset = int(streamdetails.seek_position if streamdetails else 0)
        if item_id in self._stream_offsets and self._stream_offsets[item_id] != offset:
            # A seek starts a new stream, so playback now runs from its offset.
            self._live[item_id] = (offset, monotonic())
        self._stream_offsets[item_id] = offset
        if item_id == previous or item_id in self._reconciling:
            return
        await self._reconcile(queue.queue_id, item_id, int(queue.elapsed_time or 0))

    async def _playing_episode(self, queue: PlayerQueue) -> str | None:
        """
        Return the id of the NRK episode a queue is playing, else None.

        An episode streamed by another provider, like the podcast's RSS feed, counts as
        its linked NRK episode, so it resumes from NRK's progress too.
        """
        if queue.state != PlaybackState.PLAYING or queue.current_item is None:
            return None
        streamdetails = queue.current_item.streamdetails
        if streamdetails is None or streamdetails.media_type != MediaType.PODCAST_EPISODE:
            return None
        if streamdetails.provider == self.instance_id:
            return streamdetails.item_id
        uri = f"{streamdetails.provider}://podcast_episode/{streamdetails.item_id}"
        link = await self._link_episode(uri)
        return link[0] if link else None

    async def _reconcile(self, queue_id: str, item_id: str, elapsed: int) -> None:
        """
        Seek to NRK's progress when it is newer than MA's and somewhere else.

        :param queue_id: The queue playing the episode.
        :param item_id: The episode id.
        :param elapsed: Where MA resumed playback, in seconds.
        """
        # A paused episode resumes where it was paused, which MA's elapsed time may not
        # show yet (see _correct_position).
        live = self._live.get(item_id)
        position = live[0] if live and live[1] is None else elapsed
        self._reconciling.add(item_id)
        try:
            progress = await self._read_progress(item_id)
            target = _resume_target(progress, position, self._last_reports.get(item_id))
            if target is not None:
                self.logger.debug("Resuming %s at NRK's newer position %ss", item_id, target)
                await self.mass.player_queues.seek(queue_id, target)
                position = target
                self._playback_starts[item_id] = target
        except MusicAssistantError as err:
            self.logger.debug("Could not seek %s to NRK's progress: %s", item_id, err)
        finally:
            self._reconciling.discard(item_id)
        self._live[item_id] = (position, monotonic())
        self._playback_starts.setdefault(item_id, position)

    async def _mark_unplayed(self, item_id: str) -> None:
        """Delete the NRK progress of an episode, through the link NRK gives for it."""
        podcast_id, episode_id = _split_episode_id(item_id)
        try:
            user_id, token = await self._auth.get_token()
            progress = await self._client.get_progress(user_id, token, podcast_id, episode_id)
            if progress.delete_path:
                await self._client.delete_progress(token, progress.delete_path)
        except NrkNotFoundError:
            self.logger.info("%s has no NRK progress to clear", item_id)
            return
        except _PROGRESS_ERRORS as err:
            self.logger.warning("Could not mark %s unplayed in NRK: %s", item_id, err)
            return
        self.logger.info("Marked %s unplayed in NRK", item_id)

    async def _read_progress(self, item_id: str) -> NrkProgress | None:
        """
        Return the NRK progress of one episode, None without an account or progress.

        :param item_id: The episode id, "<series id>|<episode id>".
        """
        if not self._auth.enabled:
            return None
        podcast_id, episode_id = _split_episode_id(item_id)
        try:
            user_id, token = await self._auth.get_token()
            return await self._client.get_progress(user_id, token, podcast_id, episode_id)
        except _PROGRESS_ERRORS as err:
            self._log_progress_failure("read", item_id, err)
            return None

    async def _read_podcast_progress(
        self, podcast_id: str, episodes: list[PodcastEpisode]
    ) -> dict[str, NrkProgress]:
        """
        Return the NRK progress of a podcast's episodes, keyed by NRK episode id.

        Empty without an account or when NRK fails, so MA's own resume takes over.

        :param podcast_id: The NRK series id of the podcast.
        :param episodes: The episodes to look up.
        """
        if not self._auth.enabled or not episodes:
            return {}
        episode_ids = [_split_episode_id(episode.item_id)[1] for episode in episodes]
        try:
            user_id, token = await self._auth.get_token()
            return await self._client.get_podcast_progress(user_id, token, podcast_id, episode_ids)
        except _PROGRESS_ERRORS as err:
            self._log_progress_failure("read", podcast_id, err)
            return {}

    def _log_progress_failure(self, action: str, item_id: str, err: Exception) -> None:
        """Warn once that NRK progress sync failed, then keep it at debug level."""
        if isinstance(err, NrkNotFoundError) and action == "read":
            # No stored progress for this episode yet: the normal case for a new episode.
            self.logger.debug("No NRK progress for %s", item_id)
            return
        if self._progress_warned:
            self.logger.debug("Could not %s NRK progress for %s: %s", action, item_id, err)
            return
        self._progress_warned = True
        self.logger.warning(
            "Could not %s NRK listening progress (%s); using Music Assistant's own resume "
            "position instead.",
            action,
            err,
        )

    def _browse_root(self) -> list[BrowseFolder]:
        folders = [
            self._folder(BROWSE_RADIOS, "Live radio", "radio_stations"),
            self._folder(BROWSE_PODCASTS, "Podcasts", "podcasts"),
        ]
        if self._auth.enabled:
            folders.append(self._folder(BROWSE_FOLLOWED, "Following", "followed_podcasts"))
        return folders

    def _folder(self, item_id: str, name: str, translation_key: str) -> BrowseFolder:
        return BrowseFolder(
            item_id=item_id,
            provider=self.instance_id,
            path=f"{self.instance_id}://{item_id}",
            name=name,
            translation_key=translation_key,
        )

    @use_cache(3600 * 6, base_class=Radio)
    async def _get_radios(self) -> list[Radio]:
        """Return all live NRK channels as Radio items."""
        return [self._radio_item(channel) for channel in await self._client.get_channels()]

    @use_cache(3600 * 6, base_class=Podcast)
    async def _get_curated_podcasts(self) -> list[Podcast]:
        """Return the podcasts NRK features on its podcast front page."""
        return [
            self._podcast_item(podcast) for podcast in await self._client.get_curated_podcasts()
        ]

    @use_cache(3600 * 6, base_class=PodcastEpisode)
    async def _fetch_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """
        Fetch a single episode from NRK's catalog, without user progress (it is cached).

        :param prov_episode_id: The episode id, "<series id>|<episode id>".
        """
        podcast_id, episode_id = _split_episode_id(prov_episode_id)
        episode = await self._client.get_episode(podcast_id, episode_id)
        podcast = await self.get_podcast(podcast_id)
        return self._episode_item(episode, self._podcast_mapping(podcast_id, podcast.name), 0)

    @use_cache(900, base_class=PodcastEpisode)
    async def _fetch_episodes(self, prov_podcast_id: str) -> list[PodcastEpisode]:
        """
        Fetch every episode of a podcast, numbered oldest to newest.

        :param prov_podcast_id: The NRK series id of the podcast.
        """
        podcast = await self.get_podcast(prov_podcast_id)
        mapping = self._podcast_mapping(prov_podcast_id, podcast.name)
        episodes = [ep async for ep in self._client.iter_episodes(prov_podcast_id)]
        positions = rank_episodes_by_date([ep.published for ep in episodes])
        return [
            self._episode_item(episode, mapping, position)
            for episode, position in zip(episodes, positions, strict=True)
        ]

    def _radio_item(self, channel: NrkChannel) -> Radio:
        radio = Radio(
            item_id=channel.id,
            provider=self.instance_id,
            name=channel.name,
            provider_mappings={
                ProviderMapping(
                    item_id=channel.id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )
        if channel.description:
            radio.metadata.description = channel.description
        if channel.image_url:
            radio.metadata.add_image(self._image(channel.image_url))
        return radio

    def _podcast_item(self, podcast: NrkPodcast) -> Podcast:
        item = Podcast(
            item_id=podcast.id,
            provider=self.instance_id,
            name=podcast.title,
            publisher="NRK",
            total_episodes=podcast.total_episodes,
            provider_mappings={
                ProviderMapping(
                    item_id=podcast.id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )
        if podcast.description:
            item.metadata.description = podcast.description
        if podcast.image_url:
            item.metadata.add_image(self._image(podcast.image_url))
        return item

    def _podcast_mapping(self, podcast_id: str, name: str) -> ItemMapping:
        return ItemMapping(
            media_type=MediaType.PODCAST,
            item_id=podcast_id,
            provider=self.instance_id,
            name=name,
        )

    def _episode_item(
        self, episode: NrkEpisode, podcast: ItemMapping, position: int
    ) -> PodcastEpisode:
        item_id = f"{episode.podcast_id}{EPISODE_ID_SEPARATOR}{episode.episode_id}"
        item = PodcastEpisode(
            item_id=item_id,
            provider=self.instance_id,
            name=episode.title,
            position=position,
            duration=episode.duration,
            podcast=podcast,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    available=episode.available,
                )
            },
        )
        if episode.description:
            item.metadata.description = episode.description
        if episode.image_url:
            item.metadata.add_image(self._image(episode.image_url))
        if episode.published:
            item.metadata.release_date = episode.published
        return item

    def _image(self, url: str) -> MediaItemImage:
        return MediaItemImage(
            type=ImageType.THUMB,
            path=url,
            provider=self.instance_id,
            remotely_accessible=True,
        )


def _with_progress(episode: PodcastEpisode, progress: NrkProgress | None) -> PodcastEpisode:
    """
    Return a copy of an episode carrying the user's NRK progress.

    A copy, because the episode may be the object the cache stores, and progress is
    per user and changes all the time. Without NRK progress the episode is returned
    as is, so MA falls back to its own play log.

    :param episode: The catalog episode.
    :param progress: The user's NRK progress on it, if any.
    """
    if progress is None or (not progress.fully_played and not progress.position):
        return episode
    return dataclasses.replace(
        episode,
        fully_played=progress.fully_played,
        resume_position_ms=0 if progress.fully_played else progress.position * 1000,
    )


def _find_nrk_episode(
    candidates: Sequence[PodcastEpisode], episode: PodcastEpisode
) -> PodcastEpisode | None:
    """
    Return the NRK episode of another provider's episode, else None.

    Matched on its id (NRK's RSS feeds use NRK's episode id as guid), then on its title.

    :param candidates: The podcast's NRK episodes.
    :param episode: The other provider's episode.
    """
    ids = {episode.item_id, *(mapping.item_id for mapping in episode.provider_mappings)}
    return next((c for c in candidates if _split_episode_id(c.item_id)[1] in ids), None) or next(
        (c for c in candidates if _same_title(c.name, episode.name)), None
    )


def _resume_target(
    progress: NrkProgress | None,
    position: int,
    last_report: tuple[int, datetime] | None,
) -> int | None:
    """
    Return where to seek to so playback continues from NRK's progress, or None to stay.

    NRK wins when its progress is newer than the last report from MA, which means it
    was listened to elsewhere, like the NRK Radio app. Positions that already agree
    need no seek, so resuming stays instant.

    :param progress: NRK's stored progress of the episode.
    :param position: Where MA resumed, in seconds.
    :param last_report: The last (position, time) this provider reported for it.
    """
    if progress is None or progress.fully_played or not progress.position:
        return None
    if last_report is not None:
        last_position, reported_at = last_report
        if progress.position == last_position:
            return None  # NRK holds what MA wrote itself
        if progress.updated is None or progress.updated <= reported_at + _CLOCK_SKEW:
            return None  # MA was listened to last
    if abs(progress.position - position) <= _SAME_POSITION:
        return None
    return progress.position


def _is_finished(position: int, duration: int) -> bool:
    """
    Return True when a position counts as having finished the episode.

    :param position: The position in seconds.
    :param duration: The length of the episode in seconds, 0 when unknown.
    """
    if duration <= 0:
        return False
    if position >= duration * _FINISHED_SHARE:
        return True
    # The tail rule only for episodes long enough that it does not finish them at once.
    return duration > _FINISHED_TAIL * 2 and position >= duration - _FINISHED_TAIL


def _same_title(left: str, right: str) -> bool:
    """Return True when two episode titles are the same, ignoring case and spacing."""
    return " ".join(left.casefold().split()) == " ".join(right.casefold().split())


def _split_episode_id(item_id: str) -> tuple[str, str]:
    """
    Split an episode id into its NRK series id and episode id.

    :param item_id: The episode id, "<series id>|<episode id>".
    """
    podcast_id, sep, episode_id = item_id.partition(EPISODE_ID_SEPARATOR)
    if not sep or not podcast_id or not episode_id:
        raise MediaNotFoundError(f"Invalid NRK episode id: {item_id}")
    return podcast_id, episode_id
