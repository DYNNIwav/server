"""Typed data models and error types for the NRK Radio provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    MusicAssistantError,
    ResourceTemporarilyUnavailable,
)

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class NrkChannel:
    """A live NRK radio channel."""

    id: str
    name: str
    description: str | None = None
    image_url: str | None = None


@dataclass(frozen=True, slots=True)
class NrkPodcast:
    """An NRK podcast series."""

    id: str
    title: str
    description: str | None = None
    image_url: str | None = None
    total_episodes: int | None = None


@dataclass(frozen=True, slots=True)
class NrkEpisode:
    """A single podcast episode."""

    podcast_id: str
    episode_id: str
    title: str
    description: str | None = None
    image_url: str | None = None
    duration: int = 0  # seconds
    published: datetime | None = None
    available: bool = True


@dataclass(frozen=True, slots=True)
class NrkStream:
    """The playable asset from a playback manifest."""

    url: str
    content_type: str  # "mp3" or "aac"
    is_live: bool
    duration: int = 0  # seconds


@dataclass(frozen=True, slots=True)
class NrkProgress:
    """The user's listening progress on one episode."""

    fully_played: bool
    position: int  # seconds
    updated: datetime | None = None
    # NRK's link for deleting this progress (marking the episode unplayed), if any.
    delete_path: str | None = None


class NrkApiError(ResourceTemporarilyUnavailable):
    """Raised on an NRK API transport or protocol error (network, HTTP, bad payload)."""


class NrkRequestError(MusicAssistantError):
    """Raised when NRK refuses a request (HTTP 4xx), which a retry would not fix."""


class NrkNotFoundError(MediaNotFoundError):
    """Raised when requested NRK content does not exist."""


class NrkAuthError(LoginFailed):
    """Raised when NRK rejects the configured credentials."""
