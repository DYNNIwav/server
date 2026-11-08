"""
NRK Radio provider for Music Assistant.

Live radio channels and podcasts from NRK's public radio API play without an account.
With an NRK login, the podcasts the user follows sync to the library and episode
progress is read from and written to NRK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import NrkProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Set up the NRK Radio provider."""
    return NrkProvider(mass, manifest, config)
