"""Constants for the NRK Radio provider."""

from __future__ import annotations

import aiohttp

API_URL = "https://psapi.nrk.no"
API_HEADERS = {"Accept": "application/json;api-version=3.5"}
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
REQUEST_RATE_LIMIT = 5  # requests per second
SEARCH_TIMEOUT = 10  # seconds
EPISODE_PAGE_SIZE = 50  # the most NRK accepts, larger sizes return HTTP 400
FAVOURITES_PAGE_SIZE = 50
PROGRESS_BATCH_SIZE = 50  # episode ids per progress lookup, NRK's maximum

LOGIN_URL = "https://radio.nrk.no"
AUTH_URL = "https://innlogging.nrk.no"
AUTH_RETURN_URL = "https://radio.nrk.no/mittinnhold"
AUTH_CLIENT_ID = "radio.nrk.no.web"
AUTH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
# Refresh the access token this many seconds before NRK says it expires.
TOKEN_EXPIRY_MARGIN = 120

BROWSE_RADIOS = "radio"
BROWSE_PODCASTS = "podcasts"
BROWSE_FOLLOWED = "followed"

# Episode ids carry their podcast id, as in NRK's own content ids ("series|episode").
EPISODE_ID_SEPARATOR = "|"
