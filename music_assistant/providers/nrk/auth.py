"""NRK login: a saved radio.nrk.no session (or email and password) to a bearer token."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

import aiohttp
from yarl import URL

from music_assistant.helpers.aiohttp_client import create_clientsession

from .constants import (
    AUTH_CLIENT_ID,
    AUTH_RETURN_URL,
    AUTH_URL,
    AUTH_USER_AGENT,
    LOGIN_URL,
    REQUEST_TIMEOUT,
    TOKEN_EXPIRY_MARGIN,
)
from .models import NrkApiError, NrkAuthError

if TYPE_CHECKING:
    import logging

    from music_assistant.mass import MusicAssistant

# NRK asks the client to scrypt-hash the password, "cscrypt:<log2 N>:<r>:<p>:<dkLen>".
_SCRYPT_MAXMEM = 256 * 1024 * 1024
# A login NRK refused is tried again after this many seconds, not on every request.
_RETRY_REFUSED_AFTER = 3600
_LOGIN_NEEDED = (
    "NRK login needed: the saved NRK session is no longer valid, paste a new one in the settings"
)


class NrkAuth:
    """
    Keep a valid NRK bearer token, from a saved radio.nrk.no session or a password login.

    NRK's login page asks for a captcha, so the password login (ported from
    bendikrb/nrk-psapi, MIT) mostly fails now. The dependable way in is the session a
    person logged in with in a browser: its radio.nrk.no cookies are exchanged for a
    token at NRK's tokenforsub step, the same call radio.nrk.no makes itself. Cookies
    NRK renews on the way are handed to save_session, so the session outlives restarts.
    Only a session or credentials NRK refuses raise NrkAuthError; everything else is
    temporary and raises NrkApiError so MA retries. A refused login is tried again
    after an hour.
    """

    def __init__(
        self,
        mass: MusicAssistant,
        logger: logging.Logger,
        email: str,
        password: str,
        session: str = "",
        save_session: Callable[[str], None] | None = None,
        report_login: Callable[[NrkAuthError | NrkApiError | None], None] | None = None,
    ) -> None:
        """
        Initialize the auth manager.

        :param mass: The MusicAssistant instance (used to build the login session).
        :param logger: Logger for diagnostics.
        :param email: NRK account email (empty without a password login).
        :param password: NRK account password.
        :param session: Saved radio.nrk.no cookies, as a Cookie header ("a=1; b=2").
        :param save_session: Called with the cookies whenever NRK renews them.
        :param report_login: Called after each login or refresh with its error, None when it worked.
        """
        self._mass = mass
        self._logger = logger
        self._email = email
        self._password = password
        self._lock = asyncio.Lock()
        self._jar = aiohttp.CookieJar()
        self._jar.update_cookies(_parse_cookies(session), URL(LOGIN_URL))
        self._session = _session_header(self._jar)
        self._save_session = save_session
        self._report_login = report_login or (lambda _err: None)
        self._access_token: str | None = None
        self._user_id: str | None = None
        self._expiry: float = 0.0
        self._rejected: NrkAuthError | None = None
        self._rejected_at = 0.0

    @property
    def enabled(self) -> bool:
        """Return True when a session or credentials are configured."""
        return bool(self._session or (self._email and self._password))

    async def get_token(self) -> tuple[str, str]:
        """Return a valid (user id, bearer token), logging in or refreshing as needed."""
        if not self.enabled:
            raise NrkAuthError("An NRK account is required")
        async with self._lock:
            if self._rejected:
                if monotonic() - self._rejected_at < _RETRY_REFUSED_AFTER:
                    raise NrkAuthError(str(self._rejected))
                self._rejected = None
            if not self._token_valid():
                try:
                    await self._refresh_or_login()
                except NrkAuthError as err:
                    self._rejected = err
                    self._rejected_at = monotonic()
                    self._report_login(err)
                    raise
                except NrkApiError as err:
                    self._report_login(err)
                    raise
                self._report_login(None)
                self._keep_session()
            assert self._access_token is not None
            assert self._user_id is not None
            return self._user_id, self._access_token

    def _token_valid(self) -> bool:
        return bool(self._access_token) and time.time() < self._expiry - TOKEN_EXPIRY_MARGIN

    async def _refresh_or_login(self) -> None:
        """Refresh on the NRK session, falling back to a password login when there is one."""
        headers = {"User-Agent": AUTH_USER_AGENT}
        can_login = bool(self._email and self._password)
        try:
            async with create_clientsession(
                self._mass, cookie_jar=self._jar, headers=headers, timeout=REQUEST_TIMEOUT
            ) as session:
                if self._access_token or self._session:
                    try:
                        self._store(await _token_for_sub(session))
                        return
                    except (NrkApiError, NrkAuthError) as err:
                        if not can_login:
                            raise
                        self._logger.debug("NRK token refresh failed, logging in again: %s", err)
                self._jar.clear()
                self._store(await self._login(session))
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            raise NrkApiError(f"NRK login request failed: {err}") from err
        self._logger.debug("Logged in to NRK")

    async def _login(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        """Run the full NRK web login and return the token response."""
        async with session.get(
            f"{LOGIN_URL}/auth/web/login", params={"returnUrl": AUTH_RETURN_URL}
        ) as resp:
            await _check(resp)
            if not resp.history:
                raise NrkApiError("NRK login did not redirect to the login page")
            callback_url = resp.history[-1].url
        async with session.post(
            f"{AUTH_URL}/getHashingInstructions", json={"email": self._email}
        ) as resp:
            await _check(resp)
            recipe = ((await resp.json(content_type=None)) or {}).get("current") or {}
        hashed = await asyncio.to_thread(hash_password, self._password, recipe)
        payload = {
            "username": self._email,
            "password": self._password,
            "hashedPassword": {"current": {"recipe": recipe, "hash": hashed}, "next": None},
            "clientId": AUTH_CLIENT_ID,
            "addUser": False,
        }
        async with session.post(
            f"{AUTH_URL}/logginn",
            params={"encodedExitUrl": quote_plus(AUTH_RETURN_URL)},
            json=payload,
        ) as resp:
            await _check(resp, login=True)
        async with session.get(
            f"{AUTH_URL}/connect/authorize/callback", params=dict(callback_url.query)
        ) as resp:
            await _check(resp)
        return await _token_for_sub(session)

    def _keep_session(self) -> None:
        """Hand the session cookies to save_session when NRK renewed them."""
        header = _session_header(self._jar)
        if not header or header == self._session:
            return
        self._session = header
        if self._save_session:
            self._save_session(header)
            self._logger.debug("Saved the renewed NRK session")

    def _store(self, data: dict[str, Any]) -> None:
        """Keep the bearer token, user id and expiry from a token response."""
        session = data.get("session") if isinstance(data, dict) else None
        if not isinstance(session, dict):
            # NRK answers without a session when the cookies no longer log anyone in.
            raise NrkAuthError(_LOGIN_NEEDED)
        token = session.get("accessToken")
        user_id = (session.get("user") or {}).get("sub")
        expiry = session.get("serverEpochExpiry")
        if not token or not user_id or not isinstance(expiry, (int, float)):
            raise NrkApiError("NRK token response is missing the token, user or expiry")
        self._access_token = token
        self._user_id = user_id
        self._expiry = float(expiry)


def hash_password(password: str, recipe: dict[str, Any]) -> str:
    """
    Hash a password the way NRK's login page does and return it as hex.

    :param password: The plain password.
    :param recipe: NRK's hashing instructions, e.g. {"algorithm": "cscrypt:17:8:1:32", "salt": ...}.
    """
    algorithm = str(recipe.get("algorithm") or "")
    parts = algorithm.split(":")
    salt = recipe.get("salt")
    if len(parts) != 5 or parts[0] != "cscrypt" or not isinstance(salt, str):
        raise NrkApiError(f"Unsupported NRK password hashing: {algorithm!r}")
    log_n, r, p, dklen = (int(part) for part in parts[1:])
    # The salt is used as the literal string NRK sends, not base64-decoded.
    digest = hashlib.scrypt(
        password.encode(),
        salt=salt.encode(),
        n=1 << log_n,
        r=r,
        p=p,
        dklen=dklen,
        maxmem=_SCRYPT_MAXMEM,
    )
    return digest.hex()


async def _token_for_sub(session: aiohttp.ClientSession) -> dict[str, Any]:
    """Exchange the NRK login cookies for a token response."""
    async with session.post(f"{LOGIN_URL}/auth/csrf_init") as resp:
        await _check(resp)
    async with session.get(f"{LOGIN_URL}/auth/contextinfo") as resp:
        await _check(resp)
    async with session.post(f"{LOGIN_URL}/auth/session/tokenforsub/_") as resp:
        if resp.status in (401, 403):
            raise NrkAuthError(_LOGIN_NEEDED)
        await _check(resp)
        data = await resp.json(content_type=None)
    if not isinstance(data, dict):
        raise NrkApiError("Unexpected NRK token response")
    return data


def _parse_cookies(header: str) -> dict[str, str]:
    """Return the cookies of a Cookie header ("a=1; b=2") by name."""
    cookies = {}
    for part in header.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name:
            cookies[name] = value
    return cookies


def _session_header(jar: aiohttp.CookieJar) -> str:
    """Return the jar's radio.nrk.no cookies as a Cookie header, sorted by name."""
    cookies = {}
    for morsel in jar:
        domain = morsel["domain"].lstrip(".")
        if domain in ("radio.nrk.no", "nrk.no"):
            cookies[morsel.key] = morsel.value
    return "; ".join(f"{name}={value}" for name, value in sorted(cookies.items()))


async def _check(resp: aiohttp.ClientResponse, login: bool = False) -> None:
    """
    Raise the matching error for a failed NRK auth response.

    :param resp: The response to check.
    :param login: True for the password step, the only one that can reject credentials.
    """
    if resp.status < 400:
        return
    if login and resp.status in (400, 401, 403):
        try:
            problem = await resp.json(content_type=None)
        except ValueError:
            problem = None
        if isinstance(problem, dict) and problem.get("type") == "captcha-validation-error":
            raise NrkAuthError(
                "NRK asks for a captcha at login, which Music Assistant cannot solve"
            )
        raise NrkAuthError(f"NRK rejected the login (HTTP {resp.status})")
    raise NrkApiError(f"NRK login step failed (HTTP {resp.status})")
