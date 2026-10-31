"""Bounded public HTTP downloads with DNS pinning and verified TLS."""

import asyncio
import ipaddress
import socket
from collections.abc import Mapping

import httpx

MAX_BYTES = 25 * 1024 * 1024
MAX_REDIRECTS = 5
_REDIRECTS = {301, 302, 303, 307, 308}


class UnsafeURLError(ValueError):
    """The source is not a public HTTP(S) resource."""


def _public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            return _public_address(str(address.ipv4_mapped))
        # Transition mechanisms can route to an embedded non-public IPv4 address.
        if (
            address.sixtofour
            or address.teredo
            or address not in ipaddress.IPv6Network("2000::/3")
        ):
            return False
    return address.is_global and not address.is_multicast


def _pin_url(value: str) -> tuple[httpx.URL, httpx.URL]:
    if any(ord(char) < 33 for char in value) or "\\" in value:
        raise UnsafeURLError("URL contains invalid characters")
    try:
        url = httpx.URL(value)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise UnsafeURLError(
                "Only public HTTP(S) URLs without credentials are allowed"
            )
        if "%" in url.host:
            raise UnsafeURLError("Scoped addresses are not allowed")
        try:
            literal = ipaddress.ip_address(url.host)
        except ValueError:
            literal = None
        if literal is not None and not _public_address(str(literal)):
            raise UnsafeURLError("URL must resolve only to public Internet addresses")
        addresses = socket.getaddrinfo(
            url.host,
            url.port or (443 if url.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
        ips = [str(result[4][0]) for result in addresses]
        if not ips or not all(_public_address(ip) for ip in ips):
            raise UnsafeURLError("URL must resolve only to public Internet addresses")
        return url, url.copy_with(host=ips[0])
    except (httpx.InvalidURL, socket.gaierror, ValueError) as exc:
        if isinstance(exc, UnsafeURLError):
            raise
        raise UnsafeURLError("Invalid URL or source DNS address") from exc


def _request(
    client: httpx.Client | httpx.AsyncClient,
    method: str,
    original: httpx.URL,
    pinned: httpx.URL,
    headers: Mapping[str, str] | None,
) -> httpx.Request:
    # Never forward credentials or override the validated target with caller headers.
    safe_headers = {
        key: value
        for key, value in (headers or {}).items()
        if key.lower() not in {"host", "authorization", "cookie", "proxy-authorization"}
    }
    safe_headers["Host"] = original.netloc.decode("ascii")
    client.cookies.clear()
    request = client.build_request(method, pinned, headers=safe_headers)
    request.extensions["sni_hostname"] = original.host
    return request


def _next_url(response: httpx.Response, original: httpx.URL, hop: int) -> str | None:
    if response.status_code in _REDIRECTS:
        if hop == MAX_REDIRECTS:
            raise UnsafeURLError("Too many source redirects")
        location = response.headers.get("location")
        if not location:
            raise UnsafeURLError("Source redirect has no location")
        return str(original.join(location))
    response.raise_for_status()
    return None


def fetch_url(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> httpx.Response:
    """Fetch at most 25 MiB through at most five validated redirects."""
    for hop in range(MAX_REDIRECTS + 1):
        # The pool is keyed by pinned IP, so never reuse TLS across redirect hosts.
        with httpx.Client(verify=True, trust_env=False, timeout=timeout) as client:
            original, pinned = _pin_url(url)
            request = _request(client, method, original, pinned, headers)
            response = client.send(request, stream=True)
            try:
                next_url = _next_url(response, original, hop)
                if next_url is not None:
                    url = next_url
                    continue
                chunks = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise UnsafeURLError("Source exceeds download size limit")
                    chunks.append(chunk)
                return httpx.Response(
                    response.status_code,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() not in {"content-encoding", "content-length"}
                    },
                    content=b"".join(chunks),
                    request=httpx.Request(method, original),
                )
            finally:
                response.close()
    raise UnsafeURLError("Too many source redirects")


async def fetch_url_async(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> httpx.Response:
    """Async equivalent of :func:`fetch_url`, including DNS pinning."""
    for hop in range(MAX_REDIRECTS + 1):
        async with httpx.AsyncClient(
            verify=True, trust_env=False, timeout=timeout
        ) as client:
            original, pinned = await asyncio.to_thread(_pin_url, url)
            request = _request(client, method, original, pinned, headers)
            response = await client.send(request, stream=True)
            try:
                next_url = _next_url(response, original, hop)
                if next_url is not None:
                    url = next_url
                    continue
                chunks = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise UnsafeURLError("Source exceeds download size limit")
                    chunks.append(chunk)
                return httpx.Response(
                    response.status_code,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() not in {"content-encoding", "content-length"}
                    },
                    content=b"".join(chunks),
                    request=httpx.Request(method, original),
                )
            finally:
                await response.aclose()
    raise UnsafeURLError("Too many source redirects")
