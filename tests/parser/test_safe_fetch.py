"""Outbound fetches use controlled DNS and transports, never the Internet."""

import socket

import httpx
import pytest

from parser import safe_fetch


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.org/a",
        "http://127.1/",
        "http://2130706433/",
        "http://0x7f000001/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://169.254.169.254/",
        "http://10.0.0.1/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://user:password@example.org/",
    ],
)
def test_unsafe_urls_rejected(url, monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(safe_fetch.UnsafeURLError):
        safe_fetch.fetch_url(url)


def test_mixed_dns_rejected(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))
            for ip in ["93.184.216.34", "192.168.0.1"]
        ],
    )
    with pytest.raises(safe_fetch.UnsafeURLError):
        safe_fetch.fetch_url("https://example.org/")


@pytest.mark.parametrize("async_mode", [False, True])
async def test_pins_dns_and_checks_redirect(monkeypatch, async_mode):
    calls = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def respond(request):
        calls.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.org"
        assert request.extensions["sni_hostname"] == "example.org"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/"})

    async def async_respond(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, r: respond(r))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    with pytest.raises(safe_fetch.UnsafeURLError):
        if async_mode:
            await safe_fetch.fetch_url_async("https://example.org/")
        else:
            safe_fetch.fetch_url("https://example.org/")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "address",
    [
        "0.0.0.0",
        "127.0.0.1",
        "10.1.1.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.100.100.200",
        "224.0.0.1",
        "::",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::ffff:192.168.1.1",
        "64:ff9b::7f00:1",
        "2002:7f00:1::",
        "2001::1",
    ],
)
def test_non_public_dns(address, monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, 443),
            )
        ],
    )
    with pytest.raises(safe_fetch.UnsafeURLError):
        safe_fetch.fetch_url("https://example.org/")


@pytest.mark.parametrize("async_mode", [False, True])
async def test_public_redirect_and_body_limit(monkeypatch, async_mode):
    calls = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def respond(request):
        calls.append(request)
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/end"})
        return httpx.Response(200, content=b"public page")

    async def async_respond(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, r: respond(r))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    if async_mode:
        result = await safe_fetch.fetch_url_async("https://example.org/start")
    else:
        result = safe_fetch.fetch_url("https://example.org/start")
    assert result.text == "public page"
    assert str(result.url) == "https://example.org/end"
    assert len(calls) == 2
    monkeypatch.setattr(safe_fetch, "MAX_BYTES", 2)
    with pytest.raises(safe_fetch.UnsafeURLError, match="size limit"):
        if async_mode:
            await safe_fetch.fetch_url_async("https://example.org/end")
        else:
            safe_fetch.fetch_url("https://example.org/end")


async def test_all_download_entrypoints_reject_private_url():
    from fastapi import HTTPException
    from pydantic import HttpUrl

    from api.routes.rag import parse_page_for_rag, upload_csv_documents
    from api.services.document_update import fetch_url_bytes
    from parser.url import process_url

    with pytest.raises(safe_fetch.UnsafeURLError):
        await fetch_url_bytes("http://127.0.0.1/file.pdf")
    assert await process_url("http://127.0.0.1/file.pdf") is None
    with pytest.raises(HTTPException) as error:
        await parse_page_for_rag(HttpUrl("http://127.0.0.1/"))
    assert error.value.status_code == 400

    import io

    from fastapi import UploadFile

    result = await upload_csv_documents(
        UploadFile(filename="sources.csv", file=io.BytesIO(b"Link\nhttp://127.0.0.1/\n"))
    )
    assert result.imported_count == 0


@pytest.mark.parametrize(
    "trusted,hostname",
    [
        (False, "example.org"),
        (True, "example.org"),
        (True, "wrong.example"),
    ],
)
def test_real_tls_verification_and_hostname(tmp_path, monkeypatch, trusted, hostname):
    import datetime
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example.org")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("example.org")]), False)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.headers["Host"].startswith(hostname)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"verified")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", server.server_port),
            )
        ],
    )
    destinations = []

    def connect(address, timeout=None, source_address=None, **kwargs):
        destinations.append(address)
        connection = socket.socket()
        connection.settimeout(timeout)
        connection.connect(("127.0.0.1", server.server_port))
        return connection

    monkeypatch.setattr(socket, "create_connection", connect)
    if trusted:
        original_context = ssl.create_default_context

        def trusted_context(*args, **kwargs):
            context = original_context(*args, **kwargs)
            context.load_verify_locations(cert_path)
            return context

        monkeypatch.setattr(ssl, "create_default_context", trusted_context)
    try:
        url = f"https://{hostname}:{server.server_port}/"
        if trusted and hostname == "example.org":
            assert safe_fetch.fetch_url(url).text == "verified"
        else:
            with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
                safe_fetch.fetch_url(url)
        assert destinations == [("93.184.216.34", server.server_port)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001", "0177.0.0.1"])
def test_alternate_ipv4_uses_system_numeric_resolution(host):
    # Numeric libc resolution is local and does not consult an external DNS server.
    with pytest.raises(safe_fetch.UnsafeURLError):
        safe_fetch._pin_url(f"http://{host}/")


@pytest.mark.parametrize("async_mode", [False, True])
async def test_rebinding_on_same_host_redirect_is_revalidated(monkeypatch, async_mode):
    answers = iter(["93.184.216.34", "127.0.0.1"])
    sent = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), 443))
        ],
    )

    def respond(request):
        sent.append(request)
        return httpx.Response(302, headers={"location": "/next"})

    async def async_respond(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, r: respond(r))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    with pytest.raises(safe_fetch.UnsafeURLError):
        if async_mode:
            await safe_fetch.fetch_url_async("https://example.org/")
        else:
            safe_fetch.fetch_url("https://example.org/")
    assert len(sent) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_redirect_loop_is_bounded(monkeypatch, async_mode):
    sent = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def respond(request):
        sent.append(request)
        return httpx.Response(302, headers={"location": "/again"})

    async def async_respond(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, r: respond(r))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    with pytest.raises(safe_fetch.UnsafeURLError, match="Too many"):
        if async_mode:
            await safe_fetch.fetch_url_async("https://example.org/")
        else:
            safe_fetch.fetch_url("https://example.org/")
    assert len(sent) == safe_fetch.MAX_REDIRECTS + 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_public_ipv6_and_compressed_response(monkeypatch, async_mode):
    import gzip

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("2606:4700:4700::1111", 443, 0, 0),
            )
        ],
    )

    def respond(request):
        assert request.url.host == "2606:4700:4700::1111"
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=gzip.compress(b"public page"),
        )

    async def async_respond(self, request):
        return respond(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, r: respond(r))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    if async_mode:
        response = await safe_fetch.fetch_url_async("https://example.org/")
    else:
        response = safe_fetch.fetch_url("https://example.org/")
    assert response.text == "public page"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_redirect_hosts_never_share_tls_pool(monkeypatch, async_mode):
    transports = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def respond(transport, request):
        transports.append(transport)
        if len(transports) == 1:
            assert request.extensions["sni_hostname"] == "example.org"
            return httpx.Response(302, headers={"location": "https://other.example/"})
        assert request.extensions["sni_hostname"] == "other.example"
        return httpx.Response(200, content=b"public")

    async def async_respond(self, request):
        return respond(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_respond)
    if async_mode:
        assert (await safe_fetch.fetch_url_async("https://example.org/")).text == "public"
    else:
        assert safe_fetch.fetch_url("https://example.org/").text == "public"
    assert transports[0] is not transports[1]


async def test_preview_accepts_public_server_without_head(monkeypatch):
    from pydantic import HttpUrl

    from api.routes import rag

    methods = []

    async def fetch(url, *, method="GET", **kwargs):
        methods.append(method)
        request = httpx.Request(method, url)
        if method == "HEAD":
            response = httpx.Response(405, request=request)
            response.raise_for_status()
        return httpx.Response(200, text="public page", request=request)

    async def clean(text):
        return text

    monkeypatch.setattr(rag, "fetch_url_async", fetch)
    monkeypatch.setattr(
        rag,
        "parse_page",
        lambda *a, **k: {
            "title": "public",
            "content_blocks": ["public page"],
            "documents": [],
        },
    )
    monkeypatch.setattr(rag, "clean_and_structure_text", clean)
    monkeypatch.setattr(rag, "generate_title_from_text", clean)
    result = await rag.parse_page_for_rag(HttpUrl("https://example.org/"))
    assert "public page" in result.text
    assert methods == ["HEAD", "GET"]
