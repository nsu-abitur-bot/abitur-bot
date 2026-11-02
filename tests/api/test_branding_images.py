from io import BytesIO

import pytest
from PIL import Image

from api.services.branding import prepare_logo


def test_rejects_active_and_oversized_content():
    for content in (
        b'<svg xmlns="http://www.w3.org/2000/svg"/>',
        b"x" * (2 * 1024 * 1024 + 1),
    ):
        with pytest.raises(ValueError):
            prepare_logo(content)


def test_prepares_square_icon_and_strips_metadata():
    source = BytesIO()
    Image.new("RGBA", (400, 100), "red").save(source, format="PNG")
    logo, icon = prepare_logo(source.getvalue())
    assert Image.open(BytesIO(logo)).size == (400, 100)
    favicon = Image.open(BytesIO(icon))
    assert favicon.size == (64, 64)
    pixel = favicon.getpixel((0, 0))
    assert isinstance(pixel, tuple) and pixel[3] == 0


def test_checks_dimensions_before_decode():
    source = BytesIO()
    Image.new("RGB", (4097, 1)).save(source, format="PNG")
    with pytest.raises(ValueError):
        prepare_logo(source.getvalue())


def test_storage_failure_removes_only_new_files(tmp_path, monkeypatch):
    from api.services.branding import store_logo

    monkeypatch.setenv("BRANDING_MEDIA_DIR", str(tmp_path))
    old = tmp_path / "existing.png"
    old.write_bytes(b"working-logo")
    source = BytesIO()
    Image.new("RGB", (20, 20)).save(source, format="JPEG")

    def fail_sync(fd):
        raise OSError("full disk")

    monkeypatch.setattr("api.services.branding.os.fsync", fail_sync)
    with pytest.raises(OSError):
        store_logo(source.getvalue())
    assert list(tmp_path.iterdir()) == [old]
    assert old.read_bytes() == b"working-logo"
