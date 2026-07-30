import base64
import io
import sys

import matplotlib.pyplot as plt
import pytest
from chatlas import content_image_file, content_image_plot, content_image_url
from chatlas._content_image import ContentImageInline, ContentImageRemote
from PIL import Image


def test_can_create_image_from_url():
    obj = content_image_url("https://www.r-project.org/Rlogo.png")
    assert isinstance(obj, ContentImageRemote)


def test_can_create_inline_image_from_data_url():
    obj = content_image_url("data:image/png;base64,abcd")
    assert isinstance(obj, ContentImageInline)
    assert obj.image_content_type == "image/png"
    assert obj.data == "abcd"


def test_errors_with_invalid_data_urls():
    with pytest.raises(ValueError):
        content_image_url("data:base64,abcd")

    with pytest.raises(ValueError):
        content_image_url("data:")

    with pytest.raises(ValueError):
        content_image_url("data:;;;")

    with pytest.raises(ValueError):
        content_image_url("data:image/png;abc")


def test_can_create_image_from_path(tmp_path):
    # Create a test image
    img = Image.new("RGB", (60, 30), color="red")
    path = tmp_path / "test.png"
    img.save(path)

    obj = content_image_file(str(path), resize="low")
    assert isinstance(obj, ContentImageInline)


def test_can_create_image_from_plot():
    plt.figure()
    plt.plot([1, 2, 3])

    obj = content_image_plot()
    assert isinstance(obj, ContentImageInline)
    assert obj.image_content_type == "image/png"

    plt.close()


def test_image_resizing(tmp_path):
    # Create a test image
    img = Image.new("RGB", (60, 30), color="red")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    with pytest.raises(FileNotFoundError):
        content_image_file("DOESNTEXIST")

    with pytest.raises(FileNotFoundError):
        content_image_file(str(tmp_path / "test.txt"))

    # Test valid resize options
    with pytest.warns(RuntimeWarning):
        assert content_image_file(str(img_path)) is not None
    assert content_image_file(str(img_path), resize="low") is not None
    assert content_image_file(str(img_path), resize="high") is not None
    assert content_image_file(str(img_path), resize="none") is not None
    assert content_image_file(str(img_path), resize="100x100") is not None
    assert content_image_file(str(img_path), resize="100x100>!") is not None


def test_useful_errors_if_no_display():
    plt.close("all")  # Close all plots
    with pytest.raises(RuntimeError, match="No matplotlib figure to save"):
        content_image_plot()


def test_can_create_inline_heic_image_from_data_url():
    obj = content_image_url("data:image/heic;base64,abcd")
    assert isinstance(obj, ContentImageInline)
    assert obj.image_content_type == "image/heic"


def test_can_create_inline_heif_image_from_data_url():
    obj = content_image_url("data:image/heif;base64,abcd")
    assert obj.image_content_type == "image/heif"


def test_heic_extension_maps_to_heic_mime_type(tmp_path):
    path = tmp_path / "photo.heic"
    path.write_bytes(b"fake-heic-bytes")
    obj = content_image_file(str(path), resize="none")
    assert obj.image_content_type == "image/heic"


def test_heif_extension_maps_to_heif_mime_type(tmp_path):
    path = tmp_path / "photo.heif"
    path.write_bytes(b"fake-heif-bytes")
    obj = content_image_file(str(path), resize="none")
    assert obj.image_content_type == "image/heif"


def test_heic_resize_without_pillow_heif_raises_clear_error(tmp_path, monkeypatch):
    # pillow-heif is a dev dependency, so simulate its absence rather than
    # relying on it being uninstalled -- otherwise this passes for the wrong
    # reason locally and stops testing anything once it's installed.
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    path = tmp_path / "photo.heic"
    path.write_bytes(b"fake-heic-bytes")
    with pytest.raises(ImportError, match="pillow-heif"):
        content_image_file(str(path), resize="low")


@pytest.mark.parametrize("suffix", [".heic", ".heif"])
def test_heic_heif_resize_roundtrips_through_pillow_heif(tmp_path, suffix):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()

    path = tmp_path / f"photo{suffix}"
    Image.new("RGB", (1200, 900), (200, 50, 50)).save(str(path), format="HEIF")

    obj = content_image_file(str(path), resize="low")

    assert obj.image_content_type == f"image/{suffix.lstrip('.')}"
    resized = Image.open(io.BytesIO(base64.b64decode(obj.data)))
    # `resize="low"` thumbnails into a 512x512 box, preserving aspect ratio.
    assert resized.size == (512, 384)
    # The resized bytes must still be HEIF; `img.save(format=img.format)` only
    # works because register_heif_opener() also registers the encoder.
    assert resized.format == "HEIF"
