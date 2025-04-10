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
