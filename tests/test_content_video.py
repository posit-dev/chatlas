import base64

import pytest
from chatlas import content_video_file, content_video_youtube
from chatlas._content import ContentVideoInline, ContentVideoUrl, create_content
from chatlas._turn import Turn, UserTurn


def test_can_create_video_from_local_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x01fake mp4 bytes\x02")

    obj = content_video_file(path)
    assert isinstance(obj, ContentVideoInline)
    assert obj.filename == "clip.mp4"
    assert obj.video_content_type == "video/mp4"
    assert base64.b64decode(obj.data) == b"\x00\x01fake mp4 bytes\x02"


@pytest.mark.parametrize(
    ("suffix", "mime_type"),
    [
        (".mp4", "video/mp4"),
        (".mpeg", "video/mpeg"),
        (".mpg", "video/mpg"),
        (".mov", "video/mov"),
        (".avi", "video/avi"),
        (".flv", "video/x-flv"),
        (".webm", "video/webm"),
        (".wmv", "video/wmv"),
        (".3gpp", "video/3gpp"),
        (".3gp", "video/3gpp"),
    ],
)
def test_content_video_file_guesses_mime_type_from_extension(
    tmp_path, suffix, mime_type
):
    path = tmp_path / f"clip{suffix}"
    path.write_bytes(b"data")
    assert content_video_file(path).video_content_type == mime_type


def test_content_video_file_explicit_mime_type_overrides_guess(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"data")
    obj = content_video_file(path, mime_type="video/webm")
    assert obj.video_content_type == "video/webm"


def test_content_video_file_unknown_extension_raises(tmp_path):
    path = tmp_path / "clip.xyz"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Couldn't guess a video MIME type"):
        content_video_file(path)


def test_content_video_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        content_video_file(tmp_path / "nope.mp4")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=9hE5-98ZeCg",
        "https://youtu.be/9hE5-98ZeCg",
        "https://m.youtube.com/watch?v=9hE5-98ZeCg",
        "https://youtube.com/shorts/9hE5-98ZeCg",
    ],
)
def test_content_video_youtube_accepts_youtube_urls(url):
    obj = content_video_youtube(url)
    assert isinstance(obj, ContentVideoUrl)
    assert obj.url == url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/video.mp4",
        "not a url",
        "ftp://youtube.com/watch?v=abc",
    ],
)
def test_content_video_youtube_rejects_non_youtube_urls(url):
    with pytest.raises(ValueError, match="doesn't look like a YouTube"):
        content_video_youtube(url)


def test_video_inline_round_trip():
    video = ContentVideoInline(
        video_content_type="video/mp4", data="ZmFrZQ==", filename="clip.mp4"
    )
    turn = UserTurn([video])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert isinstance(restored.contents[0], ContentVideoInline)
    assert restored.contents[0].data == "ZmFrZQ=="
    assert restored.contents[0].filename == "clip.mp4"


def test_video_url_round_trip():
    video = ContentVideoUrl(url="https://www.youtube.com/watch?v=9hE5-98ZeCg")
    turn = UserTurn([video])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert isinstance(restored.contents[0], ContentVideoUrl)
    assert restored.contents[0].url == "https://www.youtube.com/watch?v=9hE5-98ZeCg"


def test_create_content_dispatches_video_types():
    inline = create_content(
        {
            "content_type": "video_inline",
            "video_content_type": "video/mp4",
            "data": "ZmFrZQ==",
        }
    )
    assert isinstance(inline, ContentVideoInline)

    url = create_content(
        {
            "content_type": "video_url",
            "url": "https://youtu.be/9hE5-98ZeCg",
        }
    )
    assert isinstance(url, ContentVideoUrl)
