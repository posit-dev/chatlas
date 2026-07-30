from pathlib import Path

import pytest
from chatlas import content_audio_file
from chatlas._content import ContentAudio, create_content
from chatlas._turn import Turn, UserTurn


def test_can_create_audio_from_local_file():
    tone = Path(__file__).parent / "audio" / "tone.wav"
    obj = content_audio_file(tone)
    assert isinstance(obj, ContentAudio)
    assert obj.mime_type == "audio/wav"
    assert isinstance(obj.data, bytes)
    assert len(obj.data) > 0


def test_content_audio_file_infers_mime_type_from_extension(tmp_path):
    for ext, mime_type in [
        (".wav", "audio/wav"),
        (".mp3", "audio/mp3"),
        (".aiff", "audio/aiff"),
        (".aif", "audio/aiff"),
        (".aac", "audio/aac"),
        (".ogg", "audio/ogg"),
        (".flac", "audio/flac"),
    ]:
        path = tmp_path / f"clip{ext}"
        path.write_bytes(b"\x00\x01\x02\x03")
        obj = content_audio_file(path)
        assert obj.mime_type == mime_type


def test_content_audio_file_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "clip.txt"
    path.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Unsupported audio file extension"):
        content_audio_file(path)


def test_content_audio_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        content_audio_file("does-not-exist.wav")


def test_content_audio_file_explicit_content_type(tmp_path):
    path = tmp_path / "clip.bin"
    path.write_bytes(b"\x00\x01")
    obj = content_audio_file(path, content_type="audio/mp3")
    assert obj.mime_type == "audio/mp3"


def test_content_audio_bytes_round_trip():
    raw = b"\x00\x01\x02\xd6\x05\x06"
    audio = ContentAudio(data=raw, mime_type="audio/wav")
    assert audio.content_type == "audio"

    turn = UserTurn([audio])
    dumped = turn.model_dump(mode="json")
    restored = Turn.model_validate(dumped)
    assert restored.contents[0].data == raw
    assert restored.contents[0].mime_type == "audio/wav"


def test_content_audio_permits_provider_generated_mime_type():
    """
    Provider-generated audio (e.g. Gemini TTS output) may use MIME types
    outside the six chatlas validates for user-supplied files (e.g. raw PCM
    like "audio/pcm;rate=24000"). ContentAudio.mime_type must accept these
    since chatlas can't restrict what a provider echoes back.
    """
    audio = ContentAudio(data=b"\x00\x00", mime_type="audio/pcm;rate=24000")
    assert audio.mime_type == "audio/pcm;rate=24000"


def test_content_audio_create_content_roundtrip():
    c = ContentAudio(data=b"\x01\x02\x03", mime_type="audio/mp3")
    dumped = c.model_dump()
    restored = create_content(dumped)
    assert isinstance(restored, ContentAudio)
    assert restored.data == b"\x01\x02\x03"
    assert restored.mime_type == "audio/mp3"


def test_content_audio_str():
    c = ContentAudio(data=b"\x01\x02\x03", mime_type="audio/wav")
    assert str(c) == "<audio mime_type=audio/wav size=3 bytes>"
