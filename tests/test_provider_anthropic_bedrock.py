import base64

import pytest
from chatlas._content import ContentImageRemote, ContentPDF
from chatlas._provider_anthropic import AnthropicBedrockProvider


class TestUrlContentFallsBackToBytes:
    """Bedrock's InvokeModel API can't fetch `url` sources, so URL content
    must be sent as bytes instead (https://github.com/posit-dev/chatlas/issues/410).
    """

    def test_pdf_url_downloads_bytes_instead(self, monkeypatch):
        monkeypatch.setattr(
            "chatlas._content_file.download_bytes", lambda url: b"%PDF-1.4 fake"
        )
        c = ContentPDF(filename="a.pdf", url="https://example.com/a.pdf")

        block = AnthropicBedrockProvider._as_content_block(c)

        assert block["source"] == {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(b"%PDF-1.4 fake").decode("utf-8"),
        }
        # The downloaded bytes are cached back onto the original content.
        assert c.data == b"%PDF-1.4 fake"

    def test_pdf_with_data_ignores_url_without_downloading(self, monkeypatch):
        def fail_download(url):
            raise AssertionError("shouldn't download when data is already present")

        monkeypatch.setattr("chatlas._content_file.download_bytes", fail_download)
        c = ContentPDF(
            data=b"%PDF-1.4", filename="a.pdf", url="https://example.com/a.pdf"
        )

        block = AnthropicBedrockProvider._as_content_block(c)

        assert block["source"]["type"] == "base64"
        assert block["source"]["data"] == base64.b64encode(b"%PDF-1.4").decode("utf-8")

    def test_remote_image_raises(self):
        with pytest.raises(ValueError, match="Remote images aren't supported"):
            AnthropicBedrockProvider._as_content_block(
                ContentImageRemote(url="https://example.com/i.png")
            )
