import os
import tempfile
from typing import Union

import requests

from ._content import ContentPDF

__all__ = (
    "content_pdf_url",
    "content_pdf_file",
)


def content_pdf_file(path: Union[str, os.PathLike]) -> ContentPDF:
    """
    Prepare a local PDF for input to a chat.

    Not all providers support PDF input, so check the documentation for the
    provider you are using.

    Parameters
    ----------
    path
        A path to a local PDF file.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    with open(path, "rb") as f:
        data = f.read()

    return ContentPDF(data=data)


def content_pdf_url(url: str) -> ContentPDF:
    """
    Use a remote PDF for input to a chat.

    Not all providers support PDF input, so check the documentation for the
    provider you are using.

    Parameters
    ----------
    url
        A URL to a remote PDF file.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.
    """

    if url.startswith("data:"):
        content_type, base64_data = parse_data_url(url)
        if content_type != "application/pdf":
            raise ValueError(f"Unsupported PDF content type: {content_type}")
        return ContentPDF(data=base64_data.encode("utf-8"))
    # TODO: need separate ContentPDFRemote type so we can use file upload
    # apis where they exist. Might need some kind of mutable state so can
    # record point to uploaded file.
    data = download_pdf_bytes(url)
    return ContentPDF(data=data)


def parse_data_url(url: str) -> tuple[str, str]:
    parts = url[5:].split(";", 1)
    if len(parts) != 2 or not parts[1].startswith("base64,"):
        raise ValueError("url is not a valid data URL.")
    return (parts[0], parts[1][7:])


def download_pdf_bytes(url):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

            temp_file.flush()
            temp_file.seek(0)

            return temp_file.read()

        except Exception as e:
            raise e
