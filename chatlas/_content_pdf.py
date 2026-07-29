from __future__ import annotations

import base64
from pathlib import Path

from ._content import ContentPDF

__all__ = (
    "content_pdf_url",
    "content_pdf_file",
)


def content_pdf_file(path: str | Path) -> ContentPDF:
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

    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    with open(path, "rb") as f:
        data = f.read()

    return ContentPDF(data=data, filename=path.name)


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

    Raises
    ------
    ValueError
        If the URL is not valid, or if it's a `data:` URL with an unsupported
        content type.
    """

    if url.startswith("data:"):
        content_type, base64_data = parse_data_url(url)
        if content_type != "application/pdf":
            raise ValueError(f"Unsupported PDF content type: {content_type}")
        return ContentPDF(
            data=base64.b64decode(base64_data),
            filename=unique_pdf_name(),
        )
    return ContentPDF(filename=unique_pdf_name(), url=url)


def parse_data_url(url: str) -> tuple[str, str]:
    parts = url[5:].split(";", 1)
    if len(parts) != 2 or not parts[1].startswith("base64,"):
        raise ValueError("url is not a valid data URL.")
    return (parts[0], parts[1][7:])


def make_pdf_namer():
    cur_pdf_id = 0

    def unique_pdf_name():
        nonlocal cur_pdf_id
        cur_pdf_id += 1
        return f"file_{cur_pdf_id:03d}.pdf"

    return unique_pdf_name


unique_pdf_name = make_pdf_namer()
