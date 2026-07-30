from __future__ import annotations

import base64
from pathlib import Path

from ._content import ContentPDF
from ._content_file import filename_from_url, parse_data_url

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

    # Prefer the name the URL already carries: it's more meaningful to the model
    # than a counter, and it keeps the request body stable across calls.
    name = filename_from_url(url)
    return ContentPDF(
        filename=name if name.lower().endswith(".pdf") else unique_pdf_name(),
        url=url,
    )


def make_pdf_namer():
    cur_pdf_id = 0

    def unique_pdf_name():
        nonlocal cur_pdf_id
        cur_pdf_id += 1
        return f"file_{cur_pdf_id:03d}.pdf"

    return unique_pdf_name


unique_pdf_name = make_pdf_namer()
