from __future__ import annotations

import base64
import io
import os
import re
import warnings
from typing import Literal, Union, cast

from ._content import ContentImageInline, ContentImageRemote, ImageContentTypes
from ._content_pdf import parse_data_url
from ._utils import MISSING, MISSING_TYPE

__all__ = (
    "content_image_url",
    "content_image_file",
    "content_image_plot",
)


def content_image_url(
    url: str, detail: Literal["auto", "low", "high"] = "auto"
) -> Union[ContentImageInline, ContentImageRemote]:
    """
    Encode image content from a URL for chat input.

    This function is used to prepare image URLs for input to the chatbot. It can
    handle both regular URLs and data URLs.

    Parameters
    ----------
    url
        The URL of the image to include in the chat input. Can be a data: URL or a
        regular URL.
    detail
        The detail setting for this image. Can be `"auto"`, `"low"`, or `"high"`.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Examples
    --------
    ```python
    from chatlas import ChatOpenAI, content_image_url

    chat = ChatOpenAI()
    chat.chat(
        "What do you see in this image?",
        content_image_url("https://www.python.org/static/img/python-logo.png"),
    )
    ```

    Raises
    ------
    ValueError
        If the URL is not valid or the detail setting is invalid.
    """
    if detail not in ["auto", "low", "high"]:
        raise ValueError("detail must be 'auto', 'low', or 'high'")

    if url.startswith("data:"):
        content_type, base64_data = parse_data_url(url)
        if content_type not in ["image/png", "image/jpeg", "image/webp", "image/gif"]:
            raise ValueError(f"Unsupported image content type: {content_type}")
        content_type = cast(ImageContentTypes, content_type)
        return ContentImageInline(image_content_type=content_type, data=base64_data)
    else:
        return ContentImageRemote(url=url, detail=detail)


def content_image_file(
    path: str,
    content_type: Literal["auto", ImageContentTypes] = "auto",
    resize: Union[Literal["low", "high", "none"], str, MISSING_TYPE] = MISSING,
) -> ContentImageInline:
    """
    Encode image content from a file for chat input.

    This function is used to prepare image files for input to the chatbot. It
    can handle various image formats and provides options for resizing.

    Parameters
    ----------
    path
        The path to the image file to include in the chat input.
    content_type
        The content type of the image (e.g., `"image/png"`). If `"auto"`, the content
        type is inferred from the file extension.
    resize
        Resizing option for the image. Can be:
            - `"low"`: Resize to fit within 512x512
            - `"high"`: Resize to fit within 2000x768 or 768x2000
            - `"none"`: No resizing
            - Custom string (e.g., `"200x200"`, `"300x200>!"`, etc.)

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Examples
    --------
    ```python
    from chatlas import ChatOpenAI, content_image_file

    chat = ChatOpenAI()
    chat.chat(
        "What do you see in this image?",
        content_image_file("path/to/image.png"),
    )
    ```

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is unsupported or the resize option is invalid.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} must be an existing file.")

    if content_type == "auto":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".png":
            content_type = "image/png"
        elif ext in [".jpeg", ".jpg"]:
            content_type = "image/jpeg"
        elif ext == ".webp":
            content_type = "image/webp"
        elif ext == ".gif":
            content_type = "image/gif"
        else:
            raise ValueError(f"Unsupported image file extension: {ext}")

    if resize == "none":
        with open(path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Image resizing requires the `Pillow` package. "
                "Install it with `pip install Pillow`."
            )

        img = Image.open(path)

        if isinstance(resize, MISSING_TYPE):
            warnings.warn(
                "The `resize` parameter is missing. Defaulting to `resize='low'`. "
                "As a result, the image has likely lost quality before the model received it. "
                "Set `resize='low'` to suppress this warning, `resize='high'` for higher quality, or "
                "`resize='none'` to disable resizing.",
                category=MissingResizeWarning,
                stacklevel=2,
            )
            resize = "low"

        if resize == "low":
            img.thumbnail((512, 512))
        elif resize == "high":
            if img.width > img.height:
                img.thumbnail((2000, 768))
            else:
                img.thumbnail((768, 2000))
        else:
            match = re.match(r"(\d+)x(\d+)(>?)(!?)", resize)
            if match:
                width, height = map(int, match.group(1, 2))
                only_shrink = ">" in match.group(3)
                ignore_aspect = "!" in match.group(4)

                if only_shrink and (img.width <= width and img.height <= height):
                    pass  # No resize needed
                elif ignore_aspect:
                    img = img.resize((width, height))
                else:
                    img.thumbnail((width, height))
            else:
                raise ValueError(f"Invalid resize value: {resize}")

        buffer = io.BytesIO()
        img.save(buffer, format=img.format)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return ContentImageInline(image_content_type=content_type, data=base64_data)


def content_image_plot(
    width: int = 768, height: int = 768, dpi: int = 72
) -> ContentImageInline:
    """
    Encode the current matplotlib plot as an image for chat input.

    This function captures the current matplotlib plot, resizes it to the specified
    dimensions, and prepares it for chat input.

    Parameters
    ----------
    width
        The desired width of the output image in pixels.
    height
        The desired height of the output image in pixels.
    dpi
        The DPI (dots per inch) of the output image.

    Returns
    -------
    [](`~chatlas.types.Content`)
        Content suitable for a [](`~chatlas.Turn`) object.

    Raises
    ------
    ValueError
        If width or height is not a positive integer.

    Examples
    --------

    ```python
    from chatlas import ChatOpenAI, content_image_plot
    import matplotlib.pyplot as plt

    plt.scatter(faithful["eruptions"], faithful["waiting"])
    chat = ChatOpenAI()
    chat.chat(
        "Describe this plot in one paragraph, as suitable for inclusion in "
        "alt-text. You should briefly describe the plot type, the axes, and "
        "2-5 major visual patterns.",
        content_image_plot(),
    )
    ```
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "`content_image_plot()` requires the `matplotlib` package. "
            "Install it with `pip install matplotlib`."
        )

    if not plt.get_fignums():
        raise RuntimeError(
            "No matplotlib figure to save. Please create one before calling `content_image_plot()`."
        )

    fig = plt.gcf()
    size = fig.get_size_inches()

    try:
        fig.set_size_inches(width / dpi, height / dpi)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        base64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return ContentImageInline(image_content_type="image/png", data=base64_data)
    finally:
        fig.set_size_inches(*size)


class MissingResizeWarning(RuntimeWarning):
    pass
