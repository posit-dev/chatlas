# content_image_plot

``` python
content_image_plot(width=768, height=768, dpi=72)
```

Encode the current matplotlib plot as an image for chat input.

This function captures the current matplotlib plot, resizes it to the specified dimensions, and prepares it for chat input.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| width | [int](https://docs.python.org/3/library/functions.html#int) | The desired width of the output image in pixels. | `768` |
| height | [int](https://docs.python.org/3/library/functions.html#int) | The desired height of the output image in pixels. | `768` |
| dpi | [int](https://docs.python.org/3/library/functions.html#int) | The DPI (dots per inch) of the output image. | `72` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | \[\](`~chatlas.types.Content`) | Content suitable for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) object. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If width or height is not a positive integer. |

## Examples

``` python
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
