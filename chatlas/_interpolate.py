import inspect
from pathlib import Path
from typing import Any, Optional, Union

from jinja2 import Environment

__all__ = (
    "interpolate",
    "interpolate_file",
)


def interpolate(
    prompt: str,
    *,
    variables: Optional[dict[str, Any]] = None,
    variable_start: str = "{{",
    variable_end: str = "}}",
) -> str:
    """
    Interpolate variables into a prompt

    This is a light-weight wrapper around the Jinja2 templating engine, making
    it easier to interpolate dynamic data into a prompt template. Compared to
    f-strings, which expects you to wrap dynamic values in `{ }`, this function
    expects `{{{ }}}` instead, making it easier to include Python code and JSON in
    your prompt.

    Parameters
    ----------
    prompt
        The prompt to interpolate (as a string).
    variables
        A dictionary of variables to interpolate into the prompt. If not
        provided, the caller's global and local variables are used.
    variable_start
        The string that marks the beginning of a variable.
    variable_end
        The string that marks the end of a variable.

    Returns
    -------
    str
        The prompt with variables interpolated.

    Examples
    --------

    ```python
    from chatlas import interpolate

    x = 1
    interpolate("The value of `x` is: {{ x }}")
    ```
    """
    if variables is None:
        frame = inspect.currentframe()
        variables = _infer_variables(frame)
        del frame

    env = Environment(
        variable_start_string=variable_start,
        variable_end_string=variable_end,
    )

    template = env.from_string(prompt)
    return template.render(variables)


def interpolate_file(
    path: Union[str, Path],
    *,
    variables: Optional[dict[str, Any]] = None,
    variable_start: str = "{{",
    variable_end: str = "}}",
) -> str:
    """
    Interpolate variables into a prompt from a file

    This is a light-weight wrapper around the Jinja2 templating engine, making
    it easier to interpolate dynamic data into a static prompt. Compared to
    f-strings, which expects you to wrap dynamic values in `{ }`, this function
    expects `{{{ }}}` instead, making it easier to include Python code and JSON in
    your prompt.

    Parameters
    ----------
    path
        The path to the file containing the prompt to interpolate.
    variables
        A dictionary of variables to interpolate into the prompt. If not
        provided, the caller's global and local variables are used.
    variable_start
        The string that marks the beginning of a variable.
    variable_end
        The string that marks the end of a variable.

    Returns
    -------
    str
        The prompt with variables interpolated.

    See Also
    --------
    * :func:`~chatlas.interpolate` : Interpolating data into a prompt
    """
    if variables is None:
        frame = inspect.currentframe()
        variables = _infer_variables(frame)
        del frame

    with open(path, "r") as file:
        return interpolate(
            file.read(),
            variables=variables,
            variable_start=variable_start,
            variable_end=variable_end,
        )


def _infer_variables(frame) -> dict[str, Any]:
    if not inspect.isframe(frame) or frame.f_back is None:
        raise RuntimeError(
            "`interpolate()` was unable to infer the caller's global and local "
            "variables (because the caller's frame is not available). Consider "
            "passing `variables` explicitly to `interpolate()`."
        )

    return {
        **frame.f_back.f_globals,
        **frame.f_back.f_locals,
    }
