# interpolate_file

``` python
interpolate_file(
    path,
    *,
    variables=None,
    variable_start='{{',
    variable_end='}}',
)
```

Interpolate variables into a prompt from a file

This is a light-weight wrapper around the Jinja2 templating engine, making it easier to interpolate dynamic data into a static prompt. Compared to f-strings, which expects you to wrap dynamic values in `{ }`, this function expects `{{ }}` instead, making it easier to include Python code and JSON in your prompt.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| path | [Union](https://docs.python.org/3/library/typing.html#typing.Union)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)\] | The path to the file containing the prompt to interpolate. | *required* |
| variables | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | A dictionary of variables to interpolate into the prompt. If not provided, the caller’s global and local variables are used. | `None` |
| variable_start | [str](https://docs.python.org/3/library/stdtypes.html#str) | The string that marks the beginning of a variable. | `'{{'` |
| variable_end | [str](https://docs.python.org/3/library/stdtypes.html#str) | The string that marks the end of a variable. | `'}}'` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [str](https://docs.python.org/3/library/stdtypes.html#str) | The prompt with variables interpolated. |

## See Also

- [`interpolate`](https://posit-dev.github.io/chatlas/reference/interpolate.html#chatlas.interpolate) : Interpolating data into a prompt
