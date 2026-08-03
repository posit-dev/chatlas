# ChatGithub

``` python
ChatGithub(
    system_prompt=None,
    model=None,
    api_key=None,
    base_url='https://models.github.ai/inference/',
    seed=MISSING,
    kwargs=None,
)
```

Deprecated: chat with a model hosted on the GitHub model marketplace.

`ChatGithub()` is defunct because GitHub Models was retired on 2026-07-30. Use [`ChatGoogle`](https://posit-dev.github.io/chatlas/reference/ChatGoogle.html#chatlas.ChatGoogle) (offers a free tier) or [`ChatPosit`](https://posit-dev.github.io/chatlas/reference/ChatPosit.html#chatlas.ChatPosit) (offers a free trial) instead.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | Unused. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | Unused. | `None` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | Unused. | `None` |
| base_url | [str](https://docs.python.org/3/library/stdtypes.html#str) | Unused. | `'https://models.github.ai/inference/'` |
| seed | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[int](https://docs.python.org/3/library/functions.html#int)\] \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Unused. | `MISSING` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Unused. | `None` |

## Raises

| Name | Type | Description |
|----|----|----|
|  | [RuntimeError](https://docs.python.org/3/library/exceptions.html#RuntimeError) | Always, since GitHub Models is no longer available. |
