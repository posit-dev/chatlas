# ChatBedrock

``` python
ChatBedrock(
    system_prompt=None,
    model=None,
    api=None,
    aws_profile=None,
    aws_region=None,
    base_url=None,
    max_tokens=MISSING,
    cache='auto',
    kwargs=None,
)
```

Chat with a model hosted on AWS Bedrock.

Bedrock exposes three APIs across two endpoints, and `api` selects which one to use and which request format to send:

- `"responses"` uses the OpenAI Responses API on the `bedrock-mantle` endpoint. This is the only way to reach the GPT-5 family, Grok, and Gemma on Bedrock.
- `"messages"` uses the Anthropic Messages API on the `bedrock-mantle` endpoint. Only Claude models are available here, but it includes some (like Claude Mythos) that no other Bedrock API serves.
- `"converse"` uses the Converse API on the `bedrock-runtime` endpoint. This is the default path for models available through Converse.

By default the API is picked from `model`, falling back to `"converse"` for models that aren’t recognised as mantle-only. Set `api` explicitly to override this – which is also how you reach a mantle model that Converse can also serve, since those aren’t auto-routed.

Note that the two endpoints have separate token quotas, so moving a model from one to the other changes which quota it consumes.

## Prerequisites

> **NOTE:**
>
> Authentication uses botocore’s standard credential chain, so environment variables, `~/.aws/config`, SSO, and instance roles all work. Pass `aws_profile` to select a named profile.
>
> For direct Converse requests, pass `api_key` via `kwargs` or set `AWS_BEARER_TOKEN_BEDROCK` to authenticate with a bearer token instead of SigV4. An explicit `api_key` takes priority; otherwise an explicit `aws_profile` uses SigV4 before the environment bearer token is considered.

The direct Converse client uses raw HTTPX plus AWS libraries. The `bedrock` extra provides the AWS libraries and vendor SDKs for the mantle APIs (`openai` for `api="responses"` and `anthropic` for `api="messages"`). A chat uses one of these three API clients.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. Defaults to `"us.anthropic.claude-sonnet-4-6"`. | `None` |
| api | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[`BedrockAPI`\] | Which Bedrock API to use. The default, `None`, picks the API from `model`. | `None` |
| aws_profile | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The AWS profile to use. Defaults to botocore’s default profile. | `None` |
| aws_region | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The AWS region to use. Defaults to the region from your AWS config. | `None` |
| base_url | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | Override the endpoint URL. Needed to reach mantle’s other OpenAI-compatible path, `/v1`, which serves older open-weight models like `gpt-oss` and rejects the models `/openai/v1` serves. | `None` |
| max_tokens | [int](https://docs.python.org/3/library/functions.html#int) \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Maximum number of tokens to generate, defaulting to 4096 when `api="converse"` or `api="messages"`. Passing this when `api="responses"` raises, since the Responses API has no constructor-level equivalent – set a cap per-request instead via `chat.set_model_params(max_tokens=...)`. | `MISSING` |
| cache | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['auto', '5m', '1h', 'none'\] | Prompt caching for `api="converse"` and `api="messages"`. The Responses API caches automatically, so this must be left at `"auto"` when `api="responses"`. | `'auto'` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['OpenAIClientArgs \| AnthropicClientArgs \| ConverseClientArgs'\] | Additional client arguments. For `api="converse"`, these are raw `httpx.Client` arguments (except `auth` and `base_url`, which `ChatBedrock` manages) and `api_key` is a Bedrock bearer token. For `api="responses"` and `api="messages"`, these are arguments for the respective OpenAI or Anthropic SDK client. On the Converse path, `api_key` is consumed to create an `Authorization: Bearer ...` header, not passed to `httpx.Client`. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A Chat object. |

## Examples

``` python
from chatlas import ChatBedrock

# Frontier OpenAI models, which only exist on bedrock-mantle
chat = ChatBedrock(model="openai.gpt-5.6-sol")
chat.chat("What is 1 + 1? Just the number.")

# Claude through the Anthropic Messages API on mantle
chat = ChatBedrock(model="anthropic.claude-haiku-4-5", api="messages")
```
