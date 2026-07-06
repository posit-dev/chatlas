# ChatBedrockAnthropic

``` python
ChatBedrockAnthropic(
    model=None,
    max_tokens=4096,
    reasoning=None,
    cache='5m',
    structured_output_mode='auto',
    aws_secret_key=None,
    aws_access_key=None,
    aws_region=None,
    aws_profile=None,
    aws_session_token=None,
    base_url=None,
    system_prompt=None,
    kwargs=None,
)
```

Chat with an AWS bedrock model.

[AWS Bedrock](https://aws.amazon.com/bedrock/) provides a number of chat based models, including those Anthropic’s [Claude](https://aws.amazon.com/bedrock/claude/).

## Prerequisites

> **NOTE:**
>
> Consider using the approach outlined in this guide to manage your AWS credentials: <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>
>
> Rather than passing credentials directly (via `aws_access_key`, `aws_secret_key`, etc.), a common and more secure approach is to configure a named profile in `~/.aws/config` and reference it via the `aws_profile` argument (or the `AWS_PROFILE` environment variable). This works with both static credentials and AWS IAM Identity Center (SSO).
>
> For SSO-based profiles, log in from your terminal before starting a chat so that a valid session token is available:
>
> ``` bash
> aws sso login --profile my-profile
> ```
>
> Then reference that profile:
>
> ``` python
> from chatlas import ChatBedrockAnthropic
>
> chat = ChatBedrockAnthropic(aws_profile="my-profile")
> ```
>
> If the SSO session expires, you’ll see an authentication error; just run `aws sso login` again to refresh it.

> **NOTE:**
>
> `ChatBedrockAnthropic`, requires the `anthropic` package with the `bedrock` extras: `pip install "chatlas[bedrock-anthropic]"`

## Examples

``` python
from chatlas import ChatBedrockAnthropic

chat = ChatBedrockAnthropic(
    aws_profile="...",
    aws_region="us-east",
    aws_secret_key="...",
    aws_access_key="...",
    aws_session_token="...",
)
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. | `None` |
| max_tokens | [int](https://docs.python.org/3/library/functions.html#int) | Maximum number of tokens to generate before stopping. | `4096` |
| reasoning | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['int \| ThinkingConfigEnabledParam'\] | Determines how many tokens Claude can be allocated to reasoning. Must be ≥1024 and less than `max_tokens`. Larger budgets can enable more thorough analysis for complex problems, improving response quality. See [extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking) for details. | `None` |
| cache | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['5m', 'none'\] | How long to cache inputs? Defaults to “5m” (5-minute TTL), matching `ChatAnthropic`. Set to “none” to disable caching. Note that Bedrock only supports a 5-minute TTL (unlike the direct Anthropic API which also offers “1h”). | `'5m'` |
| structured_output_mode | `StructuredOutputMode` | How to handle structured data extraction (i.e., `data_model`). See `ChatAnthropic` for details. | `'auto'` |
| aws_secret_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The AWS secret key to use for authentication. | `None` |
| aws_access_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The AWS access key to use for authentication. | `None` |
| aws_region | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The AWS region to use. Defaults to the AWS_REGION environment variable. If that is not set, defaults to `'us-east-1'`. | `None` |
| aws_profile | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The name of an AWS profile (as configured in `~/.aws/config` or `~/.aws/credentials`) to use for authentication. Defaults to the `AWS_PROFILE` environment variable. This is often the most convenient way to authenticate, especially for AWS IAM Identity Center (SSO) profiles: run `aws sso login --profile <name>` in your terminal first, then pass the profile name here. | `None` |
| aws_session_token | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The AWS session token to use. | `None` |
| base_url | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The base URL to use. Defaults to the ANTHROPIC_BEDROCK_BASE_URL environment variable. If that is not set, defaults to `f"https://bedrock-runtime.{aws_region}.amazonaws.com"`. | `None` |
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatBedrockClientArgs'\] | Additional arguments to pass to the `anthropic.AnthropicBedrock()` client constructor. | `None` |

## Troubleshooting

If you encounter 400 or 403 errors when trying to use the model, keep the following in mind:

> **NOTE:**
>
> If the model name is completely incorrect, you’ll see an error like `Error code: 400 - {'message': 'The provided model identifier is invalid.'}`
>
> Make sure the model name is correct and active in the specified region.

> **NOTE:**
>
> If you encounter errors similar to `Error code: 403 - {'message': "You don't have access to the model with the specified model ID."}`, make sure your model is active in the relevant `aws_region`.
>
> Keep in mind, if `aws_region` is not specified, and AWS_REGION is not set, the region defaults to us-east-1, which may not match to your AWS config’s default region.

> **NOTE:**
>
> In some cases, even if you have the right model and the right region, you may still encounter an error like `Error code: 400 - {'message': 'Invocation of model ID anthropic.claude-3-5-sonnet-20240620-v1:0 with on-demand throughput isn't supported. Retry your request with the ID or ARN of an inference profile that contains this model.'}`
>
> In this case, you’ll need to look up the ‘cross region inference ID’ for your model. This might required opening your `aws-console` and navigating to the ‘Anthropic Bedrock’ service page. From there, go to the ‘cross region inference’ tab and copy the relevant ID.
>
> For example, if the desired model ID is `anthropic.claude-3-5-sonnet-20240620-v1:0`, the cross region ID might look something like `us.anthropic.claude-3-5-sonnet-20240620-v1:0`.

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A Chat object. |
