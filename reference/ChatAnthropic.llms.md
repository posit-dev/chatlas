# ChatAnthropic

``` python
ChatAnthropic(
    system_prompt=None,
    model=None,
    max_tokens=4096,
    reasoning=None,
    cache='5m',
    api_key=None,
    kwargs=None,
)
```

Chat with an Anthropic Claude model.

[Anthropic](https://www.anthropic.com) provides a number of chat based models under the [Claude](https://www.anthropic.com/claude) moniker.

## Prerequisites

> **NOTE:**
>
> Note that a Claude Pro membership does not give you the ability to call models via the API. You will need to go to the [developer console](https://console.anthropic.com/account/keys) to sign up (and pay for) a developer account that will give you an API key that you can use with this package.

> **NOTE:**
>
> `ChatAnthropic` requires the `anthropic` package: `pip install "chatlas[anthropic]"`.

## Examples

``` python
import os
from chatlas import ChatAnthropic

chat = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chat.chat("What is the capital of France?")
```

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | 'Optional\[ModelParam\]' | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| max_tokens | [int](https://docs.python.org/3/library/functions.html#int) | Maximum number of tokens to generate before stopping. | `4096` |
| reasoning | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['int \| ThinkingConfigEnabledParam'\] | Determines how many tokens Claude can be allocated to reasoning. Must be ≥1024 and less than `max_tokens`. Larger budgets can enable more thorough analysis for complex problems, improving response quality. See [extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking) for details. | `None` |
| cache | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['5m', '1h', 'none'\] | How long to cache inputs? Defaults to “5m” (five minutes). Set to “none” to disable caching or “1h” to cache for one hour. See the Caching section for details. | `'5m'` |
| api_key | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The API key to use for authentication. You generally should not supply this directly, but instead set the `ANTHROPIC_API_KEY` environment variable. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['ChatClientArgs'\] | Additional arguments to pass to the `anthropic.Anthropic()` client constructor. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A Chat object. |

## Note

Pasting an API key into a chat constructor (e.g., `ChatAnthropic(api_key="...")`) is the simplest way to get started, and is fine for interactive use, but is problematic for code that may be shared with others.

Instead, consider using environment variables or a configuration file to manage your credentials. One popular way to manage credentials is to use a `.env` file to store your credentials, and then use the `python-dotenv` package to load them into your environment.

``` shell
pip install python-dotenv
```

``` shell
# .env
ANTHROPIC_API_KEY=...
```

``` python
from chatlas import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
chat = ChatAnthropic()
chat.console()
```

Another, more general, solution is to load your environment variables into the shell before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

``` shell
export ANTHROPIC_API_KEY=...
```

## Caching

Caching with Claude is a bit more complicated than other providers but we believe that on average it will save you both money and time, so we have enabled it by default. With other providers, like OpenAI and Google, you only pay for cache reads, which cost 10% of the normal price. With Claude, you also pay for cache writes, which cost 125% of the normal price for 5 minute caching and 200% of the normal price for 1 hour caching.

How does this affect the total cost of a conversation? Imagine the first turn sends 1000 input tokens and receives 200 output tokens. The second turn must first send both the input and output from the previous turn (1200 tokens). It then sends a further 1000 tokens and receives 200 tokens back.

To compare the prices of these two approaches we can ignore the cost of output tokens, because they are the same for both. How much will the input tokens cost? If we don’t use caching, we send 1000 tokens in the first turn and 2200 (1000 + 200 + 1000) tokens in the second turn for a total of 3200 tokens. If we use caching, we’ll send (the equivalent of) 1000 \* 1.25 = 1250 tokens in the first turn. In the second turn, 1000 of the input tokens will be cached so the total cost is 1000 \* 0.1 + (200 + 1000) \* 1.25 = 1600 tokens. That makes a total of 2850 tokens, i.e. 11% fewer tokens, decreasing the overall cost.

Obviously, the details will vary from conversation to conversation, but if you have a large system prompt that you re-use many times you should expect to see larger savings. You can see exactly how many input and cache input tokens each turn uses, along with the total cost, with `chat.get_tokens()`. If you don’t see savings for your use case, you can suppress caching with `cache="none"`.

Note: Claude will only cache longer prompts, with caching requiring at least 1024-4096 tokens, depending on the model. So don’t be surprised if you don’t see any differences with caching if you have a short prompt.

See all the details at <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>.
