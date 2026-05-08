# Chat

``` python
Chat(provider, system_prompt=None, kwargs_chat=None)
```

A chat object that can be used to interact with a language model.

A `Chat` is an sequence of sequence of user and assistant [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn)s sent to a specific [`Provider`](https://posit-dev.github.io/chatlas/reference/Provider.html#chatlas.Provider). A `Chat` takes care of managing the state associated with the chat; i.e. it records the messages that you send to the server, and the messages that you receive back. If you register a tool (i.e. an function that the assistant can call on your behalf), it also takes care of the tool loop.

You should generally not create this object yourself, but instead call [`ChatOpenAI`](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html#chatlas.ChatOpenAI) or friends instead.

## Attributes

| Name | Description |
|----|----|
| [current_display](#chatlas.Chat.current_display) | Get the currently active markdown display, if any. |
| [system_prompt](#chatlas.Chat.system_prompt) | A property to get (or set) the system prompt for the chat. |

## Methods

| Name | Description |
|----|----|
| [add_turn](#chatlas.Chat.add_turn) | Add a turn to the chat. |
| [app](#chatlas.Chat.app) | Enter a web-based chat app to interact with the LLM. |
| [chat](#chatlas.Chat.chat) | Generate a response from the chat. |
| [chat_async](#chatlas.Chat.chat_async) | Generate a response from the chat asynchronously. |
| [chat_structured](#chatlas.Chat.chat_structured) | Extract structured data. |
| [chat_structured_async](#chatlas.Chat.chat_structured_async) | Extract structured data from the given input asynchronously. |
| [cleanup_mcp_tools](#chatlas.Chat.cleanup_mcp_tools) | Close MCP server connections (and their corresponding tools). |
| [console](#chatlas.Chat.console) | Enter a chat console to interact with the LLM. |
| [export](#chatlas.Chat.export) | Export the chat history to a file. |
| [export_eval](#chatlas.Chat.export_eval) | Creates an Inspect AI eval dataset sample from the current chat. |
| [extract_data](#chatlas.Chat.extract_data) | Deprecated: use `.chat_structured()` instead. |
| [extract_data_async](#chatlas.Chat.extract_data_async) | Deprecated: use `.chat_structured_async()` instead. |
| [get_cost](#chatlas.Chat.get_cost) | Estimate the cost of the chat. |
| [get_last_turn](#chatlas.Chat.get_last_turn) | Get the last turn in the chat with a specific role. |
| [get_tokens](#chatlas.Chat.get_tokens) | Get the tokens for each turn in the chat. |
| [get_tools](#chatlas.Chat.get_tools) | Get the list of registered tools. |
| [get_turns](#chatlas.Chat.get_turns) | Get all the turns (i.e., message contents) in the chat. |
| [list_models](#chatlas.Chat.list_models) | List all models available for the provider. |
| [on_tool_request](#chatlas.Chat.on_tool_request) | Register a callback for a tool request event. |
| [on_tool_result](#chatlas.Chat.on_tool_result) | Register a callback for a tool result event. |
| [register_mcp_tools_http_stream_async](#chatlas.Chat.register_mcp_tools_http_stream_async) | Register tools from an MCP server using streamable HTTP transport. |
| [register_mcp_tools_stdio_async](#chatlas.Chat.register_mcp_tools_stdio_async) | Register tools from a MCP server using stdio (standard input/output) transport. |
| [register_tool](#chatlas.Chat.register_tool) | Register a tool (function) with the chat. |
| [set_echo_options](#chatlas.Chat.set_echo_options) | Set echo styling options for the chat. |
| [set_model_params](#chatlas.Chat.set_model_params) | Set common model parameters for the chat. |
| [set_tools](#chatlas.Chat.set_tools) | Set the tools for the chat. |
| [set_turns](#chatlas.Chat.set_turns) | Set the turns of the chat. |
| [stream](#chatlas.Chat.stream) | Generate a response from the chat in a streaming fashion. |
| [stream_async](#chatlas.Chat.stream_async) | Generate a response from the chat in a streaming fashion asynchronously. |
| [to_solver](#chatlas.Chat.to_solver) | Create an InspectAI solver from this chat. |
| [token_count](#chatlas.Chat.token_count) | Get an estimated token count for the given input. |
| [token_count_async](#chatlas.Chat.token_count_async) | Get an estimated token count for the given input asynchronously. |

### add_turn

``` python
Chat.add_turn(turn)
```

Add a turn to the chat.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| turn | [Turn](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) | The turn to add. Turns with the role “system” are not allowed. | *required* |

### app

``` python
Chat.app(
    stream=True,
    port=0,
    host='127.0.0.1',
    launch_browser=True,
    bookmark_store='url',
    bg_thread=None,
    echo=None,
    content='all',
    kwargs=None,
)
```

Enter a web-based chat app to interact with the LLM.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| stream | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to stream the response (i.e., have the response appear in chunks). | `True` |
| port | [int](https://docs.python.org/3/library/functions.html#int) | The port to run the app on (the default is 0, which will choose a random port). | `0` |
| host | [str](https://docs.python.org/3/library/stdtypes.html#str) | The host to run the app on (the default is “127.0.0.1”). | `'127.0.0.1'` |
| launch_browser | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to launch a browser window. | `True` |
| bookmark_store | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['url', 'server', 'disable'\] | One of the following (default is “url”): - `"url"`: Store bookmarks in the URL (default). - `"server"`: Store bookmarks on the server (requires a server-side storage backend). - `"disable"`: Disable bookmarking. | `'url'` |
| bg_thread | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[bool](https://docs.python.org/3/library/functions.html#bool)\] | Whether to run the app in a background thread. If `None`, the app will run in a background thread if the current environment is a notebook. | `None` |
| echo | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[`EchoOptions`\] | One of the following (defaults to `"none"` when `stream=True` and `"text"` when `stream=False`): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `None` |
| content | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['text', 'all'\] | Whether to display text content or all content (i.e., tool calls). | `'all'` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

### chat

``` python
Chat.chat(*args, echo='output', stream=True, kwargs=None)
```

Generate a response from the chat.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The user input(s) to generate a response from. | `()` |
| echo | `EchoOptions` | One of the following (default is “output”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'output'` |
| stream | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to stream the response (i.e., have the response appear in chunks). | `True` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [ChatResponse](https://posit-dev.github.io/chatlas/reference/types.ChatResponse.html#chatlas.types.ChatResponse) | A (consumed) response from the chat. Apply `str()` to this object to get the text content of the response. |

### chat_async

``` python
Chat.chat_async(*args, echo='output', stream=True, kwargs=None)
```

Generate a response from the chat asynchronously.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The user input(s) to generate a response from. | `()` |
| echo | `EchoOptions` | One of the following (default is “output”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'output'` |
| stream | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to stream the response (i.e., have the response appear in chunks). | `True` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [ChatResponseAsync](https://posit-dev.github.io/chatlas/reference/types.ChatResponseAsync.html#chatlas.types.ChatResponseAsync) | A (consumed) response from the chat. Apply `str()` to this object to get the text content of the response. |

### chat_structured

``` python
Chat.chat_structured(*args, data_model, echo='none', stream=False, kwargs=None)
```

Extract structured data.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The input to send to the chatbot. This is typically the text you want to extract data from, but it can be omitted if the data is obvious from the existing conversation. | `()` |
| data_model | [type](https://docs.python.org/3/library/functions.html#type)\[`BaseModelT`\] | A Pydantic model describing the structure of the data to extract. | *required* |
| echo | `EchoOptions` | One of the following (default is “none”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'none'` |
| stream | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to stream the response (i.e., have the response appear in chunks). | `False` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | `BaseModelT` | An instance of the provided `data_model` containing the extracted data. |

### chat_structured_async

``` python
Chat.chat_structured_async(
    *args,
    data_model,
    echo='none',
    stream=False,
    kwargs=None,
)
```

Extract structured data from the given input asynchronously.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The input to send to the chatbot. This is typically the text you want to extract data from, but it can be omitted if the data is obvious from the existing conversation. | `()` |
| data_model | [type](https://docs.python.org/3/library/functions.html#type)\[`BaseModelT`\] | A Pydantic model describing the structure of the data to extract. | *required* |
| echo | `EchoOptions` | One of the following (default is “none”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'none'` |
| stream | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to stream the response (i.e., have the response appear in chunks). Defaults to `True` if `echo` is not “none”. | `False` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | `BaseModelT` | An instance of the provided `data_model` containing the extracted data. |

### cleanup_mcp_tools

``` python
Chat.cleanup_mcp_tools(names=None)
```

Close MCP server connections (and their corresponding tools).

This method closes the MCP client sessions and removes the tools registered from the MCP servers. If a specific `name` is provided, it will only clean up the tools and session associated with that name. If no name is provided, it will clean up all registered MCP tools and sessions.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| names | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | If provided, only clean up the tools and session associated with these names. If not provided, clean up all registered MCP tools and sessions. | `None` |

#### Returns

| Name | Type | Description |
|------|------|-------------|
|      | None |             |

### console

``` python
Chat.console(echo='output', stream=True, kwargs=None)
```

Enter a chat console to interact with the LLM.

To quit, input ‘exit’ or press Ctrl+C.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| echo | `EchoOptions` | One of the following (default is “output”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'output'` |
| stream | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to stream the response (i.e., have the response appear in chunks). | `True` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response | `None` |

#### Returns

| Name | Type | Description |
|------|------|-------------|
|      | None |             |

### export

``` python
Chat.export(
    filename,
    *,
    turns=None,
    title=None,
    content='text',
    include_system_prompt=True,
    overwrite=False,
)
```

Export the chat history to a file.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| filename | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | The filename to export the chat to. Currently this must be a `.md` or `.html` file. | *required* |
| turns | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[Turn](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn)\]\] | The `.get_turns()` to export. If not provided, the chat’s current turns will be used. | `None` |
| title | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A title to place at the top of the exported file. | `None` |
| overwrite | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to overwrite the file if it already exists. | `False` |
| content | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['text', 'all'\] | Whether to include text content, all content (i.e., tool calls), or no content. | `'text'` |
| include_system_prompt | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to include the system prompt in a | `True` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | The path to the exported file. |

### export_eval

``` python
Chat.export_eval(
    filename,
    *,
    target=None,
    include_system_prompt=True,
    turns=None,
    overwrite='append',
    **kwargs,
)
```

Creates an Inspect AI eval dataset sample from the current chat.

Creates an Inspect AI eval [Sample](https://inspect.aisi.org.uk/reference/inspect_ai.dataset.html#sample) from the current chat and appends it to a JSONL file. In Inspect, a eval dataset is a collection of Samples, where each Sample represents a single `input` (i.e., user prompt) and the expected `target` (i.e., the target answer and/or grading guidance for it). Note that each `input` of a particular sample can contain a series of messages (from both the user and assistant).

#### Note

Each call to this method appends a single Sample as a new line in the specified JSONL file. If the file does not exist, it will be created.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| filename | [str](https://docs.python.org/3/library/stdtypes.html#str) \| [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | The filename to export the chat to. Currently this must be a `.jsonl` file. | *required* |
| target | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The target output for the eval sample. By default, this is taken to be the content of the last assistant turn. | `None` |
| include_system_prompt | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to include the system prompt (if any) as the first turn in the eval sample. | `True` |
| turns | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[list](https://docs.python.org/3/library/stdtypes.html#list)\[[Turn](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn)\]\] | The input turns for the eval sample. By default, this is taken to be all turns except the last (assistant) turn. Note that system prompts are not allowed here, but controlled separately via the `include_system_prompt` parameter. | `None` |
| overwrite | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['append', True, False\] | Behavior when the file already exists: - `"append"` (default): Append to the existing file. - `True`: Overwrite the existing file. - `False`: Raise an error if the file already exists. | `'append'` |
| kwargs | [Any](https://docs.python.org/3/library/typing.html#typing.Any) | Additional keyword arguments to pass to the `Sample()` constructor. This is primarily useful for setting an ID or metadata on the sample. | `{}` |

#### Examples

Step 1: export the chat to an eval JSONL file

``` python
from chatlas import ChatOpenAI

chat = ChatOpenAI(system_prompt="You are a helpful assistant.")
chat.chat("Hello, how are you?")

chat.export_eval("my_eval_1.jsonl")
```

Step 2: load the eval JSONL file into an Inspect AI eval task

``` python
from chatlas import ChatOpenAI
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

# No need to load in system prompt -- it's included in the eval JSONL file by default
chat = ChatOpenAI()


@task
def my_eval():
    return Task(
        dataset=json_dataset("my_eval.jsonl"),
        solver=chat.to_solver(),
        scorer=model_graded_qa(model="openai/gpt-4o-mini"),
    )
```

### extract_data

``` python
Chat.extract_data(*args, data_model, echo='none', stream=False)
```

Deprecated: use `.chat_structured()` instead.

### extract_data_async

``` python
Chat.extract_data_async(
    *args,
    data_model,
    echo='none',
    stream=False,
    kwargs=None,
)
```

Deprecated: use `.chat_structured_async()` instead.

### get_cost

``` python
Chat.get_cost(include='all', token_price=None)
```

Estimate the cost of the chat.

#### Note

This is a rough estimate, treat it as such. Providers may change their pricing frequently and without notice.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| include | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['all', 'last'\] | One of the following (default is “all”): - `"all"`: Return the total cost of all turns in the chat. - `"last"`: Return the cost of the last turn in the chat. | `'all'` |
| token_price | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)\[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)\]\] | An optional tuple in the format of (input_token_cost, output_token_cost, cached_token_cost) for bringing your own cost information. - `"input_token_cost"`: The cost per user token in USD per million tokens. - `"output_token_cost"`: The cost per assistant token in USD per million tokens. - `"cached_token_cost"`: The cost per cached token read in USD per million tokens. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [float](https://docs.python.org/3/library/functions.html#float) | The cost of the chat, in USD. |

### get_last_turn

``` python
Chat.get_last_turn(role='assistant')
```

Get the last turn in the chat with a specific role.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| role | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['assistant', 'user', 'system'\] | The role of the turn to return. | `'assistant'` |

### get_tokens

``` python
Chat.get_tokens()
```

Get the tokens for each turn in the chat.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`TokensDict`\] | A list of dictionaries with the token counts for each (non-system) turn |

#### Raises

| Name | Type | Description |
|----|----|----|
|  | [ValueError](https://docs.python.org/3/library/exceptions.html#ValueError) | If the chat’s turns (i.e., `.get_turns()`) are not in an expected format. This may happen if the chat history is manually set (i.e., `.set_turns()`). In this case, you can inspect the “raw” token values via the `.get_turns()` method (each turn has a `.tokens` attribute). |

### get_tools

``` python
Chat.get_tools()
```

Get the list of registered tools.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [list](https://docs.python.org/3/library/stdtypes.html#list)\[[Tool](https://posit-dev.github.io/chatlas/reference/Tool.html#chatlas.Tool) \| `ToolBuiltIn`\] | A list of `Tool` or `ToolBuiltIn` instances that are currently registered with the chat. |

### get_turns

``` python
Chat.get_turns(include_system_prompt=False, tool_result_role='user')
```

Get all the turns (i.e., message contents) in the chat.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| include_system_prompt | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to include the system prompt in the turns. | `False` |
| tool_result_role | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['assistant', 'user'\] | The role to assign to turns containing tool results. By default, tool results are assigned a role of “user” since they represent information provided to the assistant. If set to “assistant” tool result content (plus the surrounding assistant turn contents) is collected into a single assistant turn. This is convenient for display purposes and more generally if you want the tool calling loop to be contained in a single turn. | `'user'` |

### list_models

``` python
Chat.list_models()
```

List all models available for the provider.

This method returns detailed information about all models supported by the provider, including model IDs, pricing information, creation dates, and other metadata. This is useful for discovering available models and their characteristics without needing to consult provider documentation.

#### Examples

Get all available models:

``` python
from chatlas import ChatOpenAI

chat = ChatOpenAI()
models = chat.list_models()
print(f"Found {len(models)} models")
print(f"First model: {models[0]['id']}")
```

View models in a table format:

``` python
import pandas as pd
from chatlas import ChatAnthropic

chat = ChatAnthropic()
df = pd.DataFrame(chat.list_models())
print(df[["id", "input", "output"]].head())  # Show pricing info
```

Find models by criteria:

``` python
from chatlas import ChatGoogle

chat = ChatGoogle()
models = chat.list_models()

# Find cheapest input model
cheapest = min(models, key=lambda m: m.get("input", float("inf")))
print(f"Cheapest model: {cheapest['id']}")
```

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [list](https://docs.python.org/3/library/stdtypes.html#list)\[`ModelInfo`\] | A list of ModelInfo dictionaries containing model information. Each dictionary contains: - `id` (str): The model identifier to use with the Chat constructor - `name` (str, optional): Human-readable model name - `input` (float, optional): Cost per input token in USD per million tokens - `output` (float, optional): Cost per output token in USD per million tokens - `cached_input` (float, optional): Cost per cached input token in USD per million tokens - `created_at` (date, optional): Date the model was created - `owned_by` (str, optional): Organization that owns the model - `provider` (str, optional): Model provider name - `size` (int, optional): Model size in bytes - `url` (str, optional): URL with more information about the model The list is typically sorted by creation date (most recent first). |

#### Note

Not all providers support this method. Some providers may raise NotImplementedError with information about where to find model listings online.

### on_tool_request

``` python
Chat.on_tool_request(callback)
```

Register a callback for a tool request event.

A tool request event occurs when the assistant requests a tool to be called on its behalf. Before invoking the tool, `on_tool_request` handlers are called with the relevant `ContentToolRequest` object. This is useful if you want to handle tool requests in a custom way, such as requiring logging them or requiring user approval before invoking the tool

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| callback | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[\[[ContentToolRequest](https://posit-dev.github.io/chatlas/reference/types.ContentToolRequest.html#chatlas.types.ContentToolRequest)\], None\] | A function to be called when a tool request event occurs. This function must have a single argument, which will be the tool request (i.e., a `ContentToolRequest` object). | *required* |

#### Returns

| Name | Type                                                      | Description |
|------|-----------------------------------------------------------|-------------|
|      | A callable that can be used to remove the callback later. |             |

### on_tool_result

``` python
Chat.on_tool_result(callback)
```

Register a callback for a tool result event.

A tool result event occurs when a tool has been invoked and the result is ready to be provided to the assistant. After the tool has been invoked, `on_tool_result` handlers are called with the relevant `ContentToolResult` object. This is useful if you want to handle tool results in a custom way such as logging them.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| callback | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[\[[ContentToolResult](https://posit-dev.github.io/chatlas/reference/types.ContentToolResult.html#chatlas.types.ContentToolResult)\], None\] | A function to be called when a tool result event occurs. This function must have a single argument, which will be the tool result (i.e., a `ContentToolResult` object). | *required* |

#### Returns

| Name | Type                                                      | Description |
|------|-----------------------------------------------------------|-------------|
|      | A callable that can be used to remove the callback later. |             |

### register_mcp_tools_http_stream_async

``` python
Chat.register_mcp_tools_http_stream_async(
    url,
    include_tools=(),
    exclude_tools=(),
    name=None,
    namespace=None,
    transport_kwargs=None,
)
```

Register tools from an MCP server using streamable HTTP transport.

Connects to an MCP server (that communicates over a streamable HTTP transport) and registers the available tools. This is useful for utilizing tools provided by an MCP server running on a remote server (or locally) over HTTP.

#### Pre-requisites

> **NOTE:**
>
> Requires the `mcp` package to be installed. Install it with:
>
> ``` bash
> pip install mcp
> ```

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| url | [str](https://docs.python.org/3/library/stdtypes.html#str) | URL endpoint where the Streamable HTTP server is mounted (e.g., `http://localhost:8000/mcp`) | *required* |
| name | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A unique name for the MCP server session. If not provided, the name is derived from the MCP server information. This name is primarily useful for cleanup purposes (i.e., to close a particular MCP session). | `None` |
| include_tools | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | List of tool names to include. By default, all available tools are included. | `()` |
| exclude_tools | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | List of tool names to exclude. This parameter and `include_tools` are mutually exclusive. | `()` |
| namespace | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A namespace to prepend to tool names (i.e., `namespace.tool_name`) from this MCP server. This is primarily useful to avoid name collisions with other tools already registered with the chat. This namespace applies when tools are advertised to the LLM, so try to use a meaningful name that describes the server and/or the tools it provides. For example, if you have a server that provides tools for mathematical operations, you might use `math` as the namespace. | `None` |
| transport_kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | Additional keyword arguments for the transport layer (i.e., `mcp.client.streamable_http.streamablehttp_client`). | `None` |

#### Returns

| Name | Type | Description |
|------|------|-------------|
|      | None |             |

#### See Also

- `.cleanup_mcp_tools_async()` : Cleanup registered MCP tools.
- `.register_mcp_tools_stdio_async()` : Register tools from an MCP server using stdio transport.

#### Note

Unlike the `.register_mcp_tools_stdio_async()` method, this method does not launch an MCP server. Instead, it assumes an HTTP server is already running at the specified URL. This is useful for connecting to an existing MCP server that is already running and serving tools.

#### Examples

Assuming you have a Python script `my_mcp_server.py` that implements an MCP server like so:

``` python
from mcp.server.fastmcp import FastMCP

app = FastMCP("my_server")

@app.tool(description="Add two numbers.")
def add(x: int, y: int) -> int:
    return x + y

app.run(transport="streamable-http")
```

You can launch this server like so:

``` bash
python my_mcp_server.py
```

Then, you can register this server with the chat as follows:

``` python
await chat.register_mcp_tools_http_stream_async(
    url="http://localhost:8080/mcp"
)
```

### register_mcp_tools_stdio_async

``` python
Chat.register_mcp_tools_stdio_async(
    command,
    args,
    name=None,
    include_tools=(),
    exclude_tools=(),
    namespace=None,
    transport_kwargs=None,
)
```

Register tools from a MCP server using stdio (standard input/output) transport.

Useful for launching an MCP server and registering its tools with the chat – all from the same Python process.

In more detail, this method:

1.  Executes the given `command` with the provided `args`.
    - This should start an MCP server that communicates via stdio.
2.  Establishes a client connection to the MCP server using the `mcp` package.
3.  Registers the available tools from the MCP server with the chat.
4.  Returns a cleanup callback to close the MCP session and remove the tools.

#### Pre-requisites

> **NOTE:**
>
> Requires the `mcp` package to be installed. Install it with:
>
> ``` bash
> pip install mcp
> ```

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| command | [str](https://docs.python.org/3/library/stdtypes.html#str) | System command to execute to start the MCP server (e.g., `python`). | *required* |
| args | [list](https://docs.python.org/3/library/stdtypes.html#list)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | Arguments to pass to the system command (e.g., `["-m", "my_mcp_server"]`). | *required* |
| name | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A unique name for the MCP server session. If not provided, the name is derived from the MCP server information. This name is primarily useful for cleanup purposes (i.e., to close a particular MCP session). | `None` |
| include_tools | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | List of tool names to include. By default, all available tools are included. | `()` |
| exclude_tools | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | List of tool names to exclude. This parameter and `include_tools` are mutually exclusive. | `()` |
| namespace | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A namespace to prepend to tool names (i.e., `namespace.tool_name`) from this MCP server. This is primarily useful to avoid name collisions with other tools already registered with the chat. This namespace applies when tools are advertised to the LLM, so try to use a meaningful name that describes the server and/or the tools it provides. For example, if you have a server that provides tools for mathematical operations, you might use `math` as the namespace. | `None` |
| transport_kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | Additional keyword arguments for the stdio transport layer (i.e., `mcp.client.stdio.stdio_client`). | `None` |

#### Returns

| Name | Type | Description |
|------|------|-------------|
|      | None |             |

#### See Also

- `.cleanup_mcp_tools_async()` : Cleanup registered MCP tools.
- `.register_mcp_tools_http_stream_async()` : Register tools from an MCP server using streamable HTTP transport.

#### Examples

Assuming you have a Python script `my_mcp_server.py` that implements an MCP server like so

``` python
from mcp.server.fastmcp import FastMCP

app = FastMCP("my_server")

@app.tool(description="Add two numbers.")
def add(y: int, z: int) -> int:
    return y - z

app.run(transport="stdio")
```

You can register this server with the chat as follows:

``` python
from chatlas import ChatOpenAI

chat = ChatOpenAI()

await chat.register_mcp_tools_stdio_async(
    command="python",
    args=["-m", "my_mcp_server"],
)
```

### register_tool

``` python
Chat.register_tool(
    func,
    *,
    force=False,
    name=None,
    model=None,
    annotations=None,
)
```

Register a tool (function) with the chat.

The function will always be invoked in the current Python process.

#### Examples

If your tool has straightforward input parameters, you can just register the function directly (type hints and a docstring explaning both what the function does and what the parameters are for is strongly recommended):

``` python
from chatlas import ChatOpenAI


def add(a: int, b: int) -> int:
    '''
    Add two numbers together.

####     Parameters {.doc-section .doc-section-----parameters}

    a : int
        The first number to add.
    b : int
        The second number to add.
    '''
    return a + b


chat = ChatOpenAI()
chat.register_tool(add)
chat.chat("What is 2 + 2?")
```

If your tool has more complex input parameters, you can provide a Pydantic model that corresponds to the input parameters for the function, This way, you can have fields that hold other model(s) (for more complex input parameters), and also more directly document the input parameters:

``` python
from chatlas import ChatOpenAI
from pydantic import BaseModel, Field


class AddParams(BaseModel):
    '''Add two numbers together.'''

    a: int = Field(description="The first number to add.")

    b: int = Field(description="The second number to add.")


def add(a: int, b: int) -> int:
    return a + b


chat = ChatOpenAI()
chat.register_tool(add, model=AddParams)
chat.chat("What is 2 + 2?")
```

## Parameters

func The function to be invoked when the tool is called. force If `True`, overwrite any existing tool with the same name. If `False` (the default), raise an error if a tool with the same name already exists. name The name of the tool. If not provided, the name will be inferred from the `func`’s name (or the `model`’s name, if provided). model A Pydantic model that describes the input parameters for the function. If not provided, the model will be inferred from the function’s type hints. The primary reason why you might want to provide a model in Note that the name and docstring of the model takes precedence over the name and docstring of the function. annotations Additional properties that describe the tool and its behavior.

## Raises

ValueError If a tool with the same name already exists and `force` is `False`.

### set_echo_options

``` python
Chat.set_echo_options(rich_markdown=None, rich_console=None, css_styles=None)
```

Set echo styling options for the chat.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| rich_markdown | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | A dictionary of options to pass to `rich.markdown.Markdown()`. This is only relevant when outputting to the console. | `None` |
| rich_console | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] | A dictionary of options to pass to `rich.console.Console()`. This is only relevant when outputting to the console. | `None` |
| css_styles | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[dict](https://docs.python.org/3/library/stdtypes.html#dict)\[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)\]\] | A dictionary of CSS styles to apply to `IPython.display.Markdown()`. This is only relevant when outputing to the browser. | `None` |

### set_model_params

``` python
Chat.set_model_params(
    temperature=MISSING,
    top_p=MISSING,
    top_k=MISSING,
    frequency_penalty=MISSING,
    presence_penalty=MISSING,
    seed=MISSING,
    max_tokens=MISSING,
    log_probs=MISSING,
    stop_sequences=MISSING,
)
```

Set common model parameters for the chat.

A unified interface for setting common model parameters across different providers. This method is useful for setting parameters that are commonly supported by most providers, such as temperature, top_p, etc.

By default, if the parameter is not set (i.e., set to `MISSING`), the provider’s default value is used. If you want to reset a parameter to its default value, set it to `None`.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| temperature | [float](https://docs.python.org/3/library/functions.html#float) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Temperature of the sampling distribution. | `MISSING` |
| top_p | [float](https://docs.python.org/3/library/functions.html#float) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | The cumulative probability for token selection. | `MISSING` |
| top_k | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | The number of highest probability vocabulary tokens to keep. | `MISSING` |
| frequency_penalty | [float](https://docs.python.org/3/library/functions.html#float) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Frequency penalty for generated tokens. | `MISSING` |
| presence_penalty | [float](https://docs.python.org/3/library/functions.html#float) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Presence penalty for generated tokens. | `MISSING` |
| seed | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Seed for random number generator. | `MISSING` |
| max_tokens | [int](https://docs.python.org/3/library/functions.html#int) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Maximum number of tokens to generate. | `MISSING` |
| log_probs | [bool](https://docs.python.org/3/library/functions.html#bool) \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | Include the log probabilities in the output? | `MISSING` |
| stop_sequences | [list](https://docs.python.org/3/library/stdtypes.html#list)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] \| None \| [MISSING_TYPE](https://posit-dev.github.io/chatlas/reference/types.MISSING_TYPE.html#chatlas.types.MISSING_TYPE) | A character vector of tokens to stop generation on. | `MISSING` |

### set_tools

``` python
Chat.set_tools(tools)
```

Set the tools for the chat.

This replaces any previously registered tools with the provided list of tools. This is for advanced usage – typically, you would use `.register_tool()` to register individual tools as needed.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| tools | [list](https://docs.python.org/3/library/stdtypes.html#list)\[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[…, [Any](https://docs.python.org/3/library/typing.html#typing.Any)\] \| [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)\[…, [Awaitable](https://docs.python.org/3/library/typing.html#typing.Awaitable)\[[Any](https://docs.python.org/3/library/typing.html#typing.Any)\]\] \| [Tool](https://posit-dev.github.io/chatlas/reference/Tool.html#chatlas.Tool)\] | A list of `Tool` instances to set as the chat’s tools. | *required* |

### set_turns

``` python
Chat.set_turns(turns)
```

Set the turns of the chat.

Replaces the current chat history state (i.e., turns) with the provided turns. This can be useful for: \* Clearing (or trimming) the chat history (i.e., `.set_turns([])`). \* Restoring context from a previous chat.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| turns | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)\[[Turn](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn)\] | The turns to set. Turns with the role “system” are not allowed. | *required* |

### stream

``` python
Chat.stream(*args, content='text', echo='none', data_model=None, kwargs=None)
```

Generate a response from the chat in a streaming fashion.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The user input(s) to generate a response from. | `()` |
| content | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['text', 'all'\] | Whether to yield just text content or include rich content objects (e.g., tool calls) when relevant. | `'text'` |
| echo | `EchoOptions` | One of the following (default is “none”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'none'` |
| data_model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[type](https://docs.python.org/3/library/functions.html#type)\[[BaseModel](https://docs.pydantic.dev/latest/api/pydantic/base_model/#pydantic.BaseModel)\]\] | A Pydantic model describing the structure of the data to extract. When provided, the response will be constrained to match this structure. The streamed chunks will be JSON text that, when concatenated, forms a valid JSON object matching the model. After consuming the stream, use `data_model.model_validate_json("".join(chunks))` to parse the result. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [Generator](https://docs.python.org/3/library/typing.html#typing.Generator) | An (unconsumed) response from the chat. Iterate over this object to consume the response. |

#### Examples

``` python
from chatlas import ChatOpenAI
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


chat = ChatOpenAI()
chunks = list(chat.stream("John is 25 years old", data_model=Person))
person = Person.model_validate_json("".join(chunks))
```

### stream_async

``` python
Chat.stream_async(
    *args,
    content='text',
    echo='none',
    data_model=None,
    kwargs=None,
)
```

Generate a response from the chat in a streaming fashion asynchronously.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The user input(s) to generate a response from. | `()` |
| content | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)\['text', 'all'\] | Whether to yield just text content or include rich content objects (e.g., tool calls) when relevant. | `'text'` |
| echo | `EchoOptions` | One of the following (default is “none”): - `"text"`: Echo just the text content of the response. - `"output"`: Echo text and tool call content. - `"all"`: Echo both the assistant and user turn. - `"none"`: Do not echo any content. | `'none'` |
| data_model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[type](https://docs.python.org/3/library/functions.html#type)\[[BaseModel](https://docs.pydantic.dev/latest/api/pydantic/base_model/#pydantic.BaseModel)\]\] | A Pydantic model describing the structure of the data to extract. When provided, the response will be constrained to match this structure. The streamed chunks will be JSON text that, when concatenated, forms a valid JSON object matching the model. After consuming the stream, use `data_model.model_validate_json("".join(chunks))` to parse the result. | `None` |
| kwargs | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[SubmitInputArgsT](https://posit-dev.github.io/chatlas/reference/types.SubmitInputArgsT.html#chatlas.types.SubmitInputArgsT)\] | Additional keyword arguments to pass to the method used for requesting the response. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [AsyncGenerator](https://docs.python.org/3/library/typing.html#typing.AsyncGenerator) | An (unconsumed) response from the chat. Iterate over this object to consume the response. |

#### Examples

``` python
from chatlas import ChatOpenAI
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


chat = ChatOpenAI()
chunks = [
    chunk
    async for chunk in await chat.stream_async(
        "John is 25 years old", data_model=Person
    )
]
person = Person.model_validate_json("".join(chunks))
```

### to_solver

``` python
Chat.to_solver(
    include_system_prompt=False,
    include_turns=False,
    data_model=None,
)
```

Create an InspectAI solver from this chat.

Translates this Chat instance into an InspectAI solver function that can be used with InspectAI’s evaluation framework. This solver will capture (and translate) important state from the chat, including the model, system prompt, previous turns, registered tools, model parameters, etc.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| data_model | [type](https://docs.python.org/3/library/functions.html#type)\[[BaseModel](https://docs.pydantic.dev/latest/api/pydantic/base_model/#pydantic.BaseModel)\] \| None | A Pydantic model describing the structure of the data to extract. When provided, the solver will use `.chat_structured()` instead of `.chat()` to generate responses, and the output completion will be JSON serialized from the model instance. | `None` |
| include_system_prompt | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to include the system prompt in the solver’s starting messages. | `False` |
| include_turns | [bool](https://docs.python.org/3/library/functions.html#bool) | Whether to include the chat’s existing turns in the solver’s starting messages. | `False` |

#### Note

Both `include_system_prompt` and `include_turns` default to `False` since `.export_eval()` captures this information already. Therefore, including them here would lead to duplication of context in the evaluation. However, in some cases you may want to include them, for example if you are manually constructing an evaluation dataset that does not include this information. Or, if you want to always have the same starting context regardless of the evaluation dataset.

#### Returns

| Name | Type | Description |
|----|----|----|
|  | An \[InspectAI solver\](https://inspect.ai-safety-institute.org.uk/solvers.html) |  |
|  | function that can be used with InspectAI's evaluation framework. |  |

#### Examples

First, put this code in a python script, perhaps named `eval_chat.py`

``` python
from chatlas import ChatOpenAI
from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import model_graded_qa

chat = ChatOpenAI(system_prompt="You are a helpful assistant.")

@task
def my_eval(grader_model: str = "openai/gpt-4o"):
    return Task(
        dataset=csv_dataset("my_eval_dataset.csv"),
        solver=chat.to_solver(),
        scorer=model_graded_qa(model=grader_model)
    )
```

Then run the evaluation with InspectAI’s CLI:

``` bash
inspect eval eval_chat.py -T --grader-model openai/gpt-4o
```

#### Note

Learn more about this method and InspectAI’s evaluation framework in the [Chatlas documentation](https://posit-dev.github.io/chatlas/misc/evals.html).

### token_count

``` python
Chat.token_count(*args, data_model=None)
```

Get an estimated token count for the given input.

Estimate the token size of input content. This can help determine whether input(s) and/or conversation history (i.e., `.get_turns()`) should be reduced in size before sending it to the model.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The input to get a token count for. | `()` |
| data_model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[type](https://docs.python.org/3/library/functions.html#type)\[[BaseModel](https://docs.pydantic.dev/latest/api/pydantic/base_model/#pydantic.BaseModel)\]\] | If the input is meant for data extraction (i.e., `.chat_structured()`), then this should be the Pydantic model that describes the structure of the data to extract. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [int](https://docs.python.org/3/library/functions.html#int) | The token count for the input. |

#### Note

Remember that the token count is an estimate. Also, models based on `ChatOpenAI()` currently does not take tools into account when estimating token counts.

#### Examples

``` python
from chatlas import ChatAnthropic

chat = ChatAnthropic()
# Estimate the token count before sending the input
print(chat.token_count("What is 2 + 2?"))

# Once input is sent, you can get the actual input and output
# token counts from the chat object
chat.chat("What is 2 + 2?", echo="none")
print(chat.token_usage())
```

### token_count_async

``` python
Chat.token_count_async(*args, data_model=None)
```

Get an estimated token count for the given input asynchronously.

Estimate the token size of input content. This can help determine whether input(s) and/or conversation history (i.e., `.get_turns()`) should be reduced in size before sending it to the model.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| args | [Content](https://posit-dev.github.io/chatlas/reference/types.Content.html#chatlas.types.Content) \| [str](https://docs.python.org/3/library/stdtypes.html#str) | The input to get a token count for. | `()` |
| data_model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[type](https://docs.python.org/3/library/functions.html#type)\[[BaseModel](https://docs.pydantic.dev/latest/api/pydantic/base_model/#pydantic.BaseModel)\]\] | If this input is meant for data extraction (i.e., `.chat_structured_async()`), then this should be the Pydantic model that describes the structure of the data to extract. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [int](https://docs.python.org/3/library/functions.html#int) | The token count for the input. |
