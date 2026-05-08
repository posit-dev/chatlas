# Function reference

## Chat model providers

Start a chat with a particular large language model (llm) provider.

|  |  |
|----|----|
| [ChatAnthropic](../reference/ChatAnthropic.llms.md#chatlas.ChatAnthropic) | Chat with an Anthropic Claude model. |
| [ChatAuto](../reference/ChatAuto.llms.md#chatlas.ChatAuto) | Chat with any provider. |
| [ChatAzureOpenAI](../reference/ChatAzureOpenAI.llms.md#chatlas.ChatAzureOpenAI) | Chat with a model hosted on Azure OpenAI. |
| [ChatBedrockAnthropic](../reference/ChatBedrockAnthropic.llms.md#chatlas.ChatBedrockAnthropic) | Chat with an AWS bedrock model. |
| [ChatCloudflare](../reference/ChatCloudflare.llms.md#chatlas.ChatCloudflare) | Chat with a model hosted on Cloudflare Workers AI. |
| [ChatDatabricks](../reference/ChatDatabricks.llms.md#chatlas.ChatDatabricks) | Chat with a model hosted on Databricks. |
| [ChatDeepSeek](../reference/ChatDeepSeek.llms.md#chatlas.ChatDeepSeek) | Chat with a model hosted on DeepSeek. |
| [ChatGithub](../reference/ChatGithub.llms.md#chatlas.ChatGithub) | Chat with a model hosted on the GitHub model marketplace. |
| [ChatGoogle](../reference/ChatGoogle.llms.md#chatlas.ChatGoogle) | Chat with a Google Gemini model. |
| [ChatGroq](../reference/ChatGroq.llms.md#chatlas.ChatGroq) | Chat with a model hosted on Groq. |
| [ChatHuggingFace](../reference/ChatHuggingFace.llms.md#chatlas.ChatHuggingFace) | Chat with a model hosted on Hugging Face Inference API. |
| [ChatMistral](../reference/ChatMistral.llms.md#chatlas.ChatMistral) | Chat with a model hosted on Mistral’s La Plateforme. |
| [ChatOllama](../reference/ChatOllama.llms.md#chatlas.ChatOllama) | Chat with a local Ollama model. |
| [ChatOpenAI](../reference/ChatOpenAI.llms.md#chatlas.ChatOpenAI) | Chat with an OpenAI model using the responses API. |
| [ChatOpenRouter](../reference/ChatOpenRouter.llms.md#chatlas.ChatOpenRouter) | Chat with one of the many models hosted on OpenRouter. |
| [ChatPerplexity](../reference/ChatPerplexity.llms.md#chatlas.ChatPerplexity) | Chat with a model hosted on perplexity.ai. |
| [ChatPortkey](../reference/ChatPortkey.llms.md#chatlas.ChatPortkey) | Chat with a model hosted on PortkeyAI |
| [ChatSnowflake](../reference/ChatSnowflake.llms.md#chatlas.ChatSnowflake) | Chat with a Snowflake Cortex LLM |
| [ChatVertex](../reference/ChatVertex.llms.md#chatlas.ChatVertex) | Chat with a Google Vertex AI model. |

## The chat object

Methods and attributes available on a chat instance

|  |  |
|----|----|
| [Chat](../reference/Chat.llms.md#chatlas.Chat) | A chat object that can be used to interact with a language model. |

## Image input

Submit image input to the chat

|  |  |
|----|----|
| [content_image_file](../reference/content_image_file.llms.md#chatlas.content_image_file) | Encode image content from a file for chat input. |
| [content_image_plot](../reference/content_image_plot.llms.md#chatlas.content_image_plot) | Encode the current matplotlib plot as an image for chat input. |
| [content_image_url](../reference/content_image_url.llms.md#chatlas.content_image_url) | Encode image content from a URL for chat input. |

## PDF input

Submit pdf input to the chat

|  |  |
|----|----|
| [content_pdf_file](../reference/content_pdf_file.llms.md#chatlas.content_pdf_file) | Prepare a local PDF for input to a chat. |
| [content_pdf_url](../reference/content_pdf_url.llms.md#chatlas.content_pdf_url) | Use a remote PDF for input to a chat. |

## Tool calling

Add context to python function before registering it as a tool.

|  |  |
|----|----|
| [Tool](../reference/Tool.llms.md#chatlas.Tool) | Define a tool |
| [ToolRejectError](../reference/ToolRejectError.llms.md#chatlas.ToolRejectError) | Error to represent a tool call being rejected. |

## Built-in tools

Provider-agnostic access to built-in web search and fetch capabilities.

|  |  |
|----|----|
| [tool_web_search](../reference/tool_web_search.llms.md#chatlas.tool_web_search) | Create a web search tool for use with chat models. |
| [tool_web_fetch](../reference/tool_web_fetch.llms.md#chatlas.tool_web_fetch) | Create a URL fetch tool for use with chat models. |

## Parallel and batch chat

Submit multiple chats in parallel (fast) or one batch (cheap)

|  |  |
|----|----|
| [parallel_chat](../reference/parallel_chat.llms.md#chatlas.parallel_chat) | Submit multiple chat prompts in parallel. |
| [parallel_chat_text](../reference/parallel_chat_text.llms.md#chatlas.parallel_chat_text) | Submit multiple chat prompts in parallel and return text responses. |
| [parallel_chat_structured](../reference/parallel_chat_structured.llms.md#chatlas.parallel_chat_structured) | Submit multiple chat prompts in parallel and extract structured data. |
| [batch_chat](../reference/batch_chat.llms.md#chatlas.batch_chat) | Submit multiple chat requests in a batch. |
| [batch_chat_text](../reference/batch_chat_text.llms.md#chatlas.batch_chat_text) | Submit multiple chat requests in a batch and return text responses. |
| [batch_chat_structured](../reference/batch_chat_structured.llms.md#chatlas.batch_chat_structured) | Submit multiple structured data requests in a batch. |
| [batch_chat_completed](../reference/batch_chat_completed.llms.md#chatlas.batch_chat_completed) | Check if a batch job is completed without waiting. |

## Prompt interpolation

Interpolate variables into prompt templates

|  |  |
|----|----|
| [interpolate](../reference/interpolate.llms.md#chatlas.interpolate) | Interpolate variables into a prompt |
| [interpolate_file](../reference/interpolate_file.llms.md#chatlas.interpolate_file) | Interpolate variables into a prompt from a file |

## Turns

A provider-agnostic representation of content generated during an assistant/user turn.

|  |  |
|----|----|
| [AssistantTurn](../reference/AssistantTurn.llms.md#chatlas.AssistantTurn) | Assistant turn - represents model response with additional metadata |
| [UserTurn](../reference/UserTurn.llms.md#chatlas.UserTurn) | User turn - represents user input |
| [SystemTurn](../reference/SystemTurn.llms.md#chatlas.SystemTurn) | System turn - represents system prompt |
| [Turn](../reference/Turn.llms.md#chatlas.Turn) | Base turn class |

## Query token usage

|  |  |
|----|----|
| [token_usage](../reference/token_usage.llms.md#chatlas.token_usage) | Report on token usage in the current session |

## Implement a model provider

|  |  |
|----|----|
| [Provider](../reference/Provider.llms.md#chatlas.Provider) | A model provider interface for a [`Chat`](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat). |

## User-facing types

|  |  |
|----|----|
| [types.Content](../reference/types.Content.llms.md#chatlas.types.Content) | Base class for all content types that can be appear in a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) |
| [types.ContentImage](../reference/types.ContentImage.llms.md#chatlas.types.ContentImage) | Base class for image content. |
| [types.ContentImageInline](../reference/types.ContentImageInline.llms.md#chatlas.types.ContentImageInline) | Inline image content. |
| [types.ContentImageRemote](../reference/types.ContentImageRemote.llms.md#chatlas.types.ContentImageRemote) | Image content from a URL. |
| [types.ContentJson](../reference/types.ContentJson.llms.md#chatlas.types.ContentJson) | JSON content |
| [types.ContentText](../reference/types.ContentText.llms.md#chatlas.types.ContentText) | Text content for a [`Turn`](https://posit-dev.github.io/chatlas/reference/Turn.html#chatlas.Turn) |
| [types.ContentThinking](../reference/types.ContentThinking.llms.md#chatlas.types.ContentThinking) | Thinking/reasoning content |
| [types.ContentThinkingDelta](../reference/types.ContentThinkingDelta.llms.md#chatlas.types.ContentThinkingDelta) | A streaming fragment of thinking/reasoning content. |
| [types.ContentToolRequest](../reference/types.ContentToolRequest.llms.md#chatlas.types.ContentToolRequest) | A request to call a tool/function |
| [types.ContentToolResult](../reference/types.ContentToolResult.llms.md#chatlas.types.ContentToolResult) | The result of calling a tool/function |
| [types.ContentToolRequestSearch](../reference/types.ContentToolRequestSearch.llms.md#chatlas.types.ContentToolRequestSearch) | A web search request from the model. |
| [types.ContentToolResponseSearch](../reference/types.ContentToolResponseSearch.llms.md#chatlas.types.ContentToolResponseSearch) | Web search results from the model. |
| [types.ContentToolRequestFetch](../reference/types.ContentToolRequestFetch.llms.md#chatlas.types.ContentToolRequestFetch) | A web fetch request from the model. |
| [types.ContentToolResponseFetch](../reference/types.ContentToolResponseFetch.llms.md#chatlas.types.ContentToolResponseFetch) | Web fetch results from the model. |
| [types.ChatResponse](../reference/types.ChatResponse.llms.md#chatlas.types.ChatResponse) | Chat response object. |
| [types.ChatResponseAsync](../reference/types.ChatResponseAsync.llms.md#chatlas.types.ChatResponseAsync) | Chat response (async) object. |
| [types.ImageContentTypes](../reference/types.ImageContentTypes.llms.md#chatlas.types.ImageContentTypes) | Allowable content types for images. |
| [types.MISSING_TYPE](../reference/types.MISSING_TYPE.llms.md#chatlas.types.MISSING_TYPE) | A singleton representing a missing value. |
| [types.MISSING](../reference/types.MISSING.llms.md#chatlas.types.MISSING) |  |
| [types.SubmitInputArgsT](../reference/types.SubmitInputArgsT.llms.md#chatlas.types.SubmitInputArgsT) | A TypedDict representing the provider specific arguments that can specified when |
| [types.TokenUsage](../reference/types.TokenUsage.llms.md#chatlas.types.TokenUsage) | Token usage for a given provider (name). |
| [types.ToolAnnotations](../reference/types.ToolAnnotations.llms.md#chatlas.types.ToolAnnotations) | Additional properties describing a Tool to clients. |
| [types.ToolInfo](../reference/types.ToolInfo.llms.md#chatlas.types.ToolInfo) | Serializable tool information |
