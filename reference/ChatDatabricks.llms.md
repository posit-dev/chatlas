# ChatDatabricks

``` python
ChatDatabricks(system_prompt=None, model=None, workspace_client=None)
```

Chat with a model hosted on Databricks.

Databricks provides out-of-the-box access to a number of [foundation models](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html) and can also serve as a gateway for external models hosted by a third party.

## Prerequisites

> **NOTE:**
>
> `ChatDatabricks` requires the `databricks-sdk` package: `pip install "chatlas[databricks]"`.

> **NOTE:**
>
> `chatlas` delegates to the `databricks-sdk` package for authentication with Databricks. As such, you can use any of the authentication methods discussed here:
>
> https://docs.databricks.com/aws/en/dev-tools/sdk-python#authentication
>
> Note that Python-specific article points to this language-agnostic “unified” approach to authentication:
>
> https://docs.databricks.com/aws/en/dev-tools/auth/unified-auth
>
> There, you’ll find all the options listed, but a simple approach that generally works well is to set the following environment variables:
>
> - `DATABRICKS_HOST`: The Databricks host URL for either the Databricks workspace endpoint or the Databricks accounts endpoint.
> - `DATABRICKS_TOKEN`: The Databricks personal access token.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| system_prompt | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | A system prompt to set the behavior of the assistant. | `None` |
| model | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\[[str](https://docs.python.org/3/library/stdtypes.html#str)\] | The model to use for the chat. The default, None, will pick a reasonable default, and warn you about it. We strongly recommend explicitly choosing a model for all but the most casual use. | `None` |
| workspace_client | [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)\['WorkspaceClient'\] | A `databricks.sdk.WorkspaceClient()` to use for the connection. If not provided, a new client will be created. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | [Chat](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) | A chat object that retains the state of the conversation. |
