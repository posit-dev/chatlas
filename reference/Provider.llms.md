# Provider

``` python
Provider(name, model)
```

A model provider interface for a [`Chat`](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat).

This abstract class defines the interface a model provider must implement in order to be used with a [`Chat`](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat) instance. The provider is responsible for performing the actual chat completion, and for handling the streaming of the completion results.

Note that this class is exposed for developers who wish to implement their own provider. In general, you should not need to interact with this class directly.

## Attributes

| Name                             | Description                        |
|----------------------------------|------------------------------------|
| [model](#chatlas.Provider.model) | Get the model used by the provider |
| [name](#chatlas.Provider.name)   | Get the name of the provider       |

## Methods

| Name | Description |
|----|----|
| [batch_poll](#chatlas.Provider.batch_poll) | Poll the status of a submitted batch. |
| [batch_result_turn](#chatlas.Provider.batch_result_turn) | Convert a batch result to a Turn. |
| [batch_retrieve](#chatlas.Provider.batch_retrieve) | Retrieve results from a completed batch. |
| [batch_status](#chatlas.Provider.batch_status) | Get the status of a batch. |
| [batch_submit](#chatlas.Provider.batch_submit) | Submit a batch of conversations for processing. |
| [has_batch_support](#chatlas.Provider.has_batch_support) | Returns whether this provider supports batch processing. |
| [list_models](#chatlas.Provider.list_models) | List all available models for the provider. |
| [value_cost](#chatlas.Provider.value_cost) | Compute the cost for a completion. |

### batch_poll

``` python
Provider.batch_poll(batch)
```

Poll the status of a submitted batch.

Args: batch: Batch information returned from batch_submit

Returns: Updated batch information

### batch_result_turn

``` python
Provider.batch_result_turn(result, has_data_model=False)
```

Convert a batch result to a Turn.

Args: result: Individual BatchResult from batch_retrieve has_data_model: Whether the request used a structured data model

Returns: Turn object or None if the result was an error

### batch_retrieve

``` python
Provider.batch_retrieve(batch)
```

Retrieve results from a completed batch.

Args: batch: Batch information

Returns: List of BatchResult objects, one for each request in the batch

### batch_status

``` python
Provider.batch_status(batch)
```

Get the status of a batch.

Args: batch: Batch information

Returns: BatchStatus with processing status information

### batch_submit

``` python
Provider.batch_submit(conversations, data_model=None)
```

Submit a batch of conversations for processing.

Args: conversations: List of conversation histories (each is a list of Turns) data_model: Optional structured data model for responses

Returns: BatchInfo containing batch job information

### has_batch_support

``` python
Provider.has_batch_support()
```

Returns whether this provider supports batch processing. Override this method to return True for providers that implement batch methods.

### list_models

``` python
Provider.list_models()
```

List all available models for the provider.

### value_cost

``` python
Provider.value_cost(completion, tokens=None)
```

Compute the cost for a completion.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| completion | `ChatCompletionT` | The completion object from the provider. | *required* |
| tokens | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)\[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)\] \| None | Optional pre-computed tokens tuple. If not provided, will be extracted from the completion. | `None` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | [float](https://docs.python.org/3/library/functions.html#float) \| None | The cost in USD, or None if cost cannot be computed. |
