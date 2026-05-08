# token_usage

``` python
token_usage()
```

Report on token usage in the current session

Call this function to find out the cumulative number of tokens that you have sent and received in the current session. The price will be shown if known

## Returns

| Name | Type | Description |
|----|----|----|
|  | [list](https://docs.python.org/3/library/stdtypes.html#list)\[[TokenUsage](https://posit-dev.github.io/chatlas/reference/types.TokenUsage.html#chatlas.types.TokenUsage)\] \| None | A list of dictionaries with the following keys: “name”, “input”, “output”, and “cost”. If no cost data is available for the name/model combination chosen, then “cost” will be None. If no tokens have been logged, then None is returned. |
