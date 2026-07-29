# types.ContentCitation

``` python
types.ContentCitation()
```

A source that grounds part of the assistant’s answer.

`grounded_span` is the span of the assistant’s answer this citation grounds — the words a footnote marker attaches to. It is answer-side (from the reply). `cited_quote` is the source-side evidence quote, when the provider supplies one (e.g. Anthropic web search). `source` identifies the evidence; it is `None` when the citation grounds answer text with no resolvable source.
