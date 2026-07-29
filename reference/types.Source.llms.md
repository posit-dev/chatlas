# types.Source

``` python
types.Source()
```

Identity of a piece of evidence a citation or search result points to.

Subclasses set a distinct `type` and add their identity fields. Today the only concrete source is :class:`WebSource`; file/document/RAG variants are added when that support lands.
