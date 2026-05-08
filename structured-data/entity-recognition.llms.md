# Entity recognition

The following example, which [closely inspired by the Claude documentation](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb), shows how `.chat_structured()` can be used to perform entity recognition.

``` python
from chatlas import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd

# | warning: false
text = "John works at Google in New York. He met with Sarah, the CEO of Acme Inc., last week in San Francisco."


class NamedEntity(BaseModel):
    """Named entity in the text."""

    name: str = Field(description="The extracted entity name")

    type_: str = Field(description="The entity type, e.g. 'person', 'location', 'organization'")

    context: str = Field(description="The context in which the entity appears in the text.")


class NamedEntities(BaseModel):
    """Named entities in the text."""

    entities: list[NamedEntity] = Field(description="Array of named entities")


chat = ChatOpenAI()
data = chat.chat_structured(text, data_model=NamedEntities)
pd.DataFrame([e.model_dump() for e in data.entities])
```

|  | name | type\_ | context |
|----|----|----|----|
| 0 | John | person | John works at Google in New York. |
| 1 | Google | organization | John works at Google in New York. |
| 2 | New York | location | John works at Google in New York. |
| 3 | Sarah | person | He met with Sarah, the CEO of Acme Inc., last ... |
| 4 | Acme Inc. | organization | He met with Sarah, the CEO of Acme Inc., last ... |
| 5 | San Francisco | location | He met with Sarah, the CEO of Acme Inc., last ... |
