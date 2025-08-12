---
title: System prompt
callout-appearance: simple
---

The `system_prompt` is the primary place for you (the developer) to influence the behavior of the model. 
A well-crafted system prompt steers the model toward delightful, accurate, and safe answers.
Later on, I'll offer some resources and tips for effective prompting, but for now, just know that:

* There is one `system_prompt` per `Chat` instance.
* It can be set via the `.system_prompt` property.
* It's sent to the model with every request.[^1]
* It typically remains fixed during a conversation and is invisible to the end user.
* In practice, it's often useful to create a prompt template that can regenerated with different variables.


[^1]: This is because the model is stateless, meaning it doesn't remember anything from previous requests. From a cost/efficiency perspective, repeatedly sending a large prompt is usually not a problem, especially with the help of things like [prompt caching](https://platform.openai.com/docs/guides/prompt-caching).


Here's a simple example of setting the model's role/persona using the `system_prompt`, and filling in the role using an f-string:


```python
import chatlas as ctl
chat = ctl.ChatOpenAI()

role = "Yoda"
chat.system_prompt = f"I want you to act like {role}."
chat.chat("I need your help.")
```

::: chatlas-response-container
Help you, I can. Share your troubles, you must. Guide you, I will. 
:::


As the `system_prompt` grows in size, consider moving it to (markdown) file.
Also, since more complex prompts may include things like JSON (which conflicts with f-string's `{ }` syntax), consider using a more robust templating approach.

## Templates

The [`interpolate_file()`](../reference/interpolate_file.qmd) function allows you to interpolate variables into a prompt template stored in a file.
By default, it expects variables to be wrapped using a  `{{{ x }}}` syntax, which has a much lower chance of conflicting with complex prompts in unexpected ways, and is powered by the fantastic [Jinja2](https://pypi.org/project/Jinja2/) templating engine:

```python
with open('prompt.md', 'w') as f:
    f.write('I want you to act like {{ role }}')
```

```python
import chatlas as ctl
chat = ctl.ChatOpenAI()

chat.system_prompt = ctl.interpolate_file(
  "prompt.md",
  variables={"role": "Yoda"}
)
print(chat.system_prompt)
```

```
I want you to act like Yoda.
```


As you iterate on your prompt, you'll want to keep with a small set of challenging/important examples that you can regularly re-check with your latest version of the prompt.
Writing automated tests for this can be challenging since LLMs aren't deterministic by nature.[^2]
Eventually, you may want a more systematic way to [evaluate](#evaluating-prompts) the prompt to ensure it continues to produce quality output.

[^2]: Some model providers (e.g., [ChatOpenAI](../reference/ChatOpenAI.qmd)) allow you to set a random seed, but this feature is not available for all model providers.


## Inspiration and guides

Nowadays there are many prompt "libraries" and "guides" available online, offering a wide-array of inspiration and intuition for writing your own prompts.

I particular like Anthropic's [prompt library](https://docs.anthropic.com/en/prompt-library/).
If you have an Anthropic account, this also pairs well with their prompt [generator](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-generator), [improver](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver), and more generally their [prompting guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview). 
Although the guide is specifically for Anthropic, I suspect it will be useful for other model providers as well.
That said, other major players like [OpenAI](https://platform.openai.com/docs/guides/prompt-engineering) and [Google](https://ai.google.dev/gemini-api/docs/prompting-intro) have their own prompting guides, which are worth checking out as well.

If you've never written a prompt, I also recommend reading Ethan Mollick's [Getting started with AI: Good enough prompting](https://www.oneusefulthing.org/p/getting-started-with-ai-good-enough). This quote in particular has some sage advice for how to think about prompting in general:

> Treat AI like an infinitely patient **new coworker** who **forgets everything** you tell them each new conversation, one that comes **highly recommended** but whose actual abilities are not that clear. Two parts of this are analogous to working with humans (being new on the job and being a coworker) and two of them are very alien (forgetting everything and being infinitely patient). We should **start with where AIs are closest to humans, because that is the key to good-enough prompting**


## Various tips

This section offers some specific tips for writing effective system prompts.
These tips are based on our own experience on projects like [querychat](https://github.com/posit-dev/querychat) (a tool for running SQL queries through a chat interface).

### Set the scene

To help the LLM produce output that feels natural for the end user, it can be helpful to explain the how the user will interact with the LLM (e.g., "You are a chatbot displayed in a dashboard").

### Define a purpose

Give the LLM a sense of what the user is trying to accomplish.
For example, [querychat](https://github.com/posit-dev/querychat)'s prompt includes the phrase "You will be asked to perform various tasks on the data, such as filtering, sorting, and answering questions."
This helps the model understand that it should be focused on data analysis and not just general conversation.

### Influence behavior

LLMs tend to optimize for user satisfaction, which unfortunately means they are often overly agreeable, and aren't necessarily concerned about accuracy.
When accuracy is paramount, include instructions that encourage the model to be more cautious.
For example, you might say "Only answer if you are 100% sure of the answer, otherwise say 'I don't know' or 'I'm not sure'."

LLMs can also be overly complimentary, verbose, and polite.
For example, if you ask a model to summarize a long document, it may include a lot of unnecessary praise or compliments.
To reduced this behavior, include instructions that encourage the model to be more concise and focused on the task at hand.

LLMs also tend to provide an answer even when it's not clear what the user is asking.
In this case, it can help to include instructions that encourage the model to ask for clarification before providing an answer.

### Use specific examples

Models tend to perform better when they have specific examples to use as a reference.
The examples can help reinforce what is "good" vs "bad" behavior as well as when and how to perform certain [tasks](#tasks).
It's also helpful exercise for you, the developer, to explain more precisely about how you want the model to behave in certain situations.

### Outline tasks {#tasks}

If the LLM is equipped with [tools](tools.qmd), you may want to be explicit about when and how to use them.
Hopefully each tool has a docstring that explains what it does, but you may also want to include some specific examples of when to use each tool.
It can also help to explain how the tool behavior influences with the larger user experience (e.g., what you plan on displaying to the user -- if anything -- when a tool is used).

### Provide missing info

LLMs are trained on a wide variety of data, but they don't necessarily have access to real-time or proprietary information.
When the amount of "missing" information can reasonably fit in a context window, it's often best to just include that information in the system prompt.
For example, if you're building a chatbot that needs to know about a specific table schema (like [querychat](https://github.com/posit-dev/querychat) does) to offer accurate SQL queries, include that information in the system prompt.
However, if the amount of missing information takes up a significant portion of a context window, it may be better to use a different approach like [RAG](../misc/RAG.qmd) or [tool calling](../tool-calling/how-it-works.qmd).

### Follow-up suggestions

LLMs are quite good at suggesting follow-up user prompts to explore an idea (or new ideas) further.
They tend to do this by default, but you may want to encourage offering multiple suggestions, and to be more specific about the types of follow-up questions that would be useful.
This is especially useful when the model is being used in a chat interface, where the user may not know what to ask next.
Also, with a web frameworks like [Shiny](https://shiny.rstudio.com/py), it's easy to turn these into [input suggestions](https://shiny.posit.co/py/docs/genai-chatbots.html#recommend-input) (i.e., links the user can click to quickly ask the model a follow-up question).

<!--
TODO: show a concrete example of doing this.

## Evaluating prompts

chatlas itself doesn't come with a evaludation framework, but it's fairly straightforward to feed the chatlas model and prompt to use something like [`inspect-ai`](https://pypi.org/project/inspect-ai/) for evaluation.
-->