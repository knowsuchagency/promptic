# promptic

[![Python Versions](https://img.shields.io/pypi/pyversions/promptic)](https://pypi.org/project/promptic)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/knowsuchagency/promptic/actions/workflows/tests.yml/badge.svg)](https://github.com/knowsuchagency/promptic/actions/workflows/tests.yml)

### 90% of what you need for LLM app development. Nothing you don't.

Promptic aims to be the "[requests](https://requests.readthedocs.io/en/latest/)" of LLM development -- the most productive and pythonic way to build LLM applications. It leverages [LiteLLM][litellm], so you're never locked in to an LLM provider and can switch to the latest and greatest with a single line of code. Promptic gets out of your way so you can focus entirely on building features.

> “Perfection is attained, not when there is nothing more to add, but when there is nothing more to take away.”

### At a glance

- 🎯 Type-safe structured outputs with Pydantic
- 🤖 Easy-to-build agents with function calling
- 🔄 Streaming support for real-time responses
- 📚 Automatic prompt caching for supported models
- 💾 Built-in conversation memory

## Installation

```bash
pip install promptic
```

## Usage

### Basics

Functions decorated with `@llm` use its docstring as a prompt template. When the function is called, promptic combines the docstring with the function's arguments to generate the prompt and returns the LLM's response.


```py
# examples/basic.py

from promptic import llm


@llm
def translate(text, language="Chinese"):
    """Translate '{text}' to {language}"""


print(translate("Hello world!"))
# 您好，世界！

print(translate("Hello world!", language="Spanish"))
# ¡Hola, mundo!


@llm(
    model="claude-3-haiku-20240307",
    system="You are a customer service analyst. Provide clear sentiment analysis with key points.",
)
def analyze_sentiment(text):
    """Analyze the sentiment of this customer feedback: {text}"""


print(analyze_sentiment("The product was okay but shipping took forever"))
# Sentiment: Mixed/Negative
# Key points:
# - Neutral product satisfaction
# - Significant dissatisfaction with shipping time

```

### Structured Outputs

You can use Pydantic models to ensure the LLM returns data in exactly the structure you expect. Simply define a Pydantic model and use it as the return type annotation on your decorated function. The LLM's response will be automatically validated against your model schema and returned as a Pydantic object.


```py
# examples/structured.py

from pydantic import BaseModel
from promptic import llm


class Forecast(BaseModel):
    location: str
    temperature: float
    units: str


@llm
def get_weather(location, units: str = "fahrenheit") -> Forecast:
    """What's the weather for {location} in {units}?"""


print(get_weather("San Francisco", units="celsius"))
# location='San Francisco' temperature=16.0 units='Celsius'

```

Alternatively, you can use JSON Schema dictionaries for more low-level validation:


```py
# examples/json_schema.py

from promptic import llm

schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[A-Z][a-z]+$",
            "minLength": 2,
            "maxLength": 20,
        },
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


@llm(json_schema=schema, system="You generate test data.")
def get_user_info(name: str) -> dict:
    """Get information about {name}"""


print(get_user_info("Alice"))
# {'name': 'Alice', 'age': 25, 'email': 'alice@example.com'}

```

### Agents

Functions decorated with `@llm.tool` become tools that the LLM can invoke to perform actions or retrieve information. The LLM will automatically execute the appropriate tool calls, creating a seamless agent interaction.


```py
# examples/book_meeting.py

from datetime import datetime

from promptic import llm


@llm(model="gpt-4o")
def scheduler(command):
    """{command}"""


@scheduler.tool
def get_current_time():
    """Get the current time"""
    print("getting current time")
    return datetime.now().strftime("%I:%M %p")


@scheduler.tool
def add_reminder(task: str, time: str):
    """Add a reminder for a specific task and time"""
    print(f"adding reminder: {task} at {time}")
    return f"Reminder set: {task} at {time}"


@scheduler.tool
def check_calendar(date: str):
    """Check calendar for a specific date"""
    print(f"checking calendar for {date}")
    return f"Calendar checked for {date}: No conflicts found"


cmd = """
What time is it?
Also, can you check my calendar for tomorrow
and set a reminder for a team meeting at 2pm?
"""

print(scheduler(cmd))
# getting current time
# checking calendar for 2023-10-05
# adding reminder: Team meeting at 2023-10-05T14:00:00
# The current time is 3:48 PM. I checked your calendar for tomorrow, and there are no conflicts. I've also set a reminder for your team meeting at 2 PM tomorrow.

```

### Streaming

The streaming feature allows real-time response generation, useful for long-form content or interactive applications:

```py
# examples/streaming.py

from promptic import llm


@llm(stream=True)
def write_poem(topic):
    """Write a haiku about {topic}."""


print("".join(write_poem("artificial intelligence")))
# Binary thoughts hum,
# Electron minds awake, learn,
# Future thinking now.

```

### Error Handling and Dry Runs

Dry runs allow you to see which tools will be called and their arguments without invoking the decorated tool functions. You can also enable debug mode for more detailed logging.

```py
# examples/error_handing.py

from promptic import llm


@llm(
    system="you are a posh smart home assistant named Jarvis",
    dry_run=True,
    debug=True,
)
def jarvis(command):
    """{command}"""


@jarvis.tool
def turn_light_on():
    """turn light on"""
    return True


@jarvis.tool
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    return f"The weather in {location} is 45 degrees {unit}"


print(jarvis("Please turn the light on and check the weather in San Francisco"))
# ...
# [DRY RUN]: function_name = 'turn_light_on' function_args = {}
# [DRY RUN]: function_name = 'get_current_weather' function_args = {'location': 'San Francisco'}
# ...

```

### Resiliency

`promptic` pairs perfectly with [tenacity](https://github.com/jd/tenacity) for handling rate limits, temporary API failures, and more.


```py
# examples/resiliency.py

from tenacity import retry, wait_exponential, retry_if_exception_type
from promptic import llm
from litellm.exceptions import RateLimitError


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
@llm
def generate_summary(text):
    """Summarize this text in 2-3 sentences: {text}"""


generate_summary("Long article text here...")

```

### Memory and State Management

By default, each function call is independent and stateless. Setting `memory=True` enables built-in conversation memory, allowing the LLM to maintain context across multiple interactions. Here's a practical example using Gradio to create a web-based chatbot interface:

```py
# examples/memory.py

import gradio as gr
from promptic import llm


@llm(memory=True, stream=True)
def assistant(message):
    """{message}"""


def predict(message, history):
    partial_message = ""
    for chunk in assistant(message):
        partial_message += str(chunk)
        yield partial_message


with gr.ChatInterface(title="Promptic Chatbot Demo", fn=predict) as demo:
    # ensure clearing the chat window clears the chat history
    demo.chatbot.clear(assistant.clear)

# demo.launch()

```

For custom storage solutions, you can extend the `State` class to implement persistence in any database or storage system:


```py
# examples/state.py

import json
from promptic import State, llm


class RedisState(State):
    def __init__(self, redis_client):
        super().__init__()
        self.redis = redis_client
        self.key = "chat_history"

    def add_message(self, message):
        self.redis.rpush(self.key, json.dumps(message))

    def get_messages(self, limit=None):
        messages = self.redis.lrange(self.key, 0, -1)
        return [json.loads(m) for m in messages][-limit:] if limit else messages

    def clear(self):
        self.redis.delete(self.key)


@llm(state=RedisState(redis_client))
def persistent_chat(message):
    """Chat: {message}"""

```

### Caching

For Anthropic models (Claude), promptic provides intelligent caching control to optimize context window usage and improve performance. By default, caching is enabled but can be disabled if needed. OpenAI models cache by default. Anthropic charges for cache writes, but tokens that are read from the cache are less expensive.


```py
# examples/caching.py

from promptic import llm

# imagine these are long legal documents
legal_document, another_legal_document = (
    "a legal document about Sam",
    "a legal document about Jane",
)

system_prompts = [
    "You are a helpful legal assistant",
    "You provide detailed responses based on the provided context",
    f"legal document 1: '{legal_document}'",
    f"legal document 2: '{another_legal_document}'",
]


@llm(
    system=system_prompts,
    cache=True,  # this is the default
)
def legal_chat(message):
    """{message}"""


print(legal_chat("which legal document is about Sam?"))
# The legal document about Sam is "legal document 1."

```


When caching is enabled:
- Long messages (>1KB) are automatically marked as ephemeral to optimize context window usage
- A maximum of 4 message blocks can be cached at once
- System prompts can include explicit cache control

Further reading:
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#cache-limitations
- https://docs.litellm.ai/docs/completion/prompt_caching


### Authentication

Authentication can be handled in three ways:

1. Directly via the `api_key` parameter:

```py
from promptic import llm

@llm(model="gpt-4o-mini", api_key="your-api-key-here")
def my_function(text):
    """Process this text: {text}"""
```

2. Through environment variables (recommended):

```bash

# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google
export GEMINI_API_KEY=...

# Azure OpenAI
export AZURE_API_KEY=...
export AZURE_API_BASE=...
export AZURE_API_VERSION=...
```

3. By setting the API key programmatically via litellm:

```py
from litellm import litellm

litellm.api_key = "your-api-key-here"
```

The supported environment variables correspond to the model provider:

| Provider  | Environment Variable | Model Examples                                                            |
| --------- | -------------------- | ------------------------------------------------------------------------- |
| OpenAI    | `OPENAI_API_KEY`     | gpt-4o, gpt-3.5-turbo                                                     |
| Anthropic | `ANTHROPIC_API_KEY`  | claude-3-haiku-20240307, claude-3-opus-20240229, claude-3-sonnet-20240229 |
| Google    | `GEMINI_API_KEY`     | gemini/gemini-1.5-pro-latest                                              |

## API Reference

### `llm`

The main decorator for creating LLM-powered functions. Can be used as `@llm` or `@llm()` with parameters.

#### Parameters

- `model` (str, optional): The LLM model to use. Defaults to "gpt-4o-mini".
- `system` (str | list[str] | list[dict], optional): System prompt(s) to set context for the LLM. Can be:
  - A single string: `system="You are a helpful assistant"`
  - A list of strings: `system=["You are a helpful assistant", "You speak formally"]`
  - A list of message dictionaries:
    ```python
    system=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Please be concise", "cache_control": {"type": "ephemeral"}},
        {"role": "assistant", "content": "I will be concise"}
    ]
    ```
- `dry_run` (bool, optional): If True, simulates tool calls without executing them. Defaults to False.
- `debug` (bool, optional): If True, enables detailed logging. Defaults to False.
- `memory` (bool, optional): If True, enables conversation memory using the default State implementation. Defaults to False.
- `state` (State, optional): Custom State implementation for memory management. Overrides the `memory` parameter.
- `json_schema` (dict, optional): JSON Schema dictionary for validating LLM outputs. Alternative to using Pydantic models.
- `cache` (bool, optional): If True, enables prompt caching. Defaults to True.
- `**litellm_kwargs`: Additional arguments passed directly to [litellm.completion](https://docs.litellm.ai/docs/completion/input).

#### Methods

- `tool(fn)`: Decorator method to register a function as a tool that can be called by the LLM.
- `clear()`: Clear all stored messages from memory. Raises ValueError if memory/state is not enabled.

### `State`

Base class for managing conversation memory and state. Can be extended to implement custom storage solutions.

#### Methods

- `add_message(message: dict)`: Add a message to the conversation history.
- `get_messages(prompt: str = None, limit: int = None) -> List[dict]`: Retrieve conversation history, optionally limited to the most recent messages and filtered by a prompt.
- `clear()`: Clear all stored messages.

#### Example


```py
# examples/api_ref.py

from pydantic import BaseModel
from promptic import llm


class Story(BaseModel):
    title: str
    content: str
    style: str
    word_count: int


@llm(
    model="gpt-4o-mini",
    system="You are a creative writing assistant",
    memory=True,
    temperature=0.7,
    max_tokens=800,
    cache=False,
)
def story_assistant(command: str) -> Story:
    """Process this writing request: {command}"""


@story_assistant.tool
def get_writing_style():
    """Get the current writing style preference"""
    return "whimsical and light-hearted"


@story_assistant.tool
def count_words(text: str) -> int:
    """Count words in the provided text"""
    return len(text.split())


story = story_assistant("Write a short story about a magical library")
print(f"Title: {story.title}")
print(f"Style: {story.style}")
print(f"Words: {story.word_count}")
print(story.content)

print(
    story_assistant("Write another story with the same style but about a time traveler")
)

```

## Limitations

`promptic` is a lightweight abstraction layer over [litellm][litellm] and its various LLM providers. As such, there are some provider-specific limitations that are beyond the scope of what the library addresses:

- **Streaming**:
  - Gemini models do not support streaming when using tools/function calls

These limitations reflect the underlying differences between LLM providers and their implementations. For provider-specific features or workarounds, you may need to interact with [litellm][litellm] or the provider's SDK directly.


## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

[litellm]: https://github.com/BerriAI/litellm
