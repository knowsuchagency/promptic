# promptic

[![PyPI version](https://badge.fury.io/py/promptic.svg)](https://badge.fury.io/py/promptic)
[![Python Versions](https://img.shields.io/pypi/pyversions/promptic)](https://pypi.org/project/promptic/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Tests](https://github.com/knowsuchagency/promptic/actions/workflows/tests.yml/badge.svg)](https://github.com/knowsuchagency/promptic/actions/workflows/tests.yml)

### 90% of what you need for LLM app development. Nothing you don't.

`promptic` is a lightweight, decorator-based Python library that simplifies the process of interacting with large language models (LLMs) using [litellm][litellm]. With `promptic`, you can effortlessly create prompts, handle input arguments, receive structured outputs from LLMs, and **build agents** with just a few lines of code.

## Installation

```bash
pip install promptic
```

## Usage

### Basics

Functions decorated with `@llm` will automatically interpolate arguments into the prompt. You can also customize the model, system prompt, and more. Most arguments will be passed to [litellm.completion](https://docs.litellm.ai/docs/completion/input).

```python
from promptic import llm

@llm
def translate(text, target_language="Chinese"):
    """Translate this text: {text} 
    Target language: {target_language}"""

print(translate("Hello world!"))
# 您好，世界！

@llm(
    model="claude-3-haiku-20240307",
    system="You are a customer service analyst. Provide clear sentiment analysis with key points."
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

You can use Pydantic models to ensure the LLM returns data in exactly the structure you expect. Simply define a Pydantic model and use it as the return type annotation on your decorated function. The LLM's response will be automatically validated against your model schema and returned as a proper Pydantic object.

```python
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

### Agents

Functions decorated with `@llm.tool` become tools that the LLM can invoke to perform actions or retrieve information. The LLM will automatically execute the appropriate tool calls, creating a seamless agent interaction.

```python
from datetime import datetime

from promptic import llm

@llm
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

```python
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

```python
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

```python
from tenacity import retry, wait_exponential, retry_if_exception_type
from promptic import llm
from litellm.exceptions import RateLimitError

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
@llm
def generate_summary(text):
    """Summarize this text in 2-3 sentences: {text}"""

generate_summary("Long article text here...")
```

### Memory and State Management

By default, each function call is independent and stateless. Setting `memory=True` enables built-in conversation memory, allowing the LLM to maintain context across multiple interactions. For custom storage solutions, you can extend the `State` class to implement persistence in any database or storage system. This can be useful for chatbots, agents, and other applications requiring persistent context and session management.

```python
from promptic import llm, State


@llm(memory=True)
def chat(message):
    """Chat: {message}"""

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chat(user_input)
    print(f"Bot: {response}")


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

## API Reference

### `llm`

The main decorator for creating LLM-powered functions. Can be used as `@llm` or `@llm()` with parameters.

#### Parameters

- `model` (str, optional): The LLM model to use. Defaults to "gpt-4o-mini".
- `system` (str, optional): System prompt to set context for the LLM.
- `dry_run` (bool, optional): If True, simulates tool calls without executing them. Defaults to False.
- `debug` (bool, optional): If True, enables detailed logging. Defaults to False.
- `memory` (bool, optional): If True, enables conversation memory using the default State implementation. Defaults to False.
- `state` (State, optional): Custom State implementation for memory management. Overrides the `memory` parameter.
- `**litellm_kwargs`: Additional arguments passed directly to [litellm.completion](https://docs.litellm.ai/docs/completion/input).

#### Methods

- `tool(fn)`: Decorator method to register a function as a tool that can be called by the LLM.

### `State`

Base class for managing conversation memory and state. Can be extended to implement custom storage solutions.

#### Methods

- `add_message(message: dict)`: Add a message to the conversation history.
- `get_messages(limit: Optional[int] = None) -> List[dict]`: Retrieve conversation history, optionally limited to the most recent messages.
- `clear()`: Clear all stored messages.

#### Example

```python
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
    max_tokens=500,
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

# First interaction
story1 = story_assistant("Write a short story about a magical library")
print(f"Title: {story1.title}")
print(f"Style: {story1.style}")
print(f"Words: {story1.word_count}")
print(story1.content)

# Second interaction (with memory context)
story2 = story_assistant("Write another story with the same style but about a time traveler")
```

## Limitations

`promptic` is designed to be a lightweight abstraction layer over litellm and various LLM providers. As such, there are some provider-specific limitations that are beyond the scope of what the library addresses:

- **Tool/Function Calling**:
  - Anthropic (Claude) models currently support only one tool per function

- **Streaming**:
  - Gemini models do not support streaming when using tools/function calls

These limitations reflect the underlying differences between LLM providers and their implementations. For provider-specific features or workarounds, you may need to interact with litellm or the provider's SDK directly.

## License

`promptic` is open-source software licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

[litellm]: https://github.com/BerriAI/litellm
