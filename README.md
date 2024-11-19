# promptic

`promptic` is a lightweight, decorator-based Python library that simplifies the process of interacting with large language models (LLMs) using [litellm][litellm]. With `promptic`, you can effortlessly create prompts, handle input arguments, receive structured outputs from LLMs, and enable function/tool calling capabilities with just a few lines of code.

## Installation

```bash
pip install promptic
```

## Usage

### Simple Prompt

```python
from promptic import llm

@llm
def president(year):
    """Who was the President of the United States in {year}?"""

print(president(2000))
# The President of the United States in 2000 was Bill Clinton until January 20th, when George W. Bush was inaugurated as the 43rd President.
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from promptic import llm

class Capital(BaseModel):
    country: str
    capital: str

@llm
def capital(country) -> Capital:
    """What's the capital of {country}?"""

print(capital("France"))
# country='France' capital='Paris'
```

### Streaming Response (and [litellm][litellm] integration)

```python
from promptic import llm

# most arguments are passed directly to litellm.completion
# see https://docs.litellm.ai/docs/completion

@llm(stream=True, model="claude-3-haiku-20240307")
def haiku(subject, adjective, verb="delights"):
    """Write a haiku about {subject} that is {adjective} and {verb}."""

print("".join(haiku("programming", adjective="witty")))
# Bits and bytes abound,
# Bugs and features intertwine,
# Code, the poet's rhyme.
```

### Customize System Prompt

```python
from promptic import llm

@llm(system="you are a snarky chatbot")
def answer(question):
    """{question}"""

print(answer("What's the best programming language?"))
# Well, that's like asking what's the best flavor of ice cream. 
# It really depends on what you're trying to accomplish and your personal preferences. 
# But if you want to start a flame war, just bring up Python vs JavaScript.
```

### Agents

```python
from promptic import llm

@llm(system="you are a posh smart home assistant named Jarvis")
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
# Certainly, sir. I'll assist you with that right away.
# I've turned the light on for you. As for the weather in San Francisco,
# it is currently 45 degrees fahrenheit.
```

## Features

- **Decorator-based API**: Easily define prompts using function docstrings and decorate them with `@promptic.llm`.
- **Argument interpolation**: Automatically interpolate function arguments into the prompt using `{argument_name}` placeholders within docstrings.
- **Pydantic model support**: Specify the expected output structure using Pydantic models, and `promptic` will ensure the LLM's response conforms to the defined schema.
- **Streaming support**: Receive LLM responses in real-time by setting `stream=True` when calling the decorated function.
- **Simplified LLM interaction**: No need to remember the exact shape of the OpenAPI response object or other LLM-specific details. `promptic` abstracts away the complexities, allowing you to focus on defining prompts and receiving structured outputs.
- **Build Agents Seamlessly**: Decorate functions as tools that the LLM can use to perform actions or retrieve information.


## Why promptic?

`promptic` is designed to be simple, functional, and robust, providing exactly what you need 90% of the time when working with LLMs. It eliminates the need to remember the specific shapes of OpenAPI response objects or other LLM-specific details, allowing you to focus on creating prompts and receiving structured outputs.

With its legible and concise codebase, `promptic` is reliable easy to understand. It leverages the power of [litellm][litellm] under the hood, ensuring compatibility with a wide range of LLMs.

## License

`promptic` is open-source software licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

[litellm]: https://github.com/BerriAI/litellm
