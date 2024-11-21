# promptic

`promptic` is a lightweight, decorator-based Python library that simplifies the process of interacting with large language models (LLMs) using [litellm][litellm]. With `promptic`, you can effortlessly create prompts, handle input arguments, receive structured outputs from LLMs, and **build agents** with just a few lines of code.

## Installation

```bash
pip install promptic
```

## Usage

### Basics

```python
from promptic import llm

@llm
def president(year):
    """Who was the President of the United States in {year}?"""

print(president(2000))
# The President of the United States in 2000 was Bill Clinton until January 20th, when George W. Bush was inaugurated as the 43rd President.
```

### Structured Outputs

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


### Agents

```python
from datetime import datetime

from promptic import llm

@llm(
    system="You are a helpful assistant that manages schedules and reminders",
    model="gpt-4o-mini"
)
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

print(scheduler("Can you check my calendar for tomorrow and set a reminder for a team meeting at 2pm?"))
# checking calendar for 2023-10-04
# adding reminder: Team meeting at 2023-10-04T14:00:00
# I've checked your calendar for tomorrow, and there are no conflicts. I've also set a reminder for your team meeting at 2 PM.
```


### Streaming
The streaming feature allows real-time response generation, useful for long-form content or interactive applications:

```python
from promptic import llm

@llm(stream=True)
def generate_article(topic):
    """Write a detailed article about {topic}. Include introduction, 
    main points, and conclusion."""

for chunk in generate_article("artificial intelligence"):
    print(chunk, end="", flush=True)
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
# 2024-11-21 13:29:08,587 - promptic - INFO - promptic.py:185 - [DRY RUN]: function_name = 'turn_light_on' function_args = {}
# 2024-11-21 13:29:08,587 - promptic - INFO - promptic.py:185 - [DRY RUN]: function_name = 'get_current_weather' function_args = {'location': 'San Francisco'}
# ...
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
