# promptic

### 90% of what you need for LLM app development. Nothing you don't.

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


### Resilient LLM Calls with Tenacity

`promptic` pairs perfectly with `tenacity` for handling temporary API failures, rate limits, validation errors, and so on. For example, here's how you can implement a cost-effective retry strategy that starts with smaller models:

```python
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, ValidationError
from promptic import llm

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    recommended: bool

@retry(
    # Retry only on Pydantic validation errors
    retry=retry_if_exception_type(ValidationError),
    # Try up to 3 times
    stop=stop_after_attempt(3),
)
@llm(model="gpt-3.5-turbo")  # Start with a faster, cheaper model
def analyze_movie(text) -> MovieReview:
    """Analyze this movie review and extract the key information: {text}"""

try:
    # First attempt with smaller model
    result = analyze_movie("The new Dune movie was spectacular...")
except ValidationError as e:
    # If validation fails after retries with smaller model, 
    # try one final time with a more capable model
    analyze_movie.retry.stop = stop_after_attempt(1)  # Only try once with GPT-4o
    analyze_movie.model = "gpt-4o"
    result = analyze_movie("The new Dune movie was spectacular...")

print(result)
# title='Dune' rating=9.5 summary='A spectacular sci-fi epic...' recommended=True
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
