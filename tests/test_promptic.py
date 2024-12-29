import logging
from unittest.mock import Mock
import subprocess as sp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys

import pytest
from litellm.exceptions import RateLimitError, InternalServerError, APIError, Timeout
from pydantic import BaseModel
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
)

from promptic import Promptic, State, llm

ERRORS = (RateLimitError, InternalServerError, APIError, Timeout)

# Define default model lists
CHEAP_MODELS = ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini/gemini-1.5-flash"]
REGULAR_MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini/gemini-1.5-pro"]


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_basic(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, timeout=5)
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_parens(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, timeout=5)
    def vice_president(year):
        """Who was the Vice President of the United States in {year}?"""

    result = vice_president(2001)
    assert "Dick Cheney" in result
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_pydantic(model):
    class Capital(BaseModel):
        country: str
        capital: str

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, timeout=5)
    def capital(country) -> Capital:
        """What's the capital of {country}?"""

    result = capital("France")
    assert result.country == "France"
    assert result.capital == "Paris"


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_streaming(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        stream=True,
        model=model,
        temperature=0,
        timeout=5,
    )
    def haiku(subject, adjective, verb="delights"):
        """Write a haiku about {subject} that is {adjective} and {verb}."""

    result = "".join(haiku("programming", adjective="witty"))
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_system_prompt(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(system="you are a snarky chatbot", temperature=0, model=model, timeout=5)
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_system_prompt_list_strings(model):
    system_prompts = [
        "you are a helpful assistant",
        "you always provide concise answers",
        "you speak in a formal tone",
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(system=system_prompts, temperature=0, model=model, timeout=5)
    def answer(question):
        """{question}"""

    result = answer("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result
    # Should be concise due to system prompt
    assert len(result.split()) < 30


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_system_prompt_list_dicts(model):
    system_prompts = [
        {"role": "system", "content": "you are a helpful assistant"},
        {
            "role": "system",
            "content": "you always provide concise answers",
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "system", "content": "you speak in a formal tone"},
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(system=system_prompts, temperature=0, model=model, timeout=5)
    def answer(question):
        """{question}"""

    result = answer("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result
    # Should be concise due to system prompt
    assert len(result.split()) < 30


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_agents(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system="you are a posh smart home assistant named Jarvis",
        temperature=0,
        model=model,
        timeout=5,
    )
    def jarvis(command):
        """{command}"""

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @jarvis.tool
    def turn_light_on():
        """turn light on"""
        print("turning light on")
        return True

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @jarvis.tool
    def get_current_weather(location: str, unit: str = "fahrenheit"):
        """Get the current weather in a given location"""
        return f"The weather in {location} is 45 degrees {unit}"

    result = jarvis(
        "Please turn the light on Jarvis. By the way, what is the weather in San Francisco?"
    )

    assert isinstance(result, str)

    probable_weather_words = [
        "weather",
        "temperature",
        "currently",
        "fahrenheit",
    ]

    assert any(word in result.lower() for word in probable_weather_words)


@pytest.mark.parametrize("model", REGULAR_MODELS)
def test_streaming_with_tools(model):
    if model.startswith(("gemini", "vertex")):  # pragma: no cover
        pytest.skip("Gemini models do not support streaming with tools")

    time_mock = Mock(return_value="12:00 PM")
    weather_mock = Mock(return_value="Sunny in Paris")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        stream=True,
        model=model,
        system="you are a helpful assistant",
        temperature=0,
        timeout=5,
    )
    def stream_with_tools(query):
        """{query}"""

    @stream_with_tools.tool
    def get_time():
        """Get the current time"""
        return time_mock()

    @stream_with_tools.tool
    def get_weather(location: str):
        """Get the weather for a location"""
        return weather_mock(location)

    result = "".join(
        stream_with_tools("What time is it and what's the weather in Paris?")
    )

    assert isinstance(result, str)
    time_mock.assert_called_once()
    weather_mock.assert_called_once()


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_json_schema_validation(model):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
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
    }

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, json_schema=schema, timeout=5)
    def get_user_info(name: str):
        """Get information about {name}"""

    result = get_user_info("Alice")
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result
    assert isinstance(result["age"], int)

    invalid_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 1000,
                "maximum": 1,
                "multipleOf": 0.5,
            }
        },
        "required": ["score"],
        "additionalProperties": False,
    }

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, json_schema=invalid_schema, timeout=5)
    def get_impossible_score(name: str):
        """Get score for {name}"""

    with pytest.raises(ValueError) as exc_info:
        get_impossible_score("Alice")
    assert "Schema validation failed" in str(exc_info.value)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_dry_run_with_tools(model, caplog):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(dry_run=True, debug=True, temperature=0, model=model, timeout=5)
    def assistant(command):
        """{command}"""

    @assistant.tool
    def initialize_switch():
        """Initialize a switch"""
        raise Exception("This should not be called")

    with caplog.at_level(logging.DEBUG, logger="promptic"):
        assistant("Please initialize the switch")

    assert any("[DRY RUN]" in record.message for record in caplog.records)
    assert any("initialize_switch" in record.message for record in caplog.records)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_debug_logging(model, caplog):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(debug=True, temperature=0, model=model, timeout=5)
    def debug_test(message):
        """Echo: {message}"""

    with caplog.at_level(logging.DEBUG, logger="promptic"):
        debug_test("hello")

    assert any("model =" in record.message for record in caplog.records)
    assert any("hello" in record.message for record in caplog.records)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_multiple_tool_calls(model):
    counter = Mock()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system="You are a helpful assistant that likes to double-check things",
        temperature=0,
        model=model,
        timeout=5,
    )
    def double_checker(query):
        """{query}"""

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @double_checker.tool
    def check_status():
        """Check the current status"""
        counter()
        return "Status OK"

    double_checker("Please check the status twice to be sure")
    # Ensure we tested an even number of times with retries
    assert counter.call_count // 2 * 2 == counter.call_count
    assert counter.call_count >= 2  # At least called twice


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_state_basic(model):
    state = State()
    message = {"role": "user", "content": "Hello"}

    state.add_message(message)
    assert state.get_messages() == [message]

    state.clear()
    assert state.get_messages() == []


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_state_limit(model):
    state = State()
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(3)]

    for msg in messages:
        state.add_message(msg)

    assert state.get_messages(limit=2) == messages[-2:]
    assert state.get_messages() == messages


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_memory_conversation(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(memory=True, temperature=0, model=model, timeout=5)
    def chat(message):
        """Chat: {message}"""

    # First message should be stored
    result1 = chat("What is the capital of France?")
    assert "Paris" in result1

    # Second message should reference the context from first
    result2 = chat("What did I just ask about?")
    assert "france" in result2.lower() or "paris" in result2.lower()


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_custom_state(model):
    class TestState(State):
        def __init__(self):
            super().__init__()
            self.cleared = False

        def clear(self):
            self.cleared = True
            super().clear()

    custom_state = TestState()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(state=custom_state, temperature=0, model=model, timeout=5)
    def chat(message):
        """Chat: {message}"""

    result = chat("Hello")
    assert len(custom_state.get_messages()) > 0

    custom_state.clear()
    assert custom_state.cleared
    assert len(custom_state.get_messages()) == 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_memory_disabled(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(memory=False, temperature=0, model=model, timeout=5)
    def chat(message):
        """Chat: {message}"""

    result1 = chat("What is the capital of France?")
    result2 = chat("What did I just ask about?")

    # Without memory, the second response shouldn't mention France
    assert not ("france" in result2.lower() or "paris" in result2.lower())


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_memory_with_streaming(model):
    state = State()
    p = Promptic(
        model=model,
        memory=True,
        state=state,
        stream=True,
        debug=True,
        temperature=0,
        timeout=5,
    )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @p
    def simple_conversation(input_text: str) -> str:
        """Just respond to: {input_text}"""
        return str

    # Simulate a conversation with streaming
    response_stream = simple_conversation("Hello!")
    # Consume the stream
    response = "".join(list(response_stream))

    # Verify first exchange is stored (both user and assistant messages)
    assert len(state.get_messages()) == 2
    assert state.get_messages()[0]["role"] == "user"
    assert state.get_messages()[1]["role"] == "assistant"
    assert state.get_messages()[1]["content"] == response

    # Second message
    response_stream = simple_conversation("How are you?")
    response2 = "".join(list(response_stream))

    # Verify both exchanges are stored
    assert len(state.get_messages()) == 4  # 2 user messages + 2 assistant responses
    assert state.get_messages()[2]["role"] == "user"
    assert state.get_messages()[3]["role"] == "assistant"
    assert state.get_messages()[3]["content"] == response2

    # Verify messages are in correct order
    messages = state.get_messages()
    assert messages[1]["content"] == response  # First assistant response
    assert messages[3]["content"] == response2  # Second assistant response


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_pydantic_with_tools(model):
    class WeatherReport(BaseModel):
        location: str
        temperature: float
        conditions: str

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, timeout=5)
    def get_weather_report(location: str) -> WeatherReport:
        """Create a detailed weather report for {location}"""

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @get_weather_report.tool
    def get_temperature(city: str) -> float:
        """Get the temperature for a city"""
        return 72.5

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @get_weather_report.tool
    def get_conditions(city: str) -> str:
        """Get the weather conditions for a city"""
        return "Sunny with light clouds"

    result = get_weather_report("San Francisco")
    assert isinstance(result, WeatherReport)
    assert "San Francisco" in result.location
    assert isinstance(result.temperature, float)
    assert isinstance(result.conditions, str)


@pytest.mark.parametrize("model", REGULAR_MODELS)
def test_pydantic_tools_with_memory(model):
    class TaskStatus(BaseModel):
        task_id: int
        status: str
        last_update: str

    state = State()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(memory=True, state=state, temperature=0, model=model, timeout=5)
    def task_tracker(command: str) -> TaskStatus:
        """Process the following task command: {command}"""

    @task_tracker.tool
    def get_task_status(task_id: int) -> str:
        """Get the current status of a task"""
        return "in_progress"

    @task_tracker.tool
    def get_last_update(task_id: int) -> str:
        """Get the last update timestamp for a task"""
        return "2024-03-15 10:00 AM"

    # First interaction
    result1 = task_tracker("Check status of task 123")
    assert isinstance(result1, TaskStatus)
    assert result1.task_id == 123
    assert result1.status == "in_progress"

    # Second interaction should have context from the first
    result2 = task_tracker("What was the last task we checked?")
    assert isinstance(result2, TaskStatus)
    assert result2.task_id == 123  # Should reference the previous task


def test_anthropic_tool_calling():
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        model="claude-3-haiku-20240307",
        temperature=0,
        debug=True,
        timeout=5,
    )
    def assistant(command):
        """{command}"""

    @assistant.tool
    def get_time():
        """Get the current time"""
        return "12:00 PM"

    result = assistant("What time is it?")

    assert isinstance(result, str)
    assert "12:00" in result


# Add new test to verify Gemini streaming with tools raises exception
def test_gemini_streaming_with_tools_error():
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(stream=True, model="gemini/gemini-1.5-pro", timeout=5)
    def assistant(command):
        """{command}"""

    @assistant.tool
    def get_time():
        """Get the current time"""
        return "12:00 PM"

    with pytest.raises(ValueError) as exc_info:
        next(assistant("What time is it?"))

    assert str(exc_info.value) == "Gemini models do not support streaming with tools"


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_mutually_exclusive_schemas(model):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    class Person(BaseModel):
        name: str
        age: int

    with pytest.raises(ValueError) as exc_info:

        @llm(temperature=0, model=model, json_schema=schema, timeout=5)
        def get_person(name: str) -> Person:
            """Get information about {name}"""

    assert (
        str(exc_info.value)
        == "Cannot use both Pydantic return type hints and json_schema validation together"
    )


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_wrapper_attributes(model):
    custom_state = State()
    p = Promptic(
        model=model,
        temperature=0.7,
        memory=True,
        system="test system prompt",
        stream=True,
        debug=True,
        state=custom_state,
        timeout=5,
    )

    @p
    def test_function(input_text: str):
        """Test: {input_text}"""

    assert hasattr(test_function, "tool")
    assert callable(test_function.tool)

    assert test_function.model == model
    assert test_function.memory is True
    assert test_function.system == "test system prompt"
    assert test_function.debug is True
    assert test_function.state is custom_state

    assert test_function.litellm_kwargs == {
        "temperature": 0.7,
        "stream": True,
        "timeout": 5,
    }


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_clear_state(model):
    # Test successful clearing
    state = State()

    @llm(model=model, memory=True, state=state, timeout=5)
    def chat(message):
        """Chat: {message}"""

    # Add some messages
    chat("Hello")
    assert len(state.get_messages()) > 0

    # Clear the state
    chat.clear()
    assert len(state.get_messages()) == 0

    # Test error when memory/state is disabled

    @llm(model=model, memory=False, timeout=5)
    def chat_no_memory(message):
        """Chat: {message}"""

    with pytest.raises(ValueError) as exc_info:
        chat_no_memory.clear()
    assert "Cannot clear state: memory/state is not enabled" in str(exc_info.value)


def _get_example_files():
    """Get all example Python files."""
    return map(str, Path("examples").glob("*.py"))


@pytest.mark.parametrize("example_file", _get_example_files())
def test_examples(example_file):
    """Run each example file."""
    if example_file == "examples/memory.py":
        sp.run(f"uv run --with gradio {example_file}", shell=True, check=True)
    elif example_file == "examples/state.py":
        pytest.skip("State example is not runnable without Redis.")
    else:
        sp.run(f"uv run {example_file}", shell=True, check=True)


@pytest.mark.parametrize("model", REGULAR_MODELS)
def test_weather_tools_basic(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, timeout=5, debug=True)
    def weather_assistant(command):
        """{command}"""

    @weather_assistant.tool
    def get_location(city: str) -> dict:
        """Get latitude and longitude based on city name"""
        locations = {
            "New York": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city": "New York",
            },
            "Miami": {"latitude": 25.7617, "longitude": -80.1918, "city": "Miami"},
        }
        return locations.get(city, {"error": "Location not found"})

    @weather_assistant.tool
    def get_weather(latitude: float, longitude: float) -> dict:
        """Get weather based on latitude and longitude"""
        if latitude > 35:  # Northern region
            return {
                "temperature": 59,
                "condition": "sunny",
                "humidity": 50,
                "wind_speed": 8,
            }
        else:  # Southern region
            return {
                "temperature": 77,
                "condition": "cloudy",
                "humidity": 80,
                "wind_speed": 6,
            }

    # Test single city weather
    result1 = weather_assistant("How's the weather in New York right now?")
    assert isinstance(result1, str)
    assert any(word in result1.lower() for word in ["new york", "59", "sunny"])

    # Test weather comparison
    result2 = weather_assistant("Please compare the weather between New York and Miami")
    assert isinstance(result2, str)
    assert all(city in result2.lower() for city in ["new york", "miami"])
    assert any(str(temp) in result2 for temp in ["59", "77"])

    # Test temperature difference
    result3 = weather_assistant(
        "What's the temperature difference between New York and Miami?"
    )
    assert isinstance(result3, str)
    assert "18" in result3  # 77 - 59 = 18 degrees difference


@pytest.mark.parametrize("model", REGULAR_MODELS)
def test_weather_tools_structured(model):
    class Location(BaseModel):
        latitude: float
        longitude: float
        city: str

    class Weather(BaseModel):
        temperature: float
        condition: str
        humidity: float
        wind_speed: float

    class WeatherReport(BaseModel):
        location: Location
        weather: Weather

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(temperature=0, model=model, timeout=5)
    def structured_weather_assistant(command) -> WeatherReport:
        """{command}"""

    @structured_weather_assistant.tool
    def get_location(city: str) -> Location:
        """Get latitude and longitude based on city name"""
        locations = {
            "New York": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city": "New York",
            },
            "Miami": {"latitude": 25.7617, "longitude": -80.1918, "city": "Miami"},
        }
        return Location(**locations.get(city, {"error": "Location not found"}))

    @structured_weather_assistant.tool
    def get_weather(latitude: float, longitude: float) -> Weather:
        """Get weather based on latitude and longitude"""
        if latitude > 35:
            weather_data = {
                "temperature": 59,
                "condition": "sunny",
                "humidity": 50,
                "wind_speed": 8,
            }
        else:
            weather_data = {
                "temperature": 77,
                "condition": "cloudy",
                "humidity": 80,
                "wind_speed": 6,
            }
        return Weather(**weather_data)

    # Test single city weather with structured output
    result1 = structured_weather_assistant("How's the weather in New York right now?")
    assert isinstance(result1, WeatherReport)
    assert result1.location.city == "New York"
    assert result1.weather.temperature == 59
    assert result1.weather.condition == "sunny"

    # Test weather comparison with structured output
    result2 = structured_weather_assistant("How's the weather in Miami right now?")
    assert isinstance(result2, WeatherReport)
    assert result2.location.city == "Miami"
    assert result2.weather.temperature == 77
    assert result2.weather.condition == "cloudy"


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_cache_control(model):
    """Test cache control functionality"""
    # Skip test for non-Anthropic models
    if not model.startswith(("claude", "anthropic")):
        pytest.skip("Cache control only applies to Anthropic models")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(model=model, cache=True, debug=True, timeout=12)
    def chat(message):
        """Chat: {message}"""

    # Generate a long message that should trigger cache control
    long_message = "Please analyze this text: " + ("lorem ipsum " * 100)

    # First message should have cache control for long content
    result1 = chat(long_message)
    assert isinstance(result1, str)

    # Test with cache disabled
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(model=model, cache=False, debug=True, timeout=12)
    def chat_no_cache(message):
        """Chat: {message}"""

    result2 = chat_no_cache(long_message)
    assert isinstance(result2, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_anthropic_cache_limit(model):
    """Test Anthropic cache block limit"""
    # Skip test for non-Anthropic models
    if not model.startswith(("claude", "anthropic")):
        pytest.skip("Cache control only applies to Anthropic models")

    state = State()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=5),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(model=model, cache=True, state=state, memory=True, debug=True, timeout=12)
    def chat(message):
        """Chat: {message}"""

    # Generate messages that will exceed the cache block limit
    for i in range(6):  # More than anthropic_cached_block_limit
        long_message = f"Analysis part {i}: " + ("lorem ipsum " * 100)
        result = chat(long_message)
        assert isinstance(result, str)

    # Verify the number of cached messages doesn't exceed the limit
    messages = state.get_messages()
    cached_count = sum("cache_control" in msg for msg in messages)
    assert cached_count <= 4  # anthropic_cached_block_limit


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_cache_with_system_prompts(model):
    """Test cache behavior with system prompts"""
    # Skip test for non-Anthropic models
    if not model.startswith(("claude", "anthropic")):
        pytest.skip("Cache control only applies to Anthropic models")

    system_prompts = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "system",
            "content": "You always provide detailed responses",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(model=model, system=system_prompts, cache=True, debug=True, timeout=12)
    def chat(message):
        """Chat: {message}"""

    long_message = "Please analyze: " + ("lorem ipsum " * 100)
    result = chat(long_message)
    assert isinstance(result, str)
