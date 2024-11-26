import logging
from unittest.mock import Mock
import os

import pytest
from litellm.exceptions import RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
)

from promptic import Promptic, State, llm, promptic

# Define default model lists
CHEAP_MODELS = ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini/gemini-1.5-flash"]
REGULAR_MODELS = ["gpt-4o", "claude-3.5", "gemini/gemini-1.5-pro"]


# Override with single model if running in GitHub Actions
if os.environ.get("GITHUB_ACTIONS") == "true":
    required_vars = ["TEST_MODEL", "TEST_MODEL_TYPE"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise RuntimeError(
            f"Running in GitHub Actions but missing required environment variables: {missing_vars}"
        )

    if os.environ["TEST_MODEL_TYPE"] == "cheap":
        CHEAP_MODELS = [os.environ["TEST_MODEL"]]
        REGULAR_MODELS = []
    elif os.environ["TEST_MODEL_TYPE"] == "regular":
        REGULAR_MODELS = [os.environ["TEST_MODEL"]]
        CHEAP_MODELS = []


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_basic(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(temperature=0, model=model)
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_parens(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @promptic(temperature=0, model=model)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(temperature=0, model=model)
    def capital(country) -> Capital:
        """What's the capital of {country}?"""

    result = capital("France")
    assert result.country == "France"
    assert result.capital == "Paris"


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_streaming(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(
        stream=True,
        model=model,
        temperature=0,
    )
    def haiku(subject, adjective, verb="delights"):
        """Write a haiku about {subject} that is {adjective} and {verb}."""

    result = "".join(haiku("programming", adjective="witty"))
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_system_prompt(model):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(system="you are a snarky chatbot", temperature=0, model=model)
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_agents(model):
    if "claude" in model:  # pragma: no cover
        pytest.skip("Anthropic models only support one tool")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(
        system="you are a posh smart home assistant named Jarvis",
        temperature=0,
        model=model,
    )
    def jarvis(command):
        """{command}"""

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @jarvis.tool
    def turn_light_on():
        """turn light on"""
        print("turning light on")
        return True

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
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


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_streaming_with_tools(model):
    if "claude" in model:  # pragma: no cover
        pytest.skip("Anthropic models only support one tool")
    if model.startswith(("gemini", "vertex")):  # pragma: no cover
        pytest.skip("Gemini models do not support streaming with tools")

    time_mock = Mock(return_value="12:00 PM")
    weather_mock = Mock(return_value="Sunny in Paris")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(stream=True, model=model, system="you are a helpful assistant", temperature=0)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(temperature=0, model=model, json_schema=schema)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(temperature=0, model=model, json_schema=invalid_schema)
    def get_impossible_score(name: str):
        """Get score for {name}"""

    with pytest.raises(ValueError) as exc_info:
        get_impossible_score("Alice")
    assert "Schema validation failed" in str(exc_info.value)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_dry_run_with_tools(model, caplog):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(dry_run=True, debug=True, temperature=0, model=model)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(debug=True, temperature=0, model=model)
    def debug_test(message):
        """Echo: {message}"""

    with caplog.at_level(logging.DEBUG, logger="promptic"):
        debug_test("hello")

    assert any("model =" in record.message for record in caplog.records)
    assert any("hello" in record.message for record in caplog.records)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_multiple_tool_calls(model):
    if "claude" in model:
        pytest.skip("Anthropic models only support one tool")

    counter = Mock()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(
        system="You are a helpful assistant that likes to double-check things",
        temperature=0,
        model=model,
    )
    def double_checker(query):
        """{query}"""

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(memory=True, temperature=0, model=model)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(state=custom_state, temperature=0, model=model)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(memory=False, temperature=0, model=model)
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
        temperature=0,
    )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
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
    if "claude" in model:  # pragma: no cover
        pytest.skip("Anthropic models only support one tool")

    class WeatherReport(BaseModel):
        location: str
        temperature: float
        conditions: str

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(temperature=0, model=model)
    def get_weather_report(location: str) -> WeatherReport:
        """Create a detailed weather report for {location}"""

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @get_weather_report.tool
    def get_temperature(city: str) -> float:
        """Get the temperature for a city"""
        return 72.5

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
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
    if "claude" in model:  # pragma: no cover
        pytest.skip("Anthropic models only support one tool")

    class TaskStatus(BaseModel):
        task_id: int
        status: str
        last_update: str

    state = State()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(memory=True, state=state, temperature=0, model=model)
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
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(
        model="claude-3-haiku-20240307",
        temperature=0,
        debug=True,
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


def test_anthropic_multiple_tools_error():
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(
        model="claude-3-haiku-20240307",
        temperature=0,
    )
    def assistant(command):
        """{command}"""

    @assistant.tool
    def get_time():
        """Get the current time"""
        return "12:00 PM"

    with pytest.raises(ValueError) as exc_info:

        @assistant.tool
        def get_weather(location: str):
            """Get the weather for a location"""
            return f"Sunny in {location}"

    assert str(exc_info.value) == "Anthropic models currently support only one tool."


# Add new test to verify Gemini streaming with tools raises exception
def test_gemini_streaming_with_tools_error():
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @llm(stream=True, model="gemini/gemini-1.5-pro")
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

        @llm(temperature=0, model=model, json_schema=schema)
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

    assert test_function.litellm_kwargs == {"temperature": 0.7, "stream": True}


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_clear_state(model):
    # Test successful clearing
    state = State()

    @llm(model=model, memory=True, state=state)
    def chat(message):
        """Chat: {message}"""

    # Add some messages
    chat("Hello")
    assert len(state.get_messages()) > 0

    # Clear the state
    chat.clear()
    assert len(state.get_messages()) == 0

    # Test error when memory/state is disabled

    @llm(model=model, memory=False)
    def chat_no_memory(message):
        """Chat: {message}"""

    with pytest.raises(ValueError) as exc_info:
        chat_no_memory.clear()
    assert "Cannot clear state: memory/state is not enabled" in str(exc_info.value)
