import warnings

warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import logging
from unittest.mock import Mock
import subprocess as sp
from pathlib import Path

import pytest
from litellm.exceptions import RateLimitError, InternalServerError, APIError, Timeout
from pydantic import BaseModel
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
)

from promptic import ImageBytes, Promptic, State, llm, litellm_completion
from openai import OpenAI

ERRORS = (RateLimitError, InternalServerError, APIError, Timeout)

# Define default model lists
CHEAP_MODELS = ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini/gemini-2.0-flash"]
REGULAR_MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini/gemini-1.5-pro"]

openai_completion_fn = OpenAI().chat.completions.create


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_basic(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_parens(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0, model=model, timeout=8, create_completion_fn=create_completion_fn
    )
    def vice_president(year):
        """Who was the Vice President of the United States in {year}?"""

    result = vice_president(2001)
    assert "Dick Cheney" in result
    assert isinstance(result, str)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_pydantic(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    class Capital(BaseModel):
        country: str
        capital: str

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0, model=model, timeout=8, create_completion_fn=create_completion_fn
    )
    def capital(country) -> Capital:
        """What's the capital of {country}?"""

    result = capital("France")
    assert result.country == "France"
    assert result.capital == "Paris"


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_streaming(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        stream=True,
        model=model,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def haiku(subject, adjective, verb="delights"):
        """Write a haiku about {subject} that is {adjective} and {verb}."""

    result = "".join(haiku("programming", adjective="witty"))
    assert isinstance(result, str)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_system_prompt(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system="you are a snarky chatbot",
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_system_prompt_list_strings(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    system_prompts = [
        "you are a helpful assistant",
        "you always provide concise answers",
        "you speak in a formal tone",
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system=system_prompts,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def answer(question):
        """{question}"""

    result = answer("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result
    # Should be concise due to system prompt
    assert len(result.split()) < 30


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_system_prompt_list_dicts(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

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
    @llm(
        system=system_prompts,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def answer(question):
        """{question}"""

    result = answer("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result
    # Should be concise due to system prompt
    assert len(result.split()) < 30


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_agents(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system="you are a posh smart home assistant named Jarvis",
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", REGULAR_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_streaming_with_tools(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

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
        timeout=8,
        create_completion_fn=create_completion_fn,
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_json_schema_validation(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

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
    @llm(
        temperature=0,
        model=model,
        json_schema=schema,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
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
    @llm(
        temperature=0,
        model=model,
        json_schema=invalid_schema,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def get_impossible_score(name: str):
        """Get score for {name}"""

    with pytest.raises(ValueError) as exc_info:
        get_impossible_score("Alice")
    assert "Schema validation failed" in str(exc_info.value)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_dry_run_with_tools(model, create_completion_fn, caplog):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        dry_run=True,
        debug=True,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_debug_logging(model, create_completion_fn, caplog):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        debug=True,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def debug_test(message):
        """Echo: {message}"""

    with caplog.at_level(logging.DEBUG, logger="promptic"):
        debug_test("hello")

    assert any("model =" in record.message for record in caplog.records)
    assert any("hello" in record.message for record in caplog.records)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_multiple_tool_calls(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    counter = Mock()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system="You are a helpful assistant that likes to double-check things",
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_state_basic(model):
    state = State()
    message = {"role": "user", "content": "Hello"}

    state.add_message(message)
    assert state.get_messages() == [message]

    state.clear()
    assert state.get_messages() == []


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_state_limit(model):
    state = State()
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(3)]

    for msg in messages:
        state.add_message(msg)

    assert state.get_messages(limit=2) == messages[-2:]
    assert state.get_messages() == messages


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_memory_conversation(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        memory=True,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    # First message should be stored
    result1 = chat("What is the capital of France?")
    assert "Paris" in result1

    # Second message should reference the context from first
    result2 = chat("What did I just ask about?")
    assert "france" in result2.lower() or "paris" in result2.lower()


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_custom_state(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

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
    @llm(
        state=custom_state,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    result = chat("Hello")
    assert len(custom_state.get_messages()) > 0

    custom_state.clear()
    assert custom_state.cleared
    assert len(custom_state.get_messages()) == 0


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_memory_disabled(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        memory=False,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    result1 = chat("What is the capital of France?")
    result2 = chat("What did I just ask about?")

    # Without memory, the second response shouldn't mention France
    assert not ("france" in result2.lower() or "paris" in result2.lower())


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_memory_with_streaming(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    state = State()
    p = Promptic(
        model=model,
        memory=True,
        state=state,
        stream=True,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
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
    assert len(state.get_messages()) == 2, f"Messages: {state.get_messages()}"
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_pydantic_with_tools(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    class WeatherReport(BaseModel):
        location: str
        temperature: float
        conditions: str

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", REGULAR_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_pydantic_tools_with_memory(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    class TaskStatus(BaseModel):
        task_id: int
        status: str
        last_update: str

    state = State()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        memory=True,
        state=state,
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
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
        timeout=8,
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
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_gemini_streaming_with_tools_error(create_completion_fn):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        stream=True,
        model="gemini/gemini-1.5-pro",
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def assistant(command):
        """{command}"""

    @assistant.tool
    def get_time():
        """Get the current time"""
        return "12:00 PM"

    with pytest.raises(ValueError) as exc_info:
        next(assistant("What time is it?"))

    assert str(exc_info.value) == "Gemini models do not support streaming with tools"


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_mutually_exclusive_schemas(model, create_completion_fn):
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

        @llm(
            temperature=0,
            model=model,
            json_schema=schema,
            timeout=8,
            create_completion_fn=create_completion_fn,
        )
        def get_person(name: str) -> Person:
            """Get information about {name}"""

    assert (
        str(exc_info.value)
        == "Cannot use both Pydantic return type hints and json_schema validation together"
    )


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_wrapper_attributes(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    custom_state = State()
    p = Promptic(
        model=model,
        temperature=0.7,
        memory=True,
        system="test system prompt",
        stream=True,
        debug=True,
        state=custom_state,
        timeout=8,
        create_completion_fn=create_completion_fn,
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

    assert test_function.completion_kwargs == {
        "temperature": 0.7,
        "stream": True,
        "timeout": 8,
    }


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_clear_state(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    # Test successful clearing
    state = State()

    @llm(
        model=model,
        memory=True,
        state=state,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    # Add some messages
    chat("Hello")
    assert len(state.get_messages()) > 0

    # Clear the state
    chat.clear()
    assert len(state.get_messages()) == 0

    # Test error when memory/state is disabled

    @llm(
        model=model,
        memory=False,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat_no_memory(message):
        """Chat: {message}"""

    with pytest.raises(ValueError) as exc_info:
        chat_no_memory.clear()
    assert "Cannot clear state: memory/state is not enabled" in str(exc_info.value)


def _get_example_files():
    """Get all example Python files."""
    return map(str, Path("examples").glob("*.py"))


@pytest.mark.examples
@pytest.mark.parametrize("example_file", _get_example_files())
def test_examples(example_file, max_attempts=2):
    """Run each example file.

    Args:
        example_file: Path to the example file to run
        max_attempts: Maximum number of attempts to run the example (default: 2)
    """
    if example_file == "examples/state.py":
        pytest.skip("State example is not runnable without Redis.")

    # Determine the command to run
    if example_file == "examples/memory.py":
        cmd = f"uv run --with gradio {example_file}"
    elif example_file == "examples/weave_integration.py":
        cmd = f"uv run --with weave {example_file}"
    else:
        cmd = f"uv run {example_file}"

    # Try running the command, retry if it fails
    for attempt in range(max_attempts):
        try:
            sp.run(cmd, shell=True, check=True)
            break  # Success, exit the retry loop
        except sp.CalledProcessError as e:
            if attempt == max_attempts - 1:  # Last attempt failed
                raise e  # Re-raise the exception
            # Previous attempt failed, will retry


@pytest.mark.vcr
@pytest.mark.parametrize("model", REGULAR_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_weather_tools_basic(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=8,
        debug=True,
        create_completion_fn=create_completion_fn,
    )
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", REGULAR_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_weather_tools_structured(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

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
    @llm(
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_cache_control(model, create_completion_fn):
    """Test cache control functionality"""
    # Skip test for non-Anthropic models
    if not model.startswith(("claude", "anthropic")):
        pytest.skip("Cache control only applies to Anthropic models")

    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        model=model,
        cache=True,
        debug=True,
        timeout=12,
        create_completion_fn=create_completion_fn,
    )
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
    @llm(
        model=model,
        cache=False,
        debug=True,
        timeout=12,
        create_completion_fn=create_completion_fn,
    )
    def chat_no_cache(message):
        """Chat: {message}"""

    result2 = chat_no_cache(long_message)
    assert isinstance(result2, str)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_anthropic_cache_limit(model, create_completion_fn):
    """Test Anthropic cache block limit"""
    # Skip test for non-Anthropic models
    if not model.startswith(("claude", "anthropic")):
        pytest.skip("Cache control only applies to Anthropic models")

    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    state = State()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=5),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        model=model,
        cache=True,
        state=state,
        memory=True,
        # debug=True,
        timeout=12,
        create_completion_fn=create_completion_fn,
    )
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


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_cache_with_system_prompts(model, create_completion_fn):
    """Test cache behavior with system prompts"""
    # Skip test for non-Anthropic models
    if not model.startswith(("claude", "anthropic")):
        pytest.skip("Cache control only applies to Anthropic models")

    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

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
    @llm(
        model=model,
        system=system_prompts,
        cache=True,
        debug=True,
        timeout=12,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    long_message = "Please analyze: " + ("lorem ipsum " * 100)
    result = chat(long_message)
    assert isinstance(result, str)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_system_prompt_order(model, create_completion_fn):
    """Test that system prompts are always first in the message list"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    state = State()
    system_prompt = "You are a helpful test assistant"

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        model=model,
        system=system_prompt,
        state=state,
        memory=True,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    # First interaction
    chat("Hello")
    messages = state.get_messages()
    # Verify system message is first
    assert messages[0]["role"] == "system", messages
    assert messages[0]["content"] == system_prompt

    # Second interaction should still have system message first
    chat("How are you?")
    messages = state.get_messages()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt

    # Test with list of system prompts
    state.clear()
    system_prompts = [
        "You are a helpful assistant",
        "You always provide concise answers",
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        model=model,
        system=system_prompts,
        state=state,
        memory=True,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def chat2(message):
        """Chat: {message}"""

    chat2("Hello")
    messages = state.get_messages()

    # Verify both system messages are first, in order
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompts[0]
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == system_prompts[1]


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_message_order_with_memory(model, create_completion_fn):
    """Test that messages maintain correct order with memory enabled"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    state = State()
    system_prompts = [
        "You are a helpful assistant",
        "You always provide concise answers",
        "You speak in a formal tone",
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        model=model,
        system=system_prompts,
        state=state,
        memory=True,
        temperature=0,
        timeout=8,
        debug=True,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """Chat: {message}"""

    # First interaction
    chat("Hello")
    messages = state.get_messages()

    # Check initial message order
    assert len(messages) == 5  # 3 system + 1 user + 1 assistant
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompts[0]
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == system_prompts[1]
    assert messages[2]["role"] == "system"
    assert messages[2]["content"] == system_prompts[2]
    assert messages[3]["role"] == "user"
    assert isinstance(messages[3]["content"], list)
    assert len(messages[3]["content"]) == 1
    assert messages[3]["content"][0]["type"] == "text"
    assert "Hello" in messages[3]["content"][0]["text"]
    assert messages[4]["role"] == "assistant"

    # Second interaction
    chat("How are you?")
    messages = state.get_messages()

    # Check message order after second interaction
    assert len(messages) == 7  # 3 system + 2 user + 2 assistant
    # System messages should still be first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompts[0]
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == system_prompts[1]
    assert messages[2]["role"] == "system"
    assert messages[2]["content"] == system_prompts[2]
    # First interaction messages
    assert messages[3]["role"] == "user"
    assert isinstance(messages[3]["content"], list)
    assert len(messages[3]["content"]) == 1
    assert messages[3]["content"][0]["type"] == "text"
    assert "Hello" in messages[3]["content"][0]["text"]
    assert messages[4]["role"] == "assistant"
    # Second interaction messages
    assert messages[5]["role"] == "user"
    assert isinstance(messages[5]["content"], list)
    assert len(messages[5]["content"]) == 1
    assert messages[5]["content"][0]["type"] == "text"
    assert "How are you?" in messages[5]["content"][0]["text"]
    assert messages[6]["role"] == "assistant"


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_message_method(model, create_completion_fn):
    """Test the direct message method of Promptic"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    # Test basic message functionality
    p = Promptic(
        model=model,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    result = p.message("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result

    # Test with memory enabled
    p_with_memory = Promptic(
        model=model,
        memory=True,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )

    # First message
    result1 = p_with_memory.message("What is the capital of France?")
    assert "Paris" in result1

    # Second message should have context from first
    result2 = p_with_memory.message("What did I just ask about?")
    assert any([word in result2.lower() for word in ["france", "paris"]]), (
        p_with_memory.state.get_messages()
    )

    # Verify messages are stored in state
    assert (
        len(p_with_memory.state.get_messages()) == 4
    )  # 2 user messages + 2 assistant responses
    messages = p_with_memory.state.get_messages()
    assert messages[0]["role"] == "user"
    assert "France" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[3]["role"] == "assistant"


@pytest.mark.vcr
@pytest.mark.parametrize("model", REGULAR_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_image_functionality(model, create_completion_fn):
    """Test image support functionality with different formats and prompting"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    # Test structured output with ImageDescription
    class ImageDescription(BaseModel):
        content: str
        colors: list[str]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=10,
        create_completion_fn=create_completion_fn,
    )
    def analyze_image(image: ImageBytes) -> ImageDescription:
        """Describe this image and list its main colors"""

    # Test with both JPEG and PNG formats
    for ext in ["jpeg", "png"]:
        with open(f"tests/fixtures/ocai-logo.{ext}", "rb") as f:
            image_data = ImageBytes(f.read())

        try:
            result = analyze_image(image_data)
            assert isinstance(result, ImageDescription)
            assert len(result.content) > 0
            assert len(result.colors) > 0
            assert any("orange" in color.lower() for color in result.colors)
        except Exception as e:
            # Known issue: Gemini 1.5 Pro sometimes returns schema instead of data with images
            if model == "gemini/gemini-1.5-pro" and "Field required" in str(e):
                pytest.skip(
                    f"Known issue: Gemini 1.5 Pro returns schema structure instead of data with images"
                )
            else:
                raise

    # Test free-form prompting
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=10,
        create_completion_fn=create_completion_fn,
    )
    def analyze_image_feature(img: ImageBytes, feature: str):
        """Tell me about the {feature} in this image"""

    with open("tests/fixtures/ocai-logo.jpeg", "rb") as f:
        image_data = ImageBytes(f.read())

    # Test specific feature analysis
    color_result = analyze_image_feature(image_data, "colors")
    assert isinstance(color_result, str)
    assert any(color in color_result.lower() for color in ["orange", "white", "black"])

    text_result = analyze_image_feature(image_data, "text or letters")
    assert isinstance(text_result, str)
    assert "ai" in text_result.lower() or "oc" in text_result.lower()


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_completion_method(model, create_completion_fn):
    """Test the direct completion method of Promptic"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    p = Promptic(
        model=model,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )

    # Test basic completion with a single message
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = p.completion(messages)
    assert "Paris" in response.choices[0].message.content

    # Test completion with multiple messages
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is its population?"},
    ]
    response = p.completion(messages)
    assert any(
        word in response.choices[0].message.content.lower()
        for word in ["million", "inhabitants", "people"]
    )

    # Test completion with system message
    p_with_system = Promptic(
        model=model,
        temperature=0,
        timeout=8,
        system="You are a geography expert",
        create_completion_fn=create_completion_fn,
    )
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = p_with_system.completion(messages)
    assert "Paris" in response.choices[0].message.content

    # Test warning when using with state enabled
    p_with_state = Promptic(
        model=model,
        temperature=0,
        timeout=8,
        memory=True,
        create_completion_fn=create_completion_fn,
    )
    with pytest.warns(
        UserWarning, match="State is enabled, but completion is being called directly"
    ):
        response = p_with_state.completion(messages)
        assert "Paris" in response.choices[0].message.content


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_tool_isolation_with_llm_method(model, create_completion_fn):
    """Test that tools don't leak between parent and child functions when using llm method"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    p = Promptic(
        model=model,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @p.llm
    def parent_function(command):
        """Respond to: {command}"""

    # Add a tool to the parent function
    @parent_function.tool
    def get_time():
        """Get the current time"""
        return "12:00 PM"

    # Create a child function using the parent's llm method
    @p.llm
    def child_function(command):
        """Respond to: {command}"""

    # Verify parent function has access to the tool
    result = parent_function("What time is it?")
    assert "12:00" in result

    # Verify child function does not have access to the tool
    # It should respond without using the tool
    result = child_function("What time is it?")
    assert "12:00" not in result


@pytest.mark.vcr
@pytest.mark.parametrize("model", REGULAR_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_tool_definition_with_pydantic_param(model, create_completion_fn):
    """Test that tool definitions properly handle Pydantic BaseModel parameters"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")
    if "gemini" in model:
        pytest.skip("The Gemini model is flaky with this test.")

    class UserProfile(BaseModel):
        name: str
        age: int
        email: str

    llm = Promptic(
        model=model,
        temperature=0,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )

    @llm
    def assistant(command):
        """{command}"""

    name = None

    @assistant.tool
    def update_user(profile: UserProfile) -> str:
        """Update user profile"""
        nonlocal name
        name = profile.name
        return f"Updated profile for {profile.name}"

    assistant(
        "Update the user's profile with the name 'John Doe', age 30, and email 'john@example.com'"
    )

    assert name == "John Doe"


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_docstring_validation(model, create_completion_fn):
    """Test validation of docstrings in decorated functions"""
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    # Test function with no docstring
    with pytest.raises(TypeError) as exc_info:

        @llm(model=model, create_completion_fn=create_completion_fn)
        def no_docstring_function(text):
            pass

        no_docstring_function("Hello")

    assert "no_docstring_function has no docstring" in str(exc_info.value)
    assert "Ensure the docstring is not an f-string" in str(exc_info.value)

    # Test function with an f-string (this won't actually be executed but will fail at definition time)
    name = "World"
    with pytest.raises(TypeError) as exc_info:
        # This would fail because f-strings aren't allowed in docstrings
        @llm(model=model, create_completion_fn=create_completion_fn)
        def f_string_docstring(text):
            f"""Hello {name}"""
            return text

        f_string_docstring("Hello")

    assert "f_string_docstring has no docstring" in str(exc_info.value)
    assert "Ensure the docstring is not an f-string" in str(exc_info.value)


@pytest.mark.vcr
@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_schema_simplification(model):
    """Test that schema simplification removes metadata from Pydantic models"""
    import json
    from promptic import simplify_schema

    class TestModel(BaseModel):
        """Test model with metadata"""

        name: str
        age: int

    # Test the simplify_schema function directly
    original_schema = TestModel.model_json_schema()
    simplified = simplify_schema(original_schema)

    # Check that metadata is removed
    assert "title" not in simplified
    assert "description" not in simplified
    assert "title" not in simplified["properties"]["name"]
    assert "title" not in simplified["properties"]["age"]

    # Check that essential fields are preserved
    assert simplified["type"] == "object"
    assert "name" in simplified["properties"]
    assert "age" in simplified["properties"]
    assert simplified["properties"]["name"]["type"] == "string"
    assert simplified["properties"]["age"]["type"] == "integer"
    assert simplified["required"] == ["name", "age"]

    # Test with llm decorator - simplified schema (default)
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(model=model, temperature=0, timeout=8)
    def get_person() -> TestModel:
        """Create a person named Alice who is 30 years old."""

    result = get_person()
    assert isinstance(result, TestModel)
    assert result.name == "Alice"
    assert result.age == 30

    # Test with llm decorator - full schema
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(model=model, temperature=0, timeout=8, simplify_schema=False)
    def get_person_full_schema() -> TestModel:
        """Create a person named Bob who is 25 years old."""

    result = get_person_full_schema()
    assert isinstance(result, TestModel)
    assert result.name == "Bob"
    assert result.age == 25
