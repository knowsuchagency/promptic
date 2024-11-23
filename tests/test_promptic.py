import logging
from unittest.mock import Mock
from typing import Literal

import pytest

from promptic import llm, promptic, State, Promptic
from pydantic import BaseModel

CHEAP_MODELS = ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini/gemini-1.5-flash"]
REGULAR_MODELS = ["gpt-4o", "claude-3.5", "gemini/gemini-1.5-pro"]


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_basic(model):
    @llm(temperature=0, model=model)
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_parens(model):
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

    @llm(temperature=0, model=model)
    def capital(country) -> Capital:
        """What's the capital of {country}?"""

    result = capital("France")
    assert result.country == "France"
    assert result.capital == "Paris"


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_streaming(model):
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
    @llm(system="you are a snarky chatbot", temperature=0, model=model)
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_agents(model):
    # Skip test for Anthropic models
    if "claude" in model:
        pytest.skip("Anthropic models only support one tool")

    @llm(
        system="you are a posh smart home assistant named Jarvis",
        temperature=0,
        model=model,
    )
    def jarvis(command):
        """{command}"""

    @jarvis.tool
    def turn_light_on():
        """turn light on"""
        print("turning light on")
        return True

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
    # Skip test for Anthropic models
    if "claude" in model:
        pytest.skip("Anthropic models only support one tool")
    # Skip test for Gemini models
    if model.startswith(("gemini", "vertex")):
        pytest.skip("Gemini models do not support streaming with tools")

    time_mock = Mock(return_value="12:00 PM")
    weather_mock = Mock(return_value="Sunny in Paris")

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
def test_json_schema_return(model):
    @llm(temperature=0, model=model)
    def get_user_info(
        name: str,
    ) -> {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
        "required": ["name", "age"],
    }:
        """Get information about {name}"""

    result = get_user_info("Alice")
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_dry_run_with_tools(model, caplog):
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
    @llm(debug=True, temperature=0, model=model)
    def debug_test(message):
        """Echo: {message}"""

    with caplog.at_level(logging.DEBUG, logger="promptic"):
        debug_test("hello")

    assert any("model =" in record.message for record in caplog.records)
    assert any("hello" in record.message for record in caplog.records)


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_multiple_tool_calls(model):
    # Skip test for Anthropic models
    if "claude" in model:
        pytest.skip("Anthropic models only support one tool")

    counter = Mock()

    @llm(
        system="You are a helpful assistant that likes to double-check things",
        temperature=0,
        model=model,
    )
    def double_checker(query):
        """{query}"""

    @double_checker.tool
    def check_status():
        """Check the current status"""
        counter()
        return "Status OK"

    result = double_checker("Please check the status twice to be sure")
    assert counter.call_count == 2


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
    @llm(memory=False, temperature=0, model=model)
    def chat(message):
        """Chat: {message}"""

    result1 = chat("What is the capital of France?")
    result2 = chat("What did I just ask about?")

    # Without memory, the second response shouldn't mention France
    assert not ("france" in result2.lower() or "paris" in result2.lower())


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_memory_with_streaming(model):
    # Initialize state and promptic instance
    state = State()
    p = Promptic(
        model=model,
        memory=True,
        state=state,
        stream=True,
        temperature=0,
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
    # Skip test for Anthropic models
    if "claude" in model:
        pytest.skip("Anthropic models only support one tool")

    class WeatherReport(BaseModel):
        location: str
        temperature: float
        conditions: str

    @llm(temperature=0, model=model)
    def get_weather_report(location: str) -> WeatherReport:
        """Create a detailed weather report for {location}"""

    @get_weather_report.tool
    def get_temperature(city: str) -> float:
        """Get the temperature for a city"""
        return 72.5

    @get_weather_report.tool
    def get_conditions(city: str) -> str:
        """Get the weather conditions for a city"""
        return "Sunny with light clouds"

    result = get_weather_report("San Francisco")
    assert isinstance(result, WeatherReport)
    assert result.location == "San Francisco"
    assert isinstance(result.temperature, float)
    assert isinstance(result.conditions, str)


@pytest.mark.parametrize("model", REGULAR_MODELS)
def test_pydantic_tools_with_memory(model):
    # Skip test for Anthropic models
    if "claude" in model:
        pytest.skip("Anthropic models only support one tool")

    class TaskStatus(BaseModel):
        task_id: int
        status: str
        last_update: str

    state = State()

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
