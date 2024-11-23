from promptic import llm, promptic, State
from pydantic import BaseModel
from unittest.mock import Mock
import logging


def test_basic():
    @promptic(temperature=0)
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


def test_parens():
    @promptic(model="gpt-3.5-turbo", temperature=0)
    def vice_president(year):
        """Who was the Vice President of the United States in {year}?"""

    result = vice_president(2001)
    assert "Dick Cheney" in result
    assert isinstance(result, str)


def test_pydantic():
    class Capital(BaseModel):
        country: str
        capital: str

    @llm
    def capital(country) -> Capital:
        """What's the capital of {country}?"""

    result = capital("France")
    assert result.country == "France"
    assert result.capital == "Paris"


def test_streaming():
    @llm(
        stream=True,
        model="claude-3-haiku-20240307",
        temperature=0,
    )
    def haiku(subject, adjective, verb="delights"):
        """Write a haiku about {subject} that is {adjective} and {verb}."""

    result = "".join(haiku("programming", adjective="witty"))
    assert isinstance(result, str)


def test_system_prompt():
    @llm(system="you are a snarky chatbot", temperature=0)
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_agents():
    @llm(system="you are a posh smart home assistant named Jarvis", temperature=0)
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


def test_streaming_with_tools():
    time_mock = Mock(return_value="12:00 PM")
    weather_mock = Mock(return_value="Sunny in Paris")

    @llm(
        stream=True, model="gpt-4o", system="you are a helpful assistant", temperature=0
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


def test_json_schema_return():
    @llm(temperature=0)
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


def test_dry_run_with_tools(caplog):
    @llm(dry_run=True, debug=True, temperature=0)
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


def test_debug_logging(caplog):
    @llm(debug=True, temperature=0)
    def debug_test(message):
        """Echo: {message}"""

    with caplog.at_level(logging.DEBUG, logger="promptic"):
        debug_test("hello")

    assert any("model =" in record.message for record in caplog.records)
    assert any("hello" in record.message for record in caplog.records)


def test_multiple_tool_calls():
    counter = Mock()

    @llm(
        system="You are a helpful assistant that likes to double-check things",
        temperature=0,
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


def test_state_basic():
    state = State()
    message = {"role": "user", "content": "Hello"}

    state.add_message(message)
    assert state.get_messages() == [message]

    state.clear()
    assert state.get_messages() == []


def test_state_limit():
    state = State()
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(3)]

    for msg in messages:
        state.add_message(msg)

    assert state.get_messages(limit=2) == messages[-2:]
    assert state.get_messages() == messages


def test_memory_conversation():
    @llm(memory=True, temperature=0)
    def chat(message):
        """Chat: {message}"""

    # First message should be stored
    result1 = chat("What is the capital of France?")
    assert "Paris" in result1

    # Second message should reference the context from first
    result2 = chat("What did I just ask about?")
    assert "france" in result2.lower() or "paris" in result2.lower()


def test_custom_state():
    class TestState(State):
        def __init__(self):
            super().__init__()
            self.cleared = False

        def clear(self):
            self.cleared = True
            super().clear()

    custom_state = TestState()

    @llm(state=custom_state, temperature=0)
    def chat(message):
        """Chat: {message}"""

    result = chat("Hello")
    assert len(custom_state.get_messages()) > 0

    custom_state.clear()
    assert custom_state.cleared
    assert len(custom_state.get_messages()) == 0


def test_memory_disabled():
    @llm(memory=False, temperature=0)
    def chat(message):
        """Chat: {message}"""

    result1 = chat("What is the capital of France?")
    result2 = chat("What did I just ask about?")

    # Without memory, the second response shouldn't mention France
    assert not ("france" in result2.lower() or "paris" in result2.lower())
