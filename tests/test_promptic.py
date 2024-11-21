from promptic import llm, promptic
from pydantic import BaseModel
from unittest.mock import Mock


def test_basic():
    @promptic
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


def test_parens():
    @promptic()
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
    )
    def haiku(subject, adjective, verb="delights"):
        """Write a haiku about {subject} that is {adjective} and {verb}."""

    result = "".join(haiku("programming", adjective="witty"))
    assert isinstance(result, str)


def test_system_prompt():
    @llm(system="you are a snarky chatbot")
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_agents():
    @llm(system="you are a posh smart home assistant named Jarvis")
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
    assert ("weather" in result.lower() or "temperature" in result.lower())


def test_streaming_with_tools():
    time_mock = Mock(return_value="12:00 PM")
    weather_mock = Mock(return_value="Sunny in Paris")

    @llm(stream=True, model="gpt-4o", system="you are a helpful assistant")
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
