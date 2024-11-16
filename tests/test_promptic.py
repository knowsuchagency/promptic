import pytest
from promptic import llm, promptic
from pydantic import BaseModel

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
    
    result = jarvis("Please turn the light on Jarvis. By the way, what is the weather in San Francisco?")
    assert isinstance(result, str)
    assert "weather" in result.lower() 
