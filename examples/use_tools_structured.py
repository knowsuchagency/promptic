from promptic import llm
from pydantic import BaseModel

@llm(
    model="gpt-4o-mini",
)
def scheduler(command):
    """{command}"""

@scheduler.tool
def calculator(expression: str) -> float:
    """Calculate the result of a mathematical expression"""
    return eval(expression)

@scheduler.tool
def get_location(city: str) -> dict:
    """Get latitude and longitude based on city name"""
    # Simulated data
    locations = {
        "New York": {"latitude": 40.7128, "longitude": -74.0060, "city": "New York"},
        "Chicago": {"latitude": 41.8781, "longitude": -87.6298, "city": "Chicago"},
        "Miami": {"latitude": 25.7617, "longitude": -80.1918, "city": "Miami"},
    }
    return locations.get(city, {"error": "Location not found"})

@scheduler.tool
def get_weather(latitude: float, longitude: float) -> dict:
    """
    Weather service
    """
    # Simulated weather data for different regions
    if latitude > 35:  # Northern region
        return {
            "temperature": 59,  # Fahrenheit
            "condition": "sunny",
            "humidity": 50,
            "wind_speed": 8,  # mph
        }
    else:  # Southern region
        return {
            "temperature": 77,  # Fahrenheit
            "condition": "cloudy",
            "humidity": 80,
            "wind_speed": 6,  # mph
        }

class Location(BaseModel):
    latitude: float
    longitude: float
    city: str

class Weather(BaseModel):
    temperature: float
    condition: str
    humidity: float
    wind_speed: float

@llm(
    model="gpt-4o-mini",
)
def structured_scheduler(command):
    """{command}"""

@structured_scheduler.tool
def get_location_structured(city: str) -> Location:
    """Get latitude and longitude based on city name"""
    return Location(**get_location(city))

@structured_scheduler.tool
def get_weather_structured(latitude: float, longitude: float) -> Weather:
    """Get weather based on latitude and longitude"""
    return Weather(**get_weather(latitude, longitude))

print(structured_scheduler("How's the weather in New York right now?"))

print(structured_scheduler("Please compare the weather between New York and Miami"))

print(
    structured_scheduler(
        "What's the temperature difference between New York and Miami?"
    )
)
