import os
from datetime import datetime
from promptic import llm

model = os.getenv("PROMPTIC_MODEL", "claude-3-5-haiku-20241022")

@llm(
    model=model,
    debug=True,
)
def scheduler(command):
    """{command}"""

@scheduler.tool
def get_current_time():
    """Get the current time"""
    print("getting current time")
    return datetime.now().strftime("%I:%M %p")

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

print(
    scheduler(
        "Please calculate 23 * 45, then convert timestamp 1702339200 to readable format"
    )
)

print(scheduler("How's the weather in New York right now?"))

print(scheduler("Please compare the weather between New York and Miami"))

print(scheduler("What's the temperature difference between New York and Miami?"))
