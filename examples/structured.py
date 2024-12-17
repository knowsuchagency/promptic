from pydantic import BaseModel
from promptic import llm


class Forecast(BaseModel):
    location: str
    temperature: float
    units: str


@llm
def get_weather(location, units: str = "fahrenheit") -> Forecast:
    """What's the weather for {location} in {units}?"""


print(get_weather("San Francisco", units="celsius"))
# location='San Francisco' temperature=16.0 units='Celsius'
