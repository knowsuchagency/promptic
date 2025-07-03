from pydantic import BaseModel
from promptic import llm


class Person(BaseModel):
    name: str
    age: int
    gender: str
    occupation: str


@llm(
    model="gpt-4o",
    system="You generate example data for a person given a description of the person.",
    memory=True,
    temperature=0.7,
    max_tokens=300,
    cache=False,
)
def data_generator(description: str) -> Person:
    """Generate data for the given person description: {description}"""


@data_generator.tool
def persist():
    """Persist the data to a database"""
    print("running persist")


@data_generator.tool
def count_words(text: str) -> int:
    """Count words in the provided text"""
    return len(text.split())


person = data_generator("Imagine a male software engineer and persist the data.")
print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Gender: {person.gender}")
print(f"Occupation: {person.occupation}")

print(data_generator("Now imagine them as a female but don't persist anything."))
