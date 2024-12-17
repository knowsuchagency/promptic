from pydantic import BaseModel
from promptic import llm


class Story(BaseModel):
    title: str
    content: str
    style: str
    word_count: int


@llm(
    model="gpt-4o-mini",
    system="You are a creative writing assistant",
    memory=True,
    temperature=0.7,
    max_tokens=800,
    cache=False,
)
def story_assistant(command: str) -> Story:
    """Process this writing request: {command}"""


@story_assistant.tool
def get_writing_style():
    """Get the current writing style preference"""
    return "whimsical and light-hearted"


@story_assistant.tool
def count_words(text: str) -> int:
    """Count words in the provided text"""
    return len(text.split())


story = story_assistant("Write a short story about a magical library")
print(f"Title: {story.title}")
print(f"Style: {story.style}")
print(f"Words: {story.word_count}")
print(story.content)

print(
    story_assistant("Write another story with the same style but about a time traveler")
)
