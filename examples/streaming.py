from promptic import llm


@llm(stream=True)
def write_poem(topic):
    """Write a haiku about {topic}."""


print("".join(write_poem("artificial intelligence")))
# Binary thoughts hum,
# Electron minds awake, learn,
# Future thinking now.
