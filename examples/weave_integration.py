from promptic import Promptic
import weave

# Initialize the weave client with the project name
# it doesn't need to be "promptic"
client = weave.init("promptic")

promptic = Promptic(weave_client=client)


@promptic.llm
def translate(text, language="Chinese"):
    """Translate '{text}' to {language}"""


print(translate("Hello world!"))
# 您好，世界！

print(translate("Hello world!", language="Spanish"))
# ¡Hola, mundo!


@promptic.llm(stream=True)
def write_poem(topic):
    """Write a haiku about {topic}."""


print("".join(write_poem("artificial intelligence")))
# Binary thoughts hum,
# Electron minds awake, learn,
# Future thinking now.
