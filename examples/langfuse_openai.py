from langfuse.openai import openai
from langfuse.decorators import observe
from promptic import Promptic


promptic = Promptic(openai_client=openai.OpenAI())


@observe
@promptic.llm
def greet(name):
    """Greet {name}"""


print(greet("John"))
# Hello, John!
