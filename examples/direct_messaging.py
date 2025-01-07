from promptic import llm


@llm(
    system="You are a knowledgeable history teacher.",
    model="gpt-4o-mini",
    memory=True,
    stream=True,
)
def history_chat(era: str, region: str):
    """Tell me a fun fact about the history of {region} during the {era} period."""


response = history_chat("medieval", "Japan")
for chunk in response:
    print(chunk, end="")

for chunk in history_chat.message(
    "In one sentence, was the most popular person there at the time?"
):
    print(chunk, end="")

for chunk in history_chat.message("In one sentence, who was their main rival?"):
    print(chunk, end="")
