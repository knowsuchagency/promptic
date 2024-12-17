from promptic import llm

# imagine these are long legal documents
legal_document, another_legal_document = (
    "a legal document about Sam",
    "a legal document about Jane",
)

system_prompts = [
    "You are a helpful legal assistant",
    "You provide detailed responses based on the provided context",
    f"legal document 1: '{legal_document}'",
    f"legal document 2: '{another_legal_document}'",
]


@llm(
    system=system_prompts,
    cache=True,  # this is the default
)
def legal_chat(message):
    """{message}"""


print(legal_chat("which legal document is about Sam?"))
# The legal document about Sam is "legal document 1."
