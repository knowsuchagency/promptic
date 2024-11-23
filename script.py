from promptic import llm, State

# Simple conversational chat bot using default in-memory state
@llm(memory=True)
def chat(message):
    """Chat: {message}"""

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chat(user_input)
    print(f"Bot: {response}")