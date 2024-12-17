import gradio as gr
from promptic import llm


@llm(memory=True, stream=True)
def assistant(message):
    """{message}"""


def predict(message, history):
    partial_message = ""
    for chunk in assistant(message):
        partial_message += str(chunk)
        yield partial_message


with gr.ChatInterface(title="Promptic Chatbot Demo", fn=predict) as demo:
    # ensure clearing the chat window clears the chat history
    demo.chatbot.clear(assistant.clear)

# demo.launch()
