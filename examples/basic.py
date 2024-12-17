from promptic import llm


@llm
def translate(text, language="Chinese"):
    """Translate '{text}' to {language}"""


print(translate("Hello world!"))
# 您好，世界！

print(translate("Hello world!", language="Spanish"))
# ¡Hola, mundo!


@llm(
    model="claude-3-haiku-20240307",
    system="You are a customer service analyst. Provide clear sentiment analysis with key points.",
)
def analyze_sentiment(text):
    """Analyze the sentiment of this customer feedback: {text}"""


print(analyze_sentiment("The product was okay but shipping took forever"))
# Sentiment: Mixed/Negative
# Key points:
# - Neutral product satisfaction
# - Significant dissatisfaction with shipping time
