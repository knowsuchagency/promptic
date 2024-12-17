from tenacity import retry, wait_exponential, retry_if_exception_type
from promptic import llm
from litellm.exceptions import RateLimitError


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
@llm
def generate_summary(text):
    """Summarize this text in 2-3 sentences: {text}"""


generate_summary("Long article text here...")
