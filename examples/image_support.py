from promptic import llm, ImageBytes


@llm(model="gpt-4o")  # Use a vision-capable model
def describe_image(image: ImageBytes):
    """What's in this image?"""


with open("tests/fixtures/ocai-logo.jpeg", "rb") as f:
    image_data = ImageBytes(f.read())

print(describe_image(image_data))
# The image features an illustration of a cheerful orange with glasses sitting on a laptop. There are small, sparkling stars surrounding the orange. Below the illustration, it says "Orange County AI."


@llm(model="gpt-4o")
def analyze_image_feature(image: ImageBytes, feature: str):
    """Tell me about the {feature} in this image in a sentence or less."""


print(analyze_image_feature(image_data, "colors"))
# The image features vibrant orange, green, and black colors against a light background.
