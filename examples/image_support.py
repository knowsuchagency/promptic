from promptic import llm, ImageBytes


@llm(model="gpt-4o")  # Use a vision-capable model
def describe_image(image: ImageBytes):
    """What's in this image?"""


# Load image bytes
with open("tests/fixtures/ocai-logo.jpeg", "rb") as f:
    image_data = ImageBytes(f.read())

# Single image
print(describe_image(image_data))

print("-" * 100)


# Mixed content
@llm(model="gpt-4o")
def analyze_image_feature(image: ImageBytes, feature: str):
    """Tell me about the {feature} in this image"""


print(analyze_image_feature(image_data, "colors"))
