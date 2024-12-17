from promptic import llm

schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[A-Z][a-z]+$",
            "minLength": 2,
            "maxLength": 20,
        },
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


@llm(json_schema=schema, system="You generate test data.")
def get_user_info(name: str) -> dict:
    """Get information about {name}"""


print(get_user_info("Alice"))
# {'name': 'Alice', 'age': 25, 'email': 'alice@example.com'}
