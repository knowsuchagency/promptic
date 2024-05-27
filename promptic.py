import inspect
import json
import re
from functools import wraps
from typing import Callable

import litellm
from pydantic import BaseModel


def promptic(fn=None, model="gpt-3.5-turbo", **litellm_kwargs):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the function's docstring as the prompt
            prompt_template = func.__doc__

            # Get the argument names and values using inspect
            arg_names = inspect.signature(func).parameters.keys()
            arg_values = dict(zip(arg_names, args))
            arg_values.update(kwargs)

            # Replace {name} placeholders with argument values
            prompt_text = prompt_template.format(**arg_values)

            # Check if the function has a return type hint of a Pydantic model
            return_type = func.__annotations__.get("return")
            if return_type and issubclass(return_type, BaseModel):
                # Get the JSON schema of the Pydantic model
                schema = return_type.model_json_schema()
                json_schema = json.dumps(schema, indent=2)
                # Update the prompt to specify the expected JSON format
                prompt_text += f"\n\nThe result should conform to the following JSON schema:\n```json\n{json_schema}\n```"
                prompt_text += "\n\nPlease provide the result enclosed in triple backticks with 'json' on the first line."

            # Call the LLM with the prompt
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "content": prompt_text,
                        "role": "user",
                    },
                ],
                **litellm_kwargs,
            )

            if litellm_kwargs.get("stream"):
                return _stream_response(response)
            else:
                # Get the generated text from the LLM response
                generated_text = response["choices"][0]["message"]["content"]

                if return_type and issubclass(return_type, BaseModel):
                    # Extract the JSON result using regex
                    match = re.search(r"```json\n(.*?)\n```", generated_text, re.DOTALL)
                    if match:
                        json_result = match.group(1)
                        # Parse the JSON and return an instance of the Pydantic model
                        return return_type.model_validate_json(json_result)
                    else:
                        raise ValueError(
                            "Failed to extract JSON result from the generated text."
                        )
                else:
                    # Return the generated text as is
                    return generated_text

        return wrapper

    return decorator(fn) if fn else decorator


def _stream_response(response):
    for part in response:
        yield part.choices[0].delta.content or ""


llm = promptic
