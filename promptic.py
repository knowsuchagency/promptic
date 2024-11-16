import inspect
import json
import logging
import os
import re
from functools import wraps
from textwrap import dedent
from typing import Callable

import litellm
from pydantic import BaseModel

logger = logging.getLogger("promptic")

if promptic_debug := os.getenv("PROMPTIC_DEBUG"):
    if promptic_debug.lower() in ["0", "false", "no"]:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())

regex = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)


def promptic(fn=None, model="gpt-3.5-turbo", system: str = None, **litellm_kwargs):
    """
    Decorator to call the LLM with a prompt generated from the function's docstring and arguments.

    Args:
        fn: The function to decorate.
        model: The LLM model to use.
        system: The system message to prepend to the prompt.
        litellm_kwargs: Additional keyword arguments to pass to litellm.completion.
    """
    logger.debug(f"{fn = }")
    logger.debug(f"{model = }")
    logger.debug(f"{litellm_kwargs = }")

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"{args = }")
            logger.debug(f"{kwargs = }")
            # Get the function's docstring as the prompt
            prompt_template = dedent(func.__doc__)

            # Get the argument names, default values and values using inspect
            sig = inspect.signature(func)
            arg_names = sig.parameters.keys()
            arg_values = {
                name: (
                    sig.parameters[name].default
                    if sig.parameters[name].default is not inspect.Parameter.empty
                    else None
                )
                for name in arg_names
            }
            arg_values.update(zip(arg_names, args))
            arg_values.update(kwargs)

            logger.debug(f"{arg_values = }")

            # Replace {name} placeholders with argument values
            prompt_text = prompt_template.format(**arg_values)

            # Check if the function has a return type hint of a Pydantic model
            return_type = func.__annotations__.get("return")

            logger.debug(f"{return_type = }")

            if (
                return_type
                and inspect.isclass(return_type)
                and issubclass(return_type, BaseModel)
            ):
                # Get the JSON schema of the Pydantic model
                schema = return_type.model_json_schema()
                json_schema = json.dumps(schema, indent=2)
                # Update the prompt to specify the expected JSON format
                prompt_text += f"\n\nThe result must conform to the following JSON schema:\n```json\n{json_schema}\n```"
                prompt_text += "\n\nProvide the result enclosed in triple backticks with 'json' on the first line. Don't put control characters in the wrong place or the JSON will be invalid."
            # if the return type is a dict, assume it's a json schema
            elif return_type and isinstance(return_type, dict):
                json_schema = json.dumps(return_type, indent=2)
                # Update the prompt to specify the expected JSON format
                prompt_text += f"\n\nThe result must conform to the following JSON schema:\n```json\n{json_schema}\n```"
                prompt_text += "\n\nProvide the result enclosed in triple backticks with 'json' on the first line. Don't put control characters in the wrong place or the JSON will be invalid."

            logger.debug(f"{prompt_text = }")

            messages = [{"content": prompt_text, "role": "user"}]

            if system:
                messages.insert(0, {"content": system, "role": "system"})

            logger.debug(f"{messages = }")

            # Call the LLM with the prompt
            response = litellm.completion(
                model=model,
                messages=messages,
                **litellm_kwargs,
            )

            if litellm_kwargs.get("stream"):
                return _stream_response(response)
            else:
                # Get the generated text from the LLM response
                generated_text = response["choices"][0]["message"]["content"]

                logger.debug(f"{generated_text = }")

                if (
                    return_type
                    and inspect.isclass(return_type)
                    and issubclass(return_type, BaseModel)
                ):
                    logger.debug("return_type is a Pydantic model")
                    # Extract the JSON result using regex
                    match = regex.search(generated_text)
                    if match:
                        logger.debug("JSON result extracted")
                        json_result = match.group(1)
                        # Parse the JSON and return an instance of the Pydantic model
                        return return_type.model_validate_json(json_result)
                    else:
                        logger.debug(
                            "Failed to extract JSON result from the generated text."
                        )
                        raise ValueError(
                            "Failed to extract JSON result from the generated text."
                        )
                elif return_type and isinstance(return_type, dict):
                    # Extract the JSON result using regex
                    match = regex.search(generated_text)
                    if match:
                        json_result = match.group(1)
                        # Parse the JSON and return the result
                        return json.loads(json_result)
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
        chunk = part.choices[0].delta.content or ""
        logger.debug(f"{chunk = }")
        yield chunk


llm = promptic
