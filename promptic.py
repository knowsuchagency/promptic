import inspect
import json
import logging
import re
from functools import wraps
from textwrap import dedent
from typing import Callable, Dict, Any

import litellm
from pydantic import BaseModel


class Promptic:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        system: str = None,
        dry_run: bool = False,
        debug: bool = False,
        **litellm_kwargs,
    ):
        self.model = model
        self.system = system
        self.dry_run = dry_run
        self.litellm_kwargs = litellm_kwargs
        self.tools: Dict[str, Callable] = {}

        if self.tools:
            assert litellm.supports_function_calling(
                self.model
            ), f"Model {self.model} does not support function calling"

        # Setup logger with detailed format
        self.logger = logging.getLogger("promptic")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

        self.result_regex = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)

    def __call__(self, fn=None):
        return self.decorator(fn) if fn else self.decorator

    def tool(self, fn: Callable) -> Callable:
        """Register a function as a tool that can be used by the LLM"""
        self.tools[fn.__name__] = fn
        return fn

    def _generate_tool_definition(self, fn: Callable) -> dict:
        """Generate a tool definition from a function's metadata"""
        sig = inspect.signature(fn)
        doc = dedent(fn.__doc__ or "")

        parameters = {"type": "object", "properties": {}, "required": []}

        for name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect._empty else Any
            param_default = None if param.default == inspect._empty else param.default

            if param_default is None and param.default == inspect._empty:
                parameters["required"].append(name)

            param_info = {"type": "string"}  # Default to string if no type hint
            if param_type == int:
                param_info["type"] = "integer"
            elif param_type == float:
                param_info["type"] = "number"
            elif param_type == bool:
                param_info["type"] = "boolean"

            parameters["properties"][name] = param_info

        return {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": doc,
                "parameters": parameters,
            },
        }

    def decorator(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.debug(f"{self.model = }")
            self.logger.debug(f"{self.system = }")
            self.logger.debug(f"{self.dry_run = }")
            self.logger.debug(f"{self.litellm_kwargs = }")
            self.logger.debug(f"{self.tools = }")
            self.logger.debug(f"{func = }")
            self.logger.debug(f"{args = }")
            self.logger.debug(f"{kwargs = }")

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

            self.logger.debug(f"{arg_values = }")

            # Replace {name} placeholders with argument values
            prompt_text = prompt_template.format(**arg_values)

            # Check if the function has a return type hint of a Pydantic model
            return_type = func.__annotations__.get("return")

            self.logger.debug(f"{return_type = }")

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

            self.logger.debug(f"{prompt_text = }")

            messages = [{"content": prompt_text, "role": "user"}]
            if self.system:
                messages.insert(0, {"content": self.system, "role": "system"})

            # Add tools if any are registered
            tools = None
            if self.tools:
                tools = [
                    self._generate_tool_definition(tool_fn)
                    for tool_fn in self.tools.values()
                ]

            # Call the LLM with the prompt and tools
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                **self.litellm_kwargs,
            )

            if self.litellm_kwargs.get("stream"):
                return self._stream_response(response)

            # Handle tool calls if present
            if (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                tool_calls = response.choices[0].message.tool_calls
                messages.append(response.choices[0].message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    if function_name in self.tools:
                        function_args = json.loads(tool_call.function.arguments)
                        if self.dry_run:
                            self.logger.info(
                                f"[DRY RUN]: {function_name = } {function_args = }"
                            )
                            function_response = f"[DRY RUN] Would have called {function_name = } {function_args = }"
                        else:
                            function_response = self.tools[function_name](
                                **function_args
                            )
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(function_response),
                            }
                        )

                # Get final response after tool calls
                final_response = litellm.completion(
                    model=self.model, messages=messages, **self.litellm_kwargs
                )
                return final_response.choices[0].message.content

            # Get the generated text from the LLM response
            generated_text = response["choices"][0]["message"]["content"]

            self.logger.debug(f"{generated_text = }")

            if (
                return_type
                and inspect.isclass(return_type)
                and issubclass(return_type, BaseModel)
            ):
                self.logger.debug("return_type is a Pydantic model")
                # Extract the JSON result using regex
                match = self.result_regex.search(generated_text)
                if match:
                    self.logger.debug("JSON result extracted")
                    json_result = match.group(1)
                    # Parse the JSON and return an instance of the Pydantic model
                    return return_type.model_validate_json(json_result)
                else:
                    self.logger.debug(
                        "Failed to extract JSON result from the generated text."
                    )
                    raise ValueError(
                        "Failed to extract JSON result from the generated text."
                    )
            elif return_type and isinstance(return_type, dict):
                # Extract the JSON result using regex
                match = self.result_regex.search(generated_text)
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

        # Add tool decorator method to the wrapped function
        wrapper.tool = self.tool
        return wrapper

    def _stream_response(self, response):
        # Track incomplete tool calls
        current_tool_calls = {}
        current_index = None

        for part in response:
            # Handle tool calls in streaming mode
            if (
                hasattr(part.choices[0].delta, "tool_calls")
                and part.choices[0].delta.tool_calls
            ):
                tool_calls = part.choices[0].delta.tool_calls

                for tool_call in tool_calls:
                    # If we have an ID and name, this is the start of a new tool call
                    if tool_call.id:
                        current_index = tool_call.index
                        current_tool_calls[current_index] = {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": "",
                        }

                    # If we don't have an ID but have arguments, append to current tool call
                    elif tool_call.function.arguments and current_index is not None:
                        current_tool_calls[current_index]["arguments"] += (
                            tool_call.function.arguments
                        )

                        # Try to execute if arguments look complete
                        tool_info = current_tool_calls[current_index]
                        try:
                            args_str = tool_info["arguments"]
                            if (
                                args_str.strip() and args_str[-1] == "}"
                            ):  # Check if arguments look complete
                                try:
                                    function_args = json.loads(args_str)
                                    if tool_info["name"] in self.tools:
                                        if self.dry_run:
                                            function_response = f"[DRY RUN] Would have called {tool_info['name']} with {function_args}"
                                        else:
                                            function_response = self.tools[
                                                tool_info["name"]
                                            ](**function_args)
                                        yield str(function_response)
                                        # Clear after successful execution
                                        del current_tool_calls[current_index]
                                except json.JSONDecodeError:
                                    # Arguments not complete yet, continue accumulating
                                    continue
                        except Exception as e:
                            self.logger.error(f"Error executing tool: {e}")
                            continue

            # Stream regular content
            if (
                hasattr(part.choices[0].delta, "content")
                and part.choices[0].delta.content
            ):
                yield part.choices[0].delta.content


def promptic(
    fn=None,
    model="gpt-3.5-turbo",
    system: str = None,
    dry_run: bool = False,
    debug: bool = False,
    **litellm_kwargs,
):
    decorator = Promptic(
        model=model,
        system=system,
        dry_run=dry_run,
        debug=debug,
        **litellm_kwargs,
    )
    return decorator(fn) if fn else decorator


llm = promptic
