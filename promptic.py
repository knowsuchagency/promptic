import warnings

warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import inspect
import json
import logging
import re
import base64
import mimetypes
from functools import wraps
from textwrap import dedent
from typing import Callable, Dict, Any, List, Optional, Union

import litellm
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel
from litellm import completion as litellm_completion

__version__ = "5.0.1"

SystemPrompt = Optional[Union[str, List[str], List[Dict[str, str]]]]

ImageBytes = bytes


class State:
    """Base state class for managing conversation memory"""

    def __init__(self):
        self._messages: List[Dict[str, str]] = []

    def add_message(self, message: Dict[str, str]) -> None:
        """Add a message to the conversation history"""
        self._messages.append(message)

    def get_messages(
        self, prompt: str = None, limit: int = None
    ) -> List[Dict[str, str]]:
        """Retrieve messages from the conversation history
        Args:
            prompt: Optional prompt to filter messages by
            limit: Optional number of most recent messages to return
        """
        if limit is None:
            return self._messages
        return self._messages[-limit:]

    def clear(self) -> None:
        """Clear all messages from memory"""
        self._messages = []


class Promptic:
    def __init__(
        self,
        model="gpt-4o-mini",
        system: SystemPrompt = None,
        dry_run: bool = False,
        debug: bool = False,
        memory: bool = False,
        state: Optional[State] = None,
        json_schema: Optional[Dict] = None,
        cache: bool = True,
        create_completion_fn=None,
        openai_client=None,
        **completion_kwargs,
    ):
        """Initialize a new Promptic instance.

        Args:
            model (str, optional): The LLM model to use. Defaults to "gpt-4o-mini".
            system (SystemPrompt, optional): System prompt(s) to prepend to all conversations.
                Can be a string, list of strings, or list of message dictionaries. Defaults to None.
            dry_run (bool, optional): If True, tools will not be executed. Defaults to False.
            debug (bool, optional): Enable debug logging. Defaults to False.
            memory (bool, optional): Enable conversation memory. Defaults to False.
            state (State, optional): Custom state instance for memory management. Defaults to None.
            json_schema (Dict, optional): JSON schema for response validation. Defaults to None.
            cache (bool, optional): Enable response caching for Anthropic models. Defaults to True.
            openai_client (OpenAI, optional): The OpenAI client to use for API calls. Defaults to None.
            create_completion_fn (Callable, optional): The function to use for API calls. Defaults to None.
            **client_kwargs: Additional keyword arguments passed to the create_completion_fn.
        """
        assert not (openai_client and create_completion_fn), (
            "openai_client and create_completion_fn are mutually exclusive"
        )

        self.model = model
        self.system = system
        self.dry_run = dry_run
        self.completion_kwargs = completion_kwargs
        self.tools: Dict[str, Callable] = {}
        self.json_schema = json_schema
        self.openai_client = openai_client

        if self.openai_client:
            self.create_completion_fn = self.openai_client.chat.completions.create
        elif create_completion_fn:
            self.create_completion_fn = create_completion_fn
        else:
            self.create_completion_fn = litellm_completion

        self.logger = logging.getLogger("promptic")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.debug = debug

        if debug:
            self.logger.setLevel(logging.DEBUG)
            if self.create_completion_fn == litellm:
                litellm.set_verbose = True
        else:
            self.logger.setLevel(logging.WARNING)

        self.result_regex = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)

        self.memory = memory or state is not None

        if memory and state is None:
            self.state = State()
        else:
            self.state = state

        self.anthropic = self.model.startswith(("claude", "anthropic"))
        self.gemini = self.model.startswith(("gemini", "vertex"))

        self.cache = cache
        self.anthropic_cached_block_limit = 4
        self.cached_count = 0

        self.tool_definitions = None

    @property
    def system_messages(self):
        result = []

        if not self.system:
            return result
        if isinstance(self.system, str):
            result = [{"content": self.system, "role": "system"}]
        elif isinstance(self.system, list) and isinstance(self.system[0], dict):
            result = self.system
        elif isinstance(self.system, list) and isinstance(self.system[0], str):
            result = [{"content": msg, "role": "system"} for msg in self.system]
        else:
            raise ValueError("Invalid system prompt")

        result = self._set_anthropic_cache(result)

        if self.state and not self.state.get_messages():
            for msg in result:
                self.state.add_message(msg)

        return result

    def _completion(self, messages: list[dict], **kwargs):
        new_messages = self._set_anthropic_cache(messages)
        previous_messages = self.state.get_messages() if self.state else []

        completion_messages = self.system_messages + previous_messages + new_messages

        cached_count = 0

        for msg in completion_messages:
            if hasattr(msg, "get") and msg.get("cache_control"):
                if cached_count == self.anthropic_cached_block_limit:
                    msg.pop("cache_control")
                else:
                    cached_count += 1

        result = self.create_completion_fn(
            model=self.model,
            messages=completion_messages,
            tools=self.tool_definitions,
            tool_choice="auto" if self.tool_definitions else None,
            **(self.completion_kwargs | kwargs),
        )

        if self.state:
            for msg in new_messages:
                self.state.add_message(msg)

        return result

    def message(self, message: str, **kwargs):
        messages = [{"content": message, "role": "user"}]
        response = self._completion(messages, **kwargs)

        if (self.completion_kwargs | kwargs).get("stream"):
            return self._stream_response(response)
        else:
            content = response.choices[0].message.content
            return self._parse_and_validate_response(content)

    def _set_anthropic_cache(self, messages: List[dict]):
        """Set the cache control for the message if it is an Anthropic message"""
        if not (self.cache and self.anthropic):
            return messages

        for msg in messages:
            if len(str(msg.get("content"))) * 4 > 1024:
                msg["cache_control"] = {"type": "ephemeral"}

        return messages

    def __call__(self, fn=None):
        return self._decorator(fn) if fn else self._decorator

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

        # Add dummy parameter for Gemini models if the function doesn't take any arguments
        if self.gemini and not parameters.get("required"):
            parameters["properties"]["llm_invocation"] = {
                "type": "boolean",
                "description": "True if the function was invoked by an LLM",
            }
            parameters["required"].append("llm_invocation")

        return {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": doc,
                "parameters": parameters,
            },
        }

    def _parse_and_validate_response(
        self, generated_text: str, return_type=None, json_schema=None
    ):
        """Parse and validate the response according to the return type"""

        # Handle Pydantic model return types
        if return_type and issubclass(return_type, BaseModel):
            match = self.result_regex.search(generated_text)
            if match:
                json_result = match.group(1)
                if self.state:
                    self.state.add_message(
                        {"content": json_result, "role": "assistant"}
                    )
                return return_type.model_validate(json.loads(json_result))
            raise ValueError("Failed to extract JSON result from the generated text.")

        # Handle json_schema if provided
        elif json_schema:
            match = self.result_regex.search(generated_text)
            if not match:
                raise ValueError(
                    "Failed to extract JSON result from the generated text."
                )

            try:
                json_result = match.group(1)
                parsed_result = json.loads(json_result)
                # Validate against the schema
                validate_json_schema(instance=parsed_result, schema=self.json_schema)
                if self.state:
                    self.state.add_message(
                        {"content": json_result, "role": "assistant"}
                    )
                return parsed_result
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in response: {e}")
            except Exception as e:
                raise ValueError(f"Schema validation failed: {str(e)}")

        # Handle plain text responses
        else:
            if self.state:
                self.state.add_message({"content": generated_text, "role": "assistant"})
            return generated_text

    @classmethod
    def decorate(cls, func: Callable = None, **kwargs):
        """See Promptic.__init__ for valid kwargs."""
        instance = cls(**kwargs)
        return instance._decorator(func) if func else instance

    def _decorator(self, func: Callable):
        return_type = func.__annotations__.get("return")

        if (
            return_type
            and inspect.isclass(return_type)
            and issubclass(return_type, BaseModel)
            and self.json_schema
        ):
            raise ValueError(
                "Cannot use both Pydantic return type hints and json_schema validation together"
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.debug(f"{self.model = }")
            self.logger.debug(f"{self.system = }")
            self.logger.debug(f"{self.dry_run = }")
            self.logger.debug(f"{self.completion_kwargs = }")
            self.logger.debug(f"{self.tools = }")
            self.logger.debug(f"{func = }")
            self.logger.debug(f"{args = }")
            self.logger.debug(f"{kwargs = }")
            self.logger.debug(f"{self.cache = }")

            if (
                self.tools and not self.openai_client
            ):  # assume oai clients support tools
                assert litellm.supports_function_calling(self.model), (
                    f"Model {self.model} does not support function calling"
                )

            self.tool_definitions = (
                [
                    self._generate_tool_definition(tool_fn)
                    for tool_fn in self.tools.values()
                ]
                if self.tools
                else None
            )

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

            # Extract image arguments
            image_args = {}
            for name, param in sig.parameters.items():
                if param.annotation == ImageBytes and name in arg_values:
                    image_args[name] = arg_values.pop(name)

            self.logger.debug(f"{arg_values = }")

            # Replace {name} placeholders with argument values
            prompt_text = prompt_template.format(**arg_values)

            # Create the user message with text and images
            content = [{"type": "text", "text": prompt_text}]

            # Add image content
            for img_bytes in image_args.values():
                img_b64_str = base64.b64encode(img_bytes).decode("utf-8")

                if self.anthropic:
                    # Check for PNG signature
                    if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                        img_type = "image/png"
                    # Check for JPEG signature
                    elif img_bytes.startswith(b"\xff\xd8"):
                        img_type = "image/jpeg"
                    else:
                        img_type = "image/jpeg"  # fallback
                else:
                    img_type = "image/jpeg"  # default for non-Anthropic models

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                    }
                )

            user_message = {"content": content, "role": "user"}
            messages = [user_message]

            # Check if the function has a return type hint of a Pydantic model
            return_type = func.__annotations__.get("return")

            self.logger.debug(f"{return_type = }")

            # Add schema instructions before any LLM call if return type requires it
            if (
                return_type
                and inspect.isclass(return_type)
                and issubclass(return_type, BaseModel)
            ):
                schema = return_type.model_json_schema()
                json_schema = json.dumps(schema, indent=2)
                msg = {
                    "role": "user",
                    "content": (
                        "Format your response according to this JSON schema:\n"
                        f"```json\n{json_schema}\n```\n\n"
                        "Provide the result enclosed in triple backticks with 'json' "
                        "on the first line. Don't put control characters in the wrong "
                        "place or the JSON will be invalid."
                    ),
                }
                messages.append(msg)
            elif self.json_schema:
                json_schema = json.dumps(self.json_schema, indent=2)
                msg = {
                    "role": "user",
                    "content": (
                        "Format your response according to this JSON schema:\n"
                        f"```json\n{json_schema}\n```\n\n"
                        "Provide the result enclosed in triple backticks with 'json' "
                        "on the first line. Don't put control characters in the wrong "
                        "place or the JSON will be invalid."
                    ),
                }
                messages.append(msg)

            # Add check for Gemini streaming with tools
            if self.gemini and self.completion_kwargs.get("stream") and self.tools:
                raise ValueError("Gemini models do not support streaming with tools")

            self.logger.debug("Chat History:")
            for i, msg in enumerate(messages):
                self.logger.debug(f"Message {i}:")
                self.logger.debug(f"  Role: {msg.get('role', 'unknown')}")
                self.logger.debug(f"  Content: {msg.get('content')}")
                if "tool_calls" in msg:
                    self.logger.debug("  Tool Calls:")
                    for tool_call in msg["tool_calls"]:
                        self.logger.debug(f"    Name: {tool_call.function.name}")
                        self.logger.debug(
                            f"    Arguments: {tool_call.function.arguments}"
                        )
                if "tool_call_id" in msg:
                    self.logger.debug(f"  Tool Call ID: {msg['tool_call_id']}")
                    self.logger.debug(f"  Tool Name: {msg.get('name')}")

                if self.tool_definitions:
                    self.logger.debug("\nAvailable Tools:")
                    for tool in self.tool_definitions:
                        self.logger.debug(
                            f"  {tool['function']['name']}: {tool['function']['description']}"
                        )

            while True:
                # Call the LLM with the prompt and tools
                response = self._completion(messages)

                if self.completion_kwargs.get("stream"):
                    return self._stream_response(response)

                for choice in response.choices:
                    # Handle tool calls if present
                    if (
                        hasattr(choice.message, "tool_calls")
                        and choice.message.tool_calls
                    ):
                        tool_calls = choice.message.tool_calls
                        messages.append(choice.message)

                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            if function_name in self.tools:
                                function_args = json.loads(tool_call.function.arguments)
                                if self.gemini and "llm_invocation" in function_args:
                                    function_args.pop("llm_invocation")
                                if self.dry_run:
                                    self.logger.warning(
                                        f"[DRY RUN]: {function_name = } {function_args = }"
                                    )
                                    function_response = f"[DRY RUN] Would have called {function_name = } {function_args = }"
                                else:
                                    try:
                                        function_response = self.tools[function_name](
                                            **function_args
                                        )
                                    except Exception as e:
                                        self.logger.error(
                                            f"Error calling tool {function_name}({function_args}): {e}"
                                        )
                                        function_response = f"Error calling tool {function_name}({function_args}): {e}"
                                msg = {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": to_json(function_response),
                                }
                                messages.append(msg)

                    # GPT and Claude have `stop` when conversation is complete
                    # Gemini has `stop` as a finish reason when tools are used
                    elif choice.finish_reason in ["stop", "max_tokens", "length"]:
                        generated_text = choice.message.content
                        return self._parse_and_validate_response(
                            generated_text,
                            return_type=return_type,
                            json_schema=self.json_schema,
                        )

        # Add methods explicitly
        wrapper.tool = self.tool
        wrapper.clear = self.clear
        wrapper.message = self.message

        # Automatically expose all other attributes from self
        for attr_name, attr_value in self.__dict__.items():
            if not attr_name.startswith("_"):  # Skip private attributes
                setattr(wrapper, attr_name, attr_value)

        return wrapper

    def _stream_response(self, response):
        current_tool_calls = {}
        current_index = None
        accumulated_response = ""

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
                                    if (
                                        self.gemini
                                        and "llm_invocation" in function_args
                                    ):
                                        function_args.pop("llm_invocation")

                                    if tool_info["name"] in self.tools:
                                        if self.dry_run:
                                            self.logger.warning(
                                                f"[DRY RUN] Would have called {tool_info['name']} with {function_args}"
                                            )
                                        else:
                                            self.tools[tool_info["name"]](
                                                **function_args
                                            )
                                        # Clear after successful execution
                                        del current_tool_calls[current_index]
                                except json.JSONDecodeError:
                                    # Arguments not complete yet, continue accumulating
                                    continue
                        except Exception as e:
                            self.logger.error(f"Error executing tool: {e}")
                            self.logger.exception(e)
                            continue

            # Stream regular content and accumulate
            if (
                hasattr(part.choices[0].delta, "content")
                and part.choices[0].delta.content
            ):
                content = part.choices[0].delta.content
                accumulated_response += content
                yield content

        # After streaming is complete, add to state if memory is enabled
        if self.state:
            self.state.add_message(
                {"content": accumulated_response, "role": "assistant"}
            )

    def clear(self) -> None:
        """Clear all messages from the state if it exists.

        Raises:
            ValueError: If memory/state is not enabled
        """
        if not self.memory or not self.state:
            raise ValueError("Cannot clear state: memory/state is not enabled")
        self.state.clear()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)


def to_json(obj: Any) -> str:
    return json.dumps(obj, cls=CustomJSONEncoder, ensure_ascii=False)


llm = Promptic.llm = Promptic.decorate
