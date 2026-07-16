"""
LLM inference utilities for action agent inference.
"""

import datetime
import os
import time
import json
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()


def _utc_now():
    return (
        datetime.datetime.now(tz=datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
    )


def _response_usage_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _response_raw_dict(response):
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return None


# 模型能力白名单 - 硬编码关键配置，避免配置文件缺失时出错
_REASONING_EFFORT_MODELS = frozenset([
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5",
    # Claude 模型使用 reasoning 参数
    "claude-opus-4-7-thinking",
    "claude-opus-4-8-thinking",
    # Gemini 模型使用 reasoning_effort 参数
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
])

_QWEN_THINKING_MODELS = frozenset([
    "qwen3.7-max",
    "qwen3.6-35b-a3b",
    "qwen3.6-27b",
    "qwen3.5-397b-a17b",
    "qwen3.5-9b",
    "qwen3.5-35b-a3b",
    "qwen3-235b-a22b-thinking",
    "qwen3-30b-a3b-thinking",
])


def _is_openai_official_base_url(base_url: str) -> bool:
    """检查 API 地址是否为官方 OpenAI endpoint。

    第三方 / 自建代理是否支持 reasoning_effort 等扩展参数因部署而异，
    无法通过 URL 特征通用判断，因此仅识别官方地址；其余情况一律通过
    `--agent-force-reasoning-effort` 显式开启（见 _supports_reasoning_effort）。
    """
    if not base_url:
        return True  # 未显式配置 base_url 时默认视为官方 API
    url = str(base_url).lower()
    return "openai.com" in url


def _supports_reasoning_effort(model: str, base_url: str = None, force: bool = False) -> bool:
    """
    判断是否应该传递 reasoning_effort 参数 (仅 GPT-5.x 系列)。

    Args:
        model: 模型名称
        base_url: API base URL
        force: 显式强制开启/关闭（对应 CLI 的 --agent-force-reasoning-effort），
            用于自建 / 第三方代理已知支持该参数、但域名无法被识别为官方 API 的情况。

    Returns:
        是否传递 reasoning_effort
    """
    model_name = str(model or "").lower()
    is_gpt5x = model_name in _REASONING_EFFORT_MODELS or model_name.startswith("gpt-5.")
    if not is_gpt5x:
        return False
    if force:
        return True
    return _is_openai_official_base_url(base_url)


def _supports_chat_template_kwargs(enable_thinking: bool, model: str) -> bool:
    """仅在明确需要 Qwen/vLLM 风格 thinking 控制时才传 chat_template_kwargs。"""
    if not bool(enable_thinking):
        return False
    model_name = str(model or "").lower()
    # 精确匹配 Qwen 模型
    if model_name in _QWEN_THINKING_MODELS:
        return True
    # 模糊匹配 qwen 关键字
    return "qwen" in model_name


_RESPONSES_API_MODELS = frozenset([
    "gpt-5.6-sol",
])


def _uses_responses_api(model: str, force: Optional[bool] = None) -> bool:
    """
    判断是否需要使用 OpenAI Responses API（而非 Chat Completions API）。

    某些模型（如 gpt-5.6-sol）在同时使用 reasoning + function calling 时
    要求走 Responses API，这是模型本身的接口要求，与具体的 base_url/代理无关。

    Args:
        force: 显式覆盖（对应 CLI --agent-use-responses-api），None 表示按模型名自动判断。
    """
    if force is not None:
        return bool(force)
    model_name = str(model or "").lower()
    return model_name in _RESPONSES_API_MODELS


def _create_chat_completion(client, *, model, messages, temperature, max_tokens, enable_thinking, extra_kwargs=None):
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }
    if isinstance(extra_kwargs, dict):
        request_kwargs.update(extra_kwargs)
    if _supports_chat_template_kwargs(enable_thinking=enable_thinking, model=model):
        request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    try:
        return client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        error_text = str(exc)
        if "chat_template_kwargs" not in error_text:
            raise
        request_kwargs.pop("extra_body", None)
        return client.chat.completions.create(**request_kwargs)


def openai_inference_prompt(
    model: str, 
    messages: List[Dict[str, Any]], 
    temperature: float = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
    ) -> Dict[str, Any]:
    """Non-streaming inference for prompt mode."""
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))
    retries = 0
    max_retries = 10
    while retries < max_retries:
        start_time = _utc_now()
        try:
            response = _create_chat_completion(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=10000,
                enable_thinking=enable_thinking,
            )
            content = response.choices[0].message.content
            # Get reasoning content if available
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning_content = response.choices[0].message.reasoning_content
            else:
                reasoning_content = ""
            # Prepend reasoning content if not empty (Qwen3 template style)
            if reasoning_content:
                reasoning_content = reasoning_content.strip()
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            return {
                "content": content,
                "usage": _response_usage_dict(response),
                "raw_response": _response_raw_dict(response),
                "start_time": start_time,
                "finish_time": _utc_now(),
            }

        except Exception as e:
            error_msg = str(e)
            # 不重试的错误类型：context length 超限、参数错误等
            non_retryable_patterns = [
                "maximum context length is",
                "max_context_length",
                "invalid_request_error",
                "BadRequestError",
                "parameter=input_tokens",
            ]
            if any(pattern in error_msg for pattern in non_retryable_patterns):
                print(f"Non-retryable error: {e}")
                raise
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)

            retries += 1
            
    finish_time = _utc_now()
    print(f"Failed to get response after {max_retries} retries.")
    return {"content": "", "usage": None, "raw_response": None, "start_time": finish_time, "finish_time": finish_time}

def openai_stream_inference_prompt(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
) -> Dict[str, Any]:
    """Inference for prompt mode with usage/raw response preserved."""
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    retries = 0
    max_retries = 10
    while retries < max_retries:
        start_time = _utc_now()
        try:
            response = _create_chat_completion(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=10000,
                enable_thinking=enable_thinking,
            )
            content = response.choices[0].message.content or ""
            reasoning_content = getattr(response.choices[0].message, "reasoning_content", "") or ""

            reasoning_content = reasoning_content.strip()
            content = content.strip()

            # Check if <think> tag is present in content
            if not reasoning_content and content and '</think>' in content:
                reasoning_content = content.split('</think>')[0].strip()
                if '<think>' in reasoning_content:
                    reasoning_content = reasoning_content.split('<think>')[1].strip()
                content = content.split('</think>')[1].strip()
            
            # Prepend reasoning content if not empty (Qwen3 template style)
            if reasoning_content:
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"

            if content == "":
                raise ValueError("content is empty.")
            return {
                "content": content,
                "usage": _response_usage_dict(response),
                "raw_response": _response_raw_dict(response),
                "start_time": start_time,
                "finish_time": _utc_now(),
            }
        
        except Exception as e:
            error_msg = str(e)
            # 不重试的错误类型：context length 超限、参数错误等
            non_retryable_patterns = [
                "maximum context length is",
                "max_context_length",
                "invalid_request_error",
                "BadRequestError",
                "parameter=input_tokens",
            ]
            if any(pattern in error_msg for pattern in non_retryable_patterns):
                print(f"Non-retryable error: {e}")
                raise
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            retries += 1

    print(f"Failed to get response after {max_retries} retries.")
    finish_time = _utc_now()
    return {"content": "", "usage": None, "raw_response": None, "start_time": finish_time, "finish_time": finish_time}

def openai_stream_inference_fc(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    tools: Optional[List[Dict]] = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None,
    reasoning_effort: str = "high",
    force_reasoning_effort: bool = False,
) -> Dict[str, Any]:
    """
    Streaming inference using official Model tool interface (function calling mode).
    Returns:
        {
            "reasoning_content": str,
            "tool_calls": list,
            "content": str
        }

    Args:
        reasoning_effort: "low", "medium", or "high" - only used for GPT-5.x models
        force_reasoning_effort: force-enable reasoning_effort even for non-official
            base_url (some self-hosted / third-party proxies accept it too)
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            start_time = _utc_now()

            # 构建 extra_kwargs
            extra_kwargs = {}
            if tools:
                extra_kwargs["tools"] = tools
                extra_kwargs["tool_choice"] = "auto"
                extra_kwargs["top_p"] = 0.95

            # 仅对支持的模型（+ 支持的 API）添加顶层 reasoning_effort 参数
            if reasoning_effort and _supports_reasoning_effort(model, base_url, force=force_reasoning_effort):
                extra_kwargs["reasoning_effort"] = reasoning_effort

            # 仅对支持的模型启用 enable_thinking
            effective_thinking = enable_thinking and (model.lower() in _QWEN_THINKING_MODELS or "qwen" in model.lower() or "thinking" in model.lower())

            if tools:
                response = _create_chat_completion(
                    client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=10000,
                    enable_thinking=effective_thinking,
                    extra_kwargs=extra_kwargs,
                )
            else:
                response = _create_chat_completion(
                    client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=10000,
                    enable_thinking=effective_thinking,
                    extra_kwargs=extra_kwargs if extra_kwargs else None,
                )

            choice = response.choices[0].message
            reasoning_content = getattr(choice, "reasoning_content", "") or ""
            content = choice.content or ""
            tool_calls = []
            for tool_call in getattr(choice, "tool_calls", []) or []:
                if hasattr(tool_call, "model_dump"):
                    tool_calls.append(tool_call.model_dump())
                else:
                    tool_calls.append(
                        {
                            "id": getattr(tool_call, "id", "") or "",
                            "type": getattr(tool_call, "type", "function") or "function",
                            "function": {
                                "name": getattr(getattr(tool_call, "function", None), "name", "") or "",
                                "arguments": getattr(getattr(tool_call, "function", None), "arguments", "") or "",
                            },
                        }
                    )

            # Check if <think> tag is present in content
            if not reasoning_content and content and '</think>' in content:
                reasoning_content = content.split('</think>')[0].strip()
                if '<think>' in reasoning_content:
                    reasoning_content = reasoning_content.split('<think>')[1].strip()
                content = content.split('</think>')[1].strip()
                
            if not content and not tool_calls and not reasoning_content:
                raise ValueError("all content is empty.")
        
            result = {
                "reasoning_content": reasoning_content,
                "tool_calls": tool_calls,
                "content": content,
                "usage": _response_usage_dict(response),
                "raw_response": _response_raw_dict(response),
                "start_time": start_time,
                "finish_time": _utc_now(),
            }
        
            return result
        
        except Exception as e:
            error_msg = str(e)
            # 不重试的错误类型：context length 超限、参数错误等
            non_retryable_patterns = [
                "maximum context length is",
                "max_context_length",
                "invalid_request_error",
                "BadRequestError",
                "parameter=input_tokens",
            ]
            if any(pattern in error_msg for pattern in non_retryable_patterns):
                print(f"Non-retryable error: {e}")
                raise
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            retries += 1

    finish_time = _utc_now()
    print(f"Failed to get response after {max_retries} retries.")
    return {
        "reasoning_content": "",
        "tool_calls": [],
        "content": "",
        "usage": None,
        "raw_response": None,
        "start_time": finish_time,
        "finish_time": finish_time,
    }


def openai_responses_inference_fc(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    tools: Optional[List[Dict]] = None,
    reasoning_effort: str = "medium",
    api_key: str = None,
    base_url: str = None,
    force_reasoning_effort: bool = False,
    omit_temperature: bool = False,
) -> Dict[str, Any]:
    """
    Inference using OpenAI Responses API (required for some reasoning + function-calling models).

    Args:
        reasoning_effort: "low", "medium", or "high"
        force_reasoning_effort: force-enable reasoning_effort even for non-official base_url
        omit_temperature: some third-party Responses API deployments reject the
            `temperature` field entirely; set this to drop it from the request
            (corresponds to CLI --agent-responses-api-omit-temperature)
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            start_time = _utc_now()

            # 将 messages 转换为 responses API 格式
            # Responses API 使用 input 字段而非 messages
            input_content = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    # Responses API 将 system 作为 developer 消息
                    input_content.append({"role": "developer", "content": content})
                elif role in ("user", "assistant"):
                    input_content.append({"role": role, "content": content})
                # 处理 tool_calls 和 tool_responses
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        input_content.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "")
                        })
                if role == "tool":
                    input_content.append({
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": content
                    })

            # 构建工具定义
            tools_input = []
            for tool in (tools or []):
                if tool.get("type") == "function":
                    fn = tool.get("function", {})
                    tools_input.append({
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {})
                    })

            request_kwargs = {
                "model": model,
                "input": input_content,
                "max_output_tokens": 10000,
                "tools": tools_input,
            }

            # 部分第三方 Responses API 部署不支持 temperature 参数，
            # 可通过 omit_temperature（对应 CLI --agent-responses-api-omit-temperature）显式关闭
            if not omit_temperature:
                request_kwargs["temperature"] = temperature or 0.7

            # 添加 reasoning 参数 (仅对支持的模型)
            # responses API 使用 reasoning={"effort": "high"} 格式
            if reasoning_effort and _supports_reasoning_effort(model, base_url, force=force_reasoning_effort):
                request_kwargs["reasoning"] = {"effort": reasoning_effort}

            response = client.responses.create(**request_kwargs)

            # 解析响应
            reasoning_content = ""
            content = ""
            tool_calls = []

            for item in getattr(response, "output", []) or []:
                if hasattr(item, "type"):
                    if item.type == "reasoning":
                        reasoning_content = getattr(item, "summary", "") or ""
                    elif item.type == "message":
                        # 部分第三方 Responses API 部署会返回字符串化的对象格式
                        raw_content = getattr(item, "content", "") or ""

                        # raw_content 可能是 list 或 str
                        if isinstance(raw_content, list) and raw_content:
                            raw_content = str(raw_content[0])

                        # 如果是字符串化的 ResponseOutputText 对象，提取 text 字段
                        if isinstance(raw_content, str) and "ResponseOutputText" in raw_content:
                            import re
                            # 匹配 text='...' 或 text="..."
                            # 使用 .*? 支持空字符串 text=''（会触发后续的空内容检查和重试）
                            match = re.search(r"text=['\"](.*?)['\"](?:\s*,\s*type=)", raw_content, re.DOTALL)
                            if match:
                                content = match.group(1)
                                # 处理转义字符
                                content = content.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
                            else:
                                # 降级：直接用整个 raw_content
                                content = raw_content
                        else:
                            content = raw_content
                    elif item.type == "function_call":
                        tool_calls.append({
                            "id": getattr(item, "call_id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(item, "name", ""),
                                "arguments": getattr(item, "arguments", "")
                            }
                        })

            # 如果没有单独的 reasoning 字段，尝试从 content 提取
            if not reasoning_content and content and '<think>' in content:
                if '</think>' in content:
                    reasoning_content = content.split('<think>')[1].split('</think>')[0].strip()
                    content = content.split('</think>')[1].strip()

            if not content and not tool_calls:
                raise ValueError("No content or tool_calls in response")

            return {
                "reasoning_content": reasoning_content,
                "tool_calls": tool_calls,
                "content": content,
                "usage": _response_usage_dict(response),
                "raw_response": _response_raw_dict(response),
                "start_time": start_time,
                "finish_time": _utc_now(),
            }

        except Exception as e:
            error_msg = str(e)
            # 不重试的错误类型：context length 超限、参数错误等
            non_retryable_patterns = [
                "maximum context length is",
                "max_context_length",
                "invalid_request_error",
                "BadRequestError",
                "parameter=input_tokens",
            ]
            if any(pattern in error_msg for pattern in non_retryable_patterns):
                print(f"Non-retryable error: {e}")
                raise
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            retries += 1

    finish_time = _utc_now()
    print(f"Failed to get response after {max_retries} retries.")
    return {
        "reasoning_content": "",
        "tool_calls": [],
        "content": "",
        "usage": None,
        "raw_response": None,
        "start_time": finish_time,
        "finish_time": finish_time,
    }


def _convert_tools_to_anthropic(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Convert OpenAI-style tool schemas to Anthropic's native tool schema.

    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}

    Anthropic requires property keys to match pattern: ^[a-zA-Z0-9_.-]{1,64}$
    This function sanitizes property keys to comply with that requirement.
    """
    import re

    if not tools:
        return None

    def _sanitize_property_key(key: str) -> str:
        """Sanitize property key to match Anthropic's pattern ^[a-zA-Z0-9_.-]{1,64}$"""
        # Replace invalid characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', key)
        # Truncate to 64 chars
        sanitized = sanitized[:64]
        # Ensure not empty
        if not sanitized:
            sanitized = "param"
        return sanitized

    def _sanitize_schema(schema: Dict) -> Dict:
        """Recursively sanitize all property keys in a schema"""
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                # Sanitize property keys
                sanitized_props = {}
                for prop_key, prop_value in value.items():
                    sanitized_key = _sanitize_property_key(prop_key)
                    sanitized_props[sanitized_key] = _sanitize_schema(prop_value)
                result[key] = sanitized_props
            elif isinstance(value, dict):
                result[key] = _sanitize_schema(value)
            elif isinstance(value, list):
                result[key] = [_sanitize_schema(item) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        return result

    converted = []
    for tool in tools:
        if "function" in tool:
            fn = tool["function"]
            parameters = fn.get("parameters", {"type": "object", "properties": {}})
            sanitized_schema = _sanitize_schema(parameters)
            converted.append({
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": sanitized_schema,
            })
        else:
            # already in Anthropic-native shape, still sanitize
            sanitized_tool = dict(tool)
            if "input_schema" in sanitized_tool:
                sanitized_tool["input_schema"] = _sanitize_schema(sanitized_tool["input_schema"])
            converted.append(sanitized_tool)
    return converted


def _convert_messages_to_anthropic(messages: List[Dict[str, Any]]):
    """Convert OpenAI-style chat messages into Anthropic's Messages API shape.

    - role "system" entries are pulled out and merged into a top-level `system` string.
    - role "tool" entries become role "user" with a tool_result content block.
    - assistant messages carrying a top-level `tool_calls` field become role "assistant"
      with `tool_use` content blocks (alongside any text content).
    Returns (system_text, converted_messages).

    CRITICAL: Anthropic API requires all tool_result blocks for a given assistant message
    to be in the SAME user message. We group consecutive role="tool" messages together.

    Additionally, if an assistant message has tool_calls but no tool results follow,
    we inject placeholder tool_result blocks.
    """
    system_parts = []
    converted = []
    pending_tool_results = []  # 累积连续的 tool 消息
    last_assistant_tool_use_ids = []  # 上一条 assistant 消息中的 tool_use ids

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            content = msg.get("content")
            if content:
                system_parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
            continue

        if role == "tool":
            # 累积 tool_result，但只能属于最近的 assistant 消息
            tool_content = msg.get("content")
            if not isinstance(tool_content, str):
                tool_content = json.dumps(tool_content, ensure_ascii=False)
            tool_call_id = msg.get("tool_call_id", "")

            # 只有当 tool_call_id 在 last_assistant_tool_use_ids 中时才累积
            # 否则说明这是一个延迟到达的 tool 结果，应该被忽略
            if tool_call_id in last_assistant_tool_use_ids:
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content,
                })
                last_assistant_tool_use_ids.remove(tool_call_id)
            continue

        # 遇到非 tool 消息时，先检查是否需要补全缺失的 tool_results
        if last_assistant_tool_use_ids:
            # 有未处理的 tool_use_ids，需要为它们添加占位符
            for missing_id in last_assistant_tool_use_ids:
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": missing_id,
                    "content": "Error: Tool execution failed or result was not captured.",
                    "is_error": True,
                })
            last_assistant_tool_use_ids = []

        # 处理累积的 tool_results
        if pending_tool_results:
            converted.append({
                "role": "user",
                "content": pending_tool_results,
            })
            pending_tool_results = []

        if role == "assistant" and msg.get("tool_calls"):
            blocks = []
            text_content = msg.get("content")
            if text_content:
                blocks.append({"type": "text", "text": text_content})

            # 记录这条 assistant 消息中的所有 tool_use ids
            last_assistant_tool_use_ids = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "")
                try:
                    parsed_input = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    parsed_input = {}
                tool_id = tc.get("id", "")
                blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": fn.get("name", ""),
                    "input": parsed_input,
                })
                last_assistant_tool_use_ids.append(tool_id)

            converted.append({"role": "assistant", "content": blocks})
            continue

        # plain user/assistant text messages pass through unchanged
        converted.append({"role": role, "content": msg.get("content", "")})

    # 处理末尾可能剩余的情况
    if last_assistant_tool_use_ids:
        for missing_id in last_assistant_tool_use_ids:
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": missing_id,
                "content": "Error: Tool execution failed or result was not captured.",
                "is_error": True,
            })

    if pending_tool_results:
        converted.append({
            "role": "user",
            "content": pending_tool_results,
        })

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, converted


def anthropic_inference_fc(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    tools: Optional[List[Dict]] = None,
    effort: str = "high",
    api_key: str = None,
    base_url: str = None
) -> Dict[str, Any]:
    """
    Anthropic Claude inference with function calling support.

    Args:
        effort: "low", "medium", "high", "xhigh", or "max" - controls token usage

    Note: For max effort, adaptive thinking is automatically enabled.
    """
    from anthropic import Timeout
    client = Anthropic(api_key=api_key, base_url=base_url, timeout=Timeout(connect=5.0, read=3600, write=600, pool=600))

    system_text, anthropic_messages = _convert_messages_to_anthropic(messages)
    anthropic_tools = _convert_tools_to_anthropic(tools)

    # DEBUG: 打印转换前后的消息结构，帮助诊断问题
    debug_enabled = False  # 关闭调试
    if debug_enabled:
        print("\n" + "="*80)
        print("[DEBUG] ORIGINAL OpenAI-style messages:")
        for i, msg in enumerate(messages):
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            content = msg.get("content", "")
            content_preview = str(content)[:80] if content else ""

            print(f"  [{i}] role={role}", end="")
            if tool_calls:
                print(f", tool_calls=[", end="")
                for tc in tool_calls:
                    print(f"{tc.get('id')}:{tc.get('function', {}).get('name')}", end=" ")
                print("]", end="")
            if tool_call_id:
                print(f", tool_call_id={tool_call_id}", end="")
            if content_preview:
                print(f", content={content_preview}...", end="")
            print()

        print("\n[DEBUG] CONVERTED Anthropic messages:")
        for i, msg in enumerate(anthropic_messages):
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                print(f"  [{i}] role={role}, content blocks:")
                for j, block in enumerate(content):
                    block_type = block.get("type")
                    if block_type == "tool_use":
                        print(f"      [{j}] tool_use: id={block.get('id')}, name={block.get('name')}")
                    elif block_type == "tool_result":
                        print(f"      [{j}] tool_result: tool_use_id={block.get('tool_use_id')}, content={str(block.get('content', ''))[:60]}...")
                    elif block_type == "text":
                        print(f"      [{j}] text: {block.get('text', '')[:60]}...")
            else:
                print(f"  [{i}] role={role}, content={str(content)[:80]}...")
        print("="*80 + "\n")

    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            start_time = _utc_now()

            # 构建请求参数
            request_kwargs = {
                "model": model,
                "max_tokens": 10000,  # 与 openai_responses_inference_fc 的 max_output_tokens 对齐，保持跨模型评测口径一致
                "messages": anthropic_messages,
                "output_config": {"effort": effort}
            }

            if system_text:
                request_kwargs["system"] = system_text

            # 对于 xhigh 和 max effort，启用 adaptive thinking（文档推荐）
            if effort in ["xhigh", "max"]:
                request_kwargs["thinking"] = {"type": "adaptive"}
                # adaptive thinking 要求 temperature 必须为 1 或不设置
                if temperature is not None and temperature != 1:
                    request_kwargs["temperature"] = 1
                elif temperature == 1:
                    request_kwargs["temperature"] = 1
            else:
                # 非 adaptive thinking 模式，正常设置 temperature
                if temperature is not None:
                    request_kwargs["temperature"] = temperature

            if anthropic_tools:
                request_kwargs["tools"] = anthropic_tools

            # 调用 API - 使用非流式模式（read timeout已扩大到3600秒）
            response = client.messages.create(**request_kwargs)

            # 解析响应
            tool_calls = []
            content = ""
            thinking_content = ""

            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        content += block.text
                    elif block.type == 'thinking':
                        thinking_content = block.thinking
                    elif block.type == 'tool_use':
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input, ensure_ascii=False)
                            }
                        })

            result = {
                "reasoning_content": thinking_content,
                "tool_calls": tool_calls,
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                    "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0
                },
                "raw_response": {
                    "id": response.id if hasattr(response, 'id') else "",
                    "model": response.model if hasattr(response, 'model') else model,
                    "stop_reason": response.stop_reason if hasattr(response, 'stop_reason') else ""
                },
                "start_time": start_time,
                "finish_time": _utc_now(),
            }

            return result

        except Exception as e:
            error_msg = str(e)
            non_retryable_patterns = [
                "maximum context length",
                "invalid_request_error",
                "BadRequestError",
            ]
            if any(pattern in error_msg for pattern in non_retryable_patterns):
                print(f"Non-retryable error: {e}")
                raise
            print(f"Anthropic API error: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10 + 10)
            retries += 1

    finish_time = _utc_now()
    print(f"Failed to get response from Anthropic after {max_retries} retries.")
    return {
        "reasoning_content": "",
        "tool_calls": [],
        "content": "",
        "usage": None,
        "raw_response": None,
        "start_time": finish_time,
        "finish_time": finish_time,
    }


def llm_inference_fc(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    tools: Optional[List[Dict]] = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None,
    reasoning_effort: str = "high",
    effort: str = "high",
    force_reasoning_effort: bool = False,
    use_responses_api: Optional[bool] = None,
    omit_temperature: bool = False,
) -> Dict[str, Any]:
    """
    Unified LLM inference interface for FC mode.

    Args:
        reasoning_effort: "low", "medium", or "high" - only used for GPT-5.x models
        effort: "low", "medium", "high", "xhigh", or "max" - only used for Claude models
        force_reasoning_effort: force-enable reasoning_effort even when base_url isn't
            recognized as an official API endpoint (for self-hosted/third-party proxies
            that support the parameter). Corresponds to CLI --agent-force-reasoning-effort.
        use_responses_api: explicitly force on/off the OpenAI Responses API path,
            overriding the model-based default. Corresponds to CLI --agent-use-responses-api.
        omit_temperature: drop the `temperature` field when calling the Responses API,
            for deployments that reject it. Corresponds to CLI
            --agent-responses-api-omit-temperature.
    """
    if provider == "openai":
        if _uses_responses_api(model, force=use_responses_api):
            return openai_responses_inference_fc(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                reasoning_effort=reasoning_effort,
                api_key=api_key,
                base_url=base_url,
                force_reasoning_effort=force_reasoning_effort,
                omit_temperature=omit_temperature,
            )
        else:
            return openai_stream_inference_fc(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                enable_thinking=enable_thinking,
                api_key=api_key,
                base_url=base_url,
                reasoning_effort=reasoning_effort,
                force_reasoning_effort=force_reasoning_effort,
            )
    elif provider == "anthropic":
        return anthropic_inference_fc(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            effort=effort,
            api_key=api_key,
            base_url=base_url
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")


def llm_inference_prompt(provider: str, model: str, messages: List[Dict[str, Any]], temperature: float = None, enable_thinking: bool = False, api_key: str = None, base_url: str = None) -> str:
    """
    Unified LLM inference interface for Prompt mode.
    """
    if provider == "openai":
        return openai_stream_inference_prompt(model=model, messages=messages, temperature=temperature, enable_thinking=enable_thinking, api_key=api_key, base_url=base_url)
    else:
        # add other provider support here
        raise ValueError(f"Invalid provider: {provider}")


if __name__ ==  "__main__":
    # Test FC mode with tools
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Beijing?"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather of a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    model = "gpt-4.1"
    provider = "openai"
    result = llm_inference_fc(
        provider=provider,
        model=model, 
        messages=msgs, 
        tools=tools,
    )
    print(result)
