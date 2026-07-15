"""
LLM inference utilities for user agent.
"""
import datetime
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

from typing import Optional, List, Dict, Any, Union

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
 
def openai_llm_inference(
        model: str,
        messages: List[dict],
        temperature: float = None,
        stop_strs: Optional[List[str]] = None,
        max_tokens: int = None,
        api_key: str = None,
        base_url: str = None,
        enable_thinking: bool = False):
    """Call OpenAI API with retry mechanism."""
    client = OpenAI(api_key=api_key or os.getenv("USER_OPENAI_API_KEY"), base_url=base_url or os.getenv("USER_OPENAI_BASE_URL"))
    retries = 0
    max_retries = 10
    # `think` 是 Qwen/vLLM 风格的思考控制参数，OpenAI 系列（gpt-4.1 等）不认识，
    # 直接传会 400: "Unrecognized request argument supplied: think"。
    # 仅对明确支持的模型注入，其余模型不带该参数。
    model_name = str(model or "").lower()
    supports_think = "qwen" in model_name or "thinking" in model_name
    while retries < max_retries:
        start_time = _utc_now()
        try:
            request_kwargs = {
                "model": model,
                "messages": messages,
                "stop": stop_strs,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if supports_think:
                request_kwargs["extra_body"] = {"think": bool(enable_thinking)}

            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                # 服务端不认识 think 时，去掉 extra_body 再试一次，避免陷入重试死循环
                if "think" in str(exc) and "extra_body" in request_kwargs:
                    request_kwargs.pop("extra_body", None)
                    response = client.chat.completions.create(**request_kwargs)
                else:
                    raise
            output=response.choices[0].message.content
            raw_response = response.model_dump() if hasattr(response, "model_dump") else None
            return {
                "content": output,
                "usage": _response_usage_dict(response),
                "raw_response": raw_response,
                "start_time": start_time,
                "finish_time": _utc_now(),
            }
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            break
        except Exception as e:
            print(f"Something wrong:{e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries*10)
            retries += 1
    finish_time = _utc_now()
    return {"content": "", "usage": None, "raw_response": None, "start_time": finish_time, "finish_time": finish_time}
    
    
def llm_inference(model, messages, provider, api_key=None, base_url=None, enable_thinking=False):
    """Unified LLM inference interface based on provider."""
    if provider == "openai":
        return openai_llm_inference(
            model=model,
            messages=messages,
            temperature=0.7,
            api_key=api_key,
            base_url=base_url,
            enable_thinking=enable_thinking
        )
    else:
        raise ValueError(f"Invalid provider: {provider}.")
