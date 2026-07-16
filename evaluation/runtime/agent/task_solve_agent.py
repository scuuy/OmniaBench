"""
Task solving agent for interactive environments.
"""
import datetime
from copy import deepcopy
from agent.system_prompt_util import conversational_system_prompt, non_conversational_system_prompt, conversational_system_prompt_en, non_conversational_system_prompt_en, merge_tools_into_system_prompt
from agent.agent_llm_inference import llm_inference_fc, llm_inference_prompt
from omniabench_env.utils.env_util import get_state_info


def _fix_missing_items_in_tools(schema):
    """
    Recursively fix missing 'items' in OpenAI JSON schemas where 'type' is 'array', 
    which causes 400 invalid_function_parameters errors.
    """
    if isinstance(schema, dict):
        if "type" in schema:
            t = schema["type"]
            # Check if it's an array type (string 'array' or list containing 'array')
            if (t == "array" or (isinstance(t, list) and "array" in t)) and "items" not in schema:
                schema["items"] = {"type": "string"}  # Default fallback items type
        for k, v in schema.items():
            _fix_missing_items_in_tools(v)
    elif isinstance(schema, list):
        for v in schema:
            _fix_missing_items_in_tools(v)
    return schema

class TaskSolveAgent:
    """Agent that solves tasks in interactive environments using LLM inference."""
    def __init__(
        self,
        env_name,
        env,
        model,
        provider,
        temperature,
        infer_mode,
        max_steps,
        enable_thinking,
        api_key=None,
        base_url=None,
        extra_system_prompt="",
        lang="cn",
        reasoning_effort="high",
        effort="high",
        force_reasoning_effort=False,
        use_responses_api=None,
        omit_temperature=False,
    ):
        self.env_name = env_name
        self.env = env

        # LLM settings
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        assert infer_mode in ["prompt", "fc"]  # prompt: tool use via prompts, fc: tool use via function calling interface
        self.infer_mode = infer_mode
        self.enable_thinking = enable_thinking
        # reasoning_effort 用于 gpt-5.x 的 responses API
        self.reasoning_effort = reasoning_effort or "high"
        # effort 用于 Claude 的 output_config
        self.effort = effort or "high"
        # 以下三项用于兼容自建/第三方 OpenAI 兼容代理（详见 agent_llm_inference.llm_inference_fc）
        self.force_reasoning_effort = bool(force_reasoning_effort)
        self.use_responses_api = use_responses_api
        self.omit_temperature = bool(omit_temperature)
        self.extra_system_prompt = str(extra_system_prompt or "").strip()
        self.lang = lang

        # Runtime settings
        self.max_steps = max_steps
        self.max_tool_errors = 20

        # Runtime state
        self.messages = []  # Conversation history
        self.current_observation = None
        self.current_info = None
        self.total_reward = 0.0
        self.terminated = False  # Environment terminated normally
        self.truncated = False  # Environment truncated
        self.step_count = 0
        self.tool_error_count = 0

        # Trajectory recording
        self.trajectory = []  # Detailed execution information for each step

    @staticmethod
    def _utc_now():
        return (
            datetime.datetime.now(tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
        )

    def _append_message(self, role, content, **extra):
        timestamp = self._utc_now()
        message = {
            "role": role,
            "content": content,
            "start_time": extra.pop("start_time", timestamp),
            "finish_time": extra.pop("finish_time", timestamp),
        }
        for key, value in extra.items():
            if value is not None:
                message[key] = deepcopy(value)
        self.messages.append(message)
        return message

    def reset(self, task_index=None):
        """Reset environment and conversation history."""
        # Reset environment and get initial observation
        observation, info = self.env.reset(task_index=task_index)

        # Get environment introduction and available tools, build system prompt
        self.tools = _fix_missing_items_in_tools(deepcopy(info["tools"]))
        self.user_tools = info.get("user_tools", [])
        # Select system prompt based on conversation/non-conversation mode
        if self.env_name in ["tau_bench_retail", "tau_bench_airline", "omniabench_conversation_rl", "omniabench_conversation_sft", "conv_custom_wo_reward", "acebench_multi_turn"] :
            system_prompt = conversational_system_prompt_en if self.lang == "en" else conversational_system_prompt
        elif self.env_name in ["omniabench_non_conversation_rl", "omniabench_non_conversation_sft","bfcl", "acebench_multi_step"]:
            system_prompt = non_conversational_system_prompt_en if self.lang == "en" else non_conversational_system_prompt 
        else:
            raise RuntimeError(f"Unknown env_name: {self.env_name}")  
        
        # Add environment introduction to system prompt if available
        if "env_introduction" in info and info["env_introduction"]:
            system_prompt = f"{system_prompt}\n\nThe following is an introduction to the current environment:\n{info['env_introduction']}"       

        if self.extra_system_prompt:
            system_prompt = f"{system_prompt}\n\nAdditional execution hint:\n{self.extra_system_prompt}"
        
        # In prompt mode, merge tool information into system prompt
        if self.infer_mode == "prompt":
            system_prompt = merge_tools_into_system_prompt(system_prompt=system_prompt, tools=info["tools"])

        # Extract task info for logging
        task_item = deepcopy(info["task"])
        if self.env_name in ["tau_bench_retail", "tau_bench_airline"]:
            self.task_info = {"user_id": task_item.user_id, "instruction": task_item.instruction}
        elif self.env_name in ["omniabench_non_conversation_rl", "omniabench_conversation_rl", "omniabench_non_conversation_sft", "omniabench_conversation_sft"]:
            self.task_info = {
                "env_id": task_item.get("env_id"),
                "task_id": task_item.get("task_id"),
                "task": task_item.get("task"),
                "global_id": task_item.get("global_id"),
                "lang": task_item.get("lang"),
            }
            # Preserve ground truth fields for visualization
            for field in [
                "persona_idx",
                "persona",
                "persona_card_md",
                "tool_param_plan",
                "selected_serial_order",
                "execution_chain",
                "fs_inputs",
                "fs_runtime_bootstrap",
                "bootstrap_info",
                "strategy_config",
                "strategy_injected_operations",
                "dag_id",
                "tool_calls",
                "reference_tool_sequence",
                "rubrics_text",
                "rubric_items",
                "rubric_count",
                "rubric_total_score",
                "general_rubric_score",
                "task_specific_rubric_score",
                "rubric_explanation",
            ]:
                if field in task_item:
                    self.task_info[field] = deepcopy(task_item[field])
        elif self.env_name in ["bfcl"]:
            self.task_info = {"id": task_item["id"], "questions": task_item["questions"], "involved_classes": task_item["involved_classes"]}
        elif self.env_name in ["acebench_multi_step", "acebench_multi_turn"]:
            self.task_info = {"id": task_item["id"], "question": task_item["question"], "involved_classes": task_item["involved_classes"]}
        else:
            raise RuntimeError(f"Unknown env_name: {self.env_name}")


        # Initialize runtime state and message history
        self.messages = []
        self._append_message("system", system_prompt)
        self.current_observation = deepcopy(observation)
        self.current_info = deepcopy(info)
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.user_messages = None
        self.step_count = 0
        self.tool_error_count = 0
        self.trajectory = []
        self._append_message("user", observation)

        # Record initial trajectory information
        self.trajectory.append({
            "step": self.step_count,
            "observation": observation,
            # "info": info,
        })

        return observation, info

    def _append_observation_message(self, observation, tool_call=None):
        """Append environment observation back to the conversation history."""
        if observation["type"] == "tool":
            if self.infer_mode == "prompt":
                observation_content = f"<tool_response>\n{observation['content']}\n</tool_response>"
                self._append_message(
                    "user",
                    observation_content,
                    start_time=(self.current_info or {}).get("step_start_time"),
                    finish_time=(self.current_info or {}).get("step_finish_time"),
                )
            else:
                if tool_call:
                    self._append_message(
                        "tool",
                        observation["content"],
                        start_time=(self.current_info or {}).get("step_start_time"),
                        finish_time=(self.current_info or {}).get("step_finish_time"),
                        tool_call_id=tool_call.get("id", ""),
                        name=((tool_call.get("function") or {}).get("name") or ""),
                    )
                else:
                    # FC 模式下若没有可绑定的 tool_call，只能退化为普通 user 观察。
                    self._append_message(
                        "user",
                        observation["content"],
                        start_time=(self.current_info or {}).get("step_start_time"),
                        finish_time=(self.current_info or {}).get("step_finish_time"),
                    )
        else:
            self._append_message(
                "user",
                observation["content"],
                start_time=(self.current_info or {}).get("step_start_time"),
                finish_time=(self.current_info or {}).get("step_finish_time"),
            )

    def _apply_env_step_result(self, observation, reward, terminated, truncated, info, action, tool_call=None):
        """Update runtime state after one environment step and record trajectory/messages."""
        self.step_count += 1
        self.total_reward += float(reward or 0.0)
        self.current_observation = observation
        self.current_info = info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        if isinstance(info, dict) and info.get("tool_exception"):
            self.tool_error_count += 1
        elif isinstance(info, dict) and info.get("error_type") in {"parse_response_error", "parse_action_error", "invalid_action"}:
            self.tool_error_count += 1

        self._append_observation_message(observation=observation, tool_call=tool_call)

        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "state_diff_ref": deepcopy((info or {}).get("state_diff_ref")),
            "start_time": (info or {}).get("step_start_time"),
            "finish_time": (info or {}).get("step_finish_time"),
            # "info": info,
            # Note: updated_state 已移除，状态快照通过 state_diff_ref 指向外部文件
        })
        if "user_messages" in info:
            self.user_messages = info["user_messages"]
        if self.tool_error_count >= int(self.max_tool_errors):
            self.terminated = True
            self.truncated = True
            self.current_observation = {
                "type": "user",
                "content": "Error: Too many tool errors",
            }
            self.current_info = deepcopy(self.current_info or {})
            self.current_info["termination_reason"] = "TOO_MUCH_TOOL_ERRORS"

    def _execute_fc_tool_calls(self, raw_response, max_steps=None):
        """Execute all FC tool calls in order and bind each tool result to its tool_call_id."""
        tool_calls = raw_response.get("tool_calls") or []
        if not tool_calls:
            return None, 0.0, False, False, {"action": {}}, {}

        last_observation = None
        last_reward = 0.0
        last_terminated = False
        last_truncated = False
        last_info = {"action": {}}
        last_action = {}

        for tool_call in tool_calls:
            if max_steps is not None and self.step_count >= max_steps:
                break

            single_tool_response = {
                "reasoning_content": "",
                "content": "",
                "tool_calls": [deepcopy(tool_call)],
            }
            observation, reward, terminated, truncated, info = self.env.step(action=single_tool_response)
            action = info["action"]
            self._apply_env_step_result(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                action=action,
                tool_call=tool_call,
            )

            last_observation = observation
            last_reward = reward
            last_terminated = terminated
            last_truncated = truncated
            last_info = info
            last_action = action

            if terminated or truncated:
                break

        return last_observation, last_reward, last_terminated, last_truncated, last_info, last_action

    def step(self, max_steps=None):
        """
        Execute one environment step:
        1) Call LLM with current messages to generate response
        2) Pass LLM response as action to env.step
        3) Update conversation history, accumulated reward, trajectory, etc.
        Returns: (observation, reward, terminated, truncated, info, action)
        """

        # Check if environment has already finished
        if self.terminated or self.truncated:
            raise RuntimeError("Environment already finished. Please reset before calling step again.")

        # Call LLM inference (response parsing is done in env)
        if self.infer_mode == "prompt":
            # Prompt mode: returns LLM text (str)
            raw_response = llm_inference_prompt(
                provider=self.provider,
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking,
                api_key=self.api_key,
                base_url=self.base_url
            )
            response_text = str(raw_response.get("content", ""))
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()
        else:
            # FC mode: returns (reasoning_content, tool_calls, content) as dict
            raw_response = llm_inference_fc(
                provider=self.provider,
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                tools=self.tools,
                enable_thinking=self.enable_thinking,
                api_key=self.api_key,
                base_url=self.base_url,
                reasoning_effort=self.reasoning_effort,
                effort=self.effort,
                force_reasoning_effort=self.force_reasoning_effort,
                use_responses_api=self.use_responses_api,
                omit_temperature=self.omit_temperature,
            )

        # Add model output to conversation history
        if self.infer_mode == "prompt":
            message_content = response_text
            self._append_message(
                "assistant",
                message_content,
                start_time=raw_response.get("start_time"),
                finish_time=raw_response.get("finish_time"),
                usage=raw_response.get("usage"),
                raw_response=raw_response.get("raw_response"),
            )
        else:
            self._append_message(
                "assistant",
                raw_response["content"],
                start_time=raw_response.get("start_time"),
                finish_time=raw_response.get("finish_time"),
                tool_calls=raw_response.get("tool_calls") or None,
                reasoning_content=raw_response.get("reasoning_content") or None,
                usage=raw_response.get("usage"),
                raw_response=raw_response.get("raw_response"),
            )

        # Execute one environment step
        if self.infer_mode == "prompt" and response_text == '':  # Special handling for empty response
            print("raw_response is empty, please check the model")
            observation, reward, terminated, truncated, info, action = (
                {"type": "user", "content": "action is empty, please check the model"},
                0,
                True,
                True,
                {"action": "", "termination_reason": "INFRA_ERROR"},
                "",
            )
            self.current_info = deepcopy(info)
            self.current_observation = deepcopy(observation)
            self.terminated = True
            self.truncated = True
        elif self.infer_mode == "fc" and raw_response.get("tool_calls"):
            observation, reward, terminated, truncated, info, action = self._execute_fc_tool_calls(
                raw_response=raw_response,
                max_steps=max_steps,
            )
        else:
            env_action = response_text if self.infer_mode == "prompt" else raw_response
            observation, reward, terminated, truncated, info = self.env.step(action=env_action)
            action = info["action"]
            self._apply_env_step_result(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                action=action,
            )
        return observation, reward, terminated, truncated, info, action

    def run(self, task_index=None, max_steps=None, timeout_seconds=3600):
        """
        Run agent from reset, looping step until:
        - terminated is True, or
        - truncated is True, or
        - max_steps is reached
        Returns execution result dictionary.
        """
        max_steps = max_steps if max_steps is not None else self.max_steps
        return self.run_with_checkpoint(
            task_index=task_index,
            max_steps=max_steps,
            checkpoint_callback=None,
            checkpoint_every_steps=0,
            timeout_seconds=timeout_seconds,
        )

    def _build_result(self):
        """构造当前运行快照结果。"""
        return {
            "task_info": deepcopy(self.task_info),
            "tools": deepcopy(self.tools),
            "messages": deepcopy(self.messages),
            "user_messages": deepcopy(self.user_messages),
            "trajectory": deepcopy(self.trajectory),
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_observation": deepcopy(self.current_observation),
            "final_info": deepcopy(self.current_info),
            "steps": self.step_count,
            "termination_reason": str(
                (self.current_info or {}).get("termination_reason")
                or (self.current_info or {}).get("terminated_by")
                or ("MAX_TURNS" if self.truncated else "COMPLETED")
            ),
        }

    def run_with_checkpoint(self, task_index=None, max_steps=None, checkpoint_callback=None, checkpoint_every_steps=0, timeout_seconds=3600):
        """
        运行任务，并可按步回调中间快照，供外部做断点落盘。

        Args:
            timeout_seconds: 单个 task 的最大执行时间（秒），默认 3600 秒（1 小时）
        """
        max_steps = max_steps if max_steps is not None else self.max_steps

        # Initialize environment
        self.reset(task_index=task_index)
        task_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Loop execution
        while (not self.terminated) and (not self.truncated) and (self.step_count < max_steps):
            # Check timeout
            elapsed = (datetime.datetime.now(tz=datetime.timezone.utc) - task_start_time).total_seconds()
            if elapsed > timeout_seconds:
                print(f"[TIMEOUT] Task exceeded {timeout_seconds}s limit ({elapsed:.1f}s elapsed). Terminating.")
                self.terminated = True
                self.truncated = True
                self.current_info = self.current_info or {}
                self.current_info["termination_reason"] = "TIMEOUT"
                self.current_info["timeout_seconds"] = timeout_seconds
                self.current_info["elapsed_seconds"] = elapsed
                break
            self.step(max_steps=max_steps)
            if (
                callable(checkpoint_callback)
                and int(checkpoint_every_steps or 0) > 0
                and self.step_count > 0
                and self.step_count % int(checkpoint_every_steps) == 0
            ):
                checkpoint_callback(self._build_result())

        # If the loop exits due to max_steps rather than env termination,
        # still compute a final checklist evaluation from the current state.
        if (
            (not self.terminated)
            and (not self.truncated)
            and (self.step_count >= max_steps)
            and hasattr(self.env, "calculate_reward")
            and hasattr(self.env, "checklist_with_func")
            and hasattr(self.env, "init_state")
            and hasattr(self.env, "env_instance")
        ):
            try:
                pred_final_state = get_state_info(self.env.env_instance)
                self.env.pred_final_state = pred_final_state
                reward = self.env.calculate_reward(self.env.checklist_with_func, self.env.init_state, pred_final_state)
                self.total_reward = float(reward or 0.0)
                self.current_info = deepcopy(self.current_info or {})
                self.current_info["terminated_by"] = "max_steps"
                self.current_info["termination_reason"] = "MAX_TURNS"
                self.truncated = True
            except Exception:
                pass

        # Aggregate results
        return self._build_result()
