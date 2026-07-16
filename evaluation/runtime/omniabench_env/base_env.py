"""
Base environment class for OmniaBench.
"""
import os
import json
import random
import traceback
import inspect
import datetime
from copy import deepcopy

from omniabench_env.utils.env_util import (
    init_env_class,
    init_env_instance,
    prepare_env_instance_runtime,
    get_state_diff,
    get_state_info,
    run_check_function,
)
from omniabench_env.utils.parse_util import parse_response, parse_action


class OmniaBenchBaseEnv:
    """
    Base environment class:
    - Manages task dataset and environment dataset loading
    - Provides reset/step workflow
    - Records trajectory and calculates rewards
    Subclasses must implement abstract methods (construct prompt, initial observation, termination conditions, etc.)
    """

    def __init__(self, mode, env_items_path=None, task_items_path=None, verbose=True):
        self.mode = mode
        self.verbose = bool(verbose)

        # Load task dataset and environment dataset
        if task_items_path is not None:
            self.task_items = json.load(open(task_items_path, encoding="utf-8"))
            if self.verbose:
                print(f"Ignore the mode {self.mode}.\nLoad task_items from {task_items_path}, total {len(self.task_items)} tasks!")
        else:
            self.task_items = self.load_task_items()
        if env_items_path is not None:
            self.env_items = json.load(open(env_items_path, encoding="utf-8"))
            if self.verbose:
                print(f"Load env_items from {env_items_path}, total {len(self.env_items)} envs!")
        else:
            # 单文件模式：当 task_items 内嵌 env 信息时，直接从 task_items 构建 env_items
            extracted_env_items = self.extract_env_items_from_task_items(self.task_items)
            if extracted_env_items is not None:
                self.env_items = extracted_env_items
                if self.verbose:
                    print(f"Load env_items from task_items (single-file mode), total {len(self.env_items)} envs!")
            else:
                if task_items_path is not None:
                    raise ValueError(
                        "env_items_path is None and task_items do not contain embedded env info "
                        "(requires env_item or env_class_code+tools). "
                        "Please run single-file preprocessing or pass env_items_path."
                    )
                self.env_items = self.load_env_items()

        # Initialize logs and environment state
        self.reset_attributes()

    # ==============================
    # Data loading methods
    # ==============================

    def load_env_items(self):
        """Load environment dataset."""
        folder_path = os.path.join(os.path.dirname(__file__), "data")
        env_items_path = os.path.join(folder_path, "your_env_items.json")
        with open(env_items_path, encoding="utf-8") as f:
            env_items = json.load(f)

        if self.verbose:
            print(f"Load {len(env_items)} envs from {env_items_path}!")
        return env_items


    def load_task_items(self):
        """Load task dataset."""
        folder_path = os.path.join(os.path.dirname(__file__), "data")

        if self.mode == "eval":
            task_items_path = os.path.join(folder_path, "your_eval_task_items.json")
        elif self.mode == "train":
            task_items_path = os.path.join(folder_path, "your_train_task_items.json")
        else:
            raise ValueError("mode must be eval or train")

        with open(task_items_path, encoding="utf-8") as f:
            task_items = json.load(f)

        if self.verbose:
            print(f"Load {len(task_items)} tasks from {task_items_path}!")
        return task_items

    def extract_env_items_from_task_items(self, task_items):
        """
        尝试从 task_items 中提取 env_items（单文件模式）。
        支持两种嵌入方式：
        1) task_item["env_item"] = {...}
        2) task_item 直接包含 env 关键字段（env_class_code/tools 等）
        """
        if not isinstance(task_items, list):
            return None

        env_items = {}
        for task in task_items:
            if not isinstance(task, dict):
                continue
            env_id = task.get("env_id")
            if not env_id:
                continue

            if isinstance(task.get("env_item"), dict):
                env_items[env_id] = deepcopy(task["env_item"])
                continue

            # 兜底：从 task 平铺字段提取最小 env_item
            # 支持两种字段名：tools 或 candidate_tools（route3 使用）
            tools_field = "tools" if "tools" in task else ("candidate_tools" if "candidate_tools" in task else None)
            has_minimal_env = ("env_class_code" in task) and (tools_field is not None)
            if has_minimal_env:
                # env_class_name 缺失时从 env_class_code 中提取类名（最后一个 class 定义）
                env_class_name = task.get("env_class_name")
                if not env_class_name:
                    import re
                    class_code = task.get("env_class_code", "")
                    # 只匹配真正的类定义：行首（可选缩进）+ class 关键字 + 类名 + ( 或 :
                    classes = re.findall(r'^(?:\s*)class\s+(\w+)(?:\s*\(|\s*:)', class_code, re.MULTILINE)
                    env_class_name = classes[-1] if classes else "OmniaBenchNonConvRLEnv"
                env_item = {
                    "env_id": env_id,
                    "env_class_name": env_class_name,
                    "environment_introduction": task.get("environment_introduction", ""),
                    "constraints_rules": task.get("constraints_rules", []),
                    "env_class_code": task.get("env_class_code"),
                    "tools": task.get(tools_field, []),
                }
                for key in ("strategy_config", "strategy_injected_operations"):
                    value = task.get(key)
                    if value not in (None, "", [], {}):
                        env_item[key] = deepcopy(value)
                env_items[env_id] = env_item

        if not env_items:
            return None
        return env_items

    # ==============================
    # State reset
    # ==============================

    def reset_attributes(self):
        """Reset class attributes (logs and environment state)."""
        # Log related
        self.current_step = 0
        self.trajectory = []

        # Environment related
        self.env_item = None
        self.env_class = None
        self.env_instance = None
        self.system_prompt = None

        # Scenario related (initial state, task item, check functions, etc.)
        self.init_config = None
        self.init_state = None
        self.pred_final_state = None
        self.task_item = None
        self.checklist_with_func = None
        self.checklist_eval = None


    def reset(self, seed=None, task_index=None):
        """Reset environment and return initial observation + tool info + task info."""
        self.reset_attributes()

        if seed is not None:
            random.seed(seed)

        # Randomly select task or load specified task
        if task_index is None:
            task_index = random.randrange(0, len(self.task_items))

        self.task_item = self.task_items[task_index]
        self.checklist_with_func = self.task_item.get("checklist_with_func", [])
        self.task_id = self.task_item["task_id"]
        self.env_id = self.task_item["env_id"]
        self.init_config = self.task_item["init_config"]

        # Load environment and instance
        self.load_env_and_instance(env_id=self.env_id, init_config=self.init_config)
        # Construct system prompt
        self.env_introduction = self.construct_env_introduction(env_item=self.env_item)
        # Get tool list
        self.tools = deepcopy(self.env_item["tools"])
        # Get initial observation
        init_observation = self.get_initial_observation(task_item=self.task_item)

        info = deepcopy({"env_introduction": self.env_introduction, "tools": self.tools, "task": self.task_item})
        return init_observation, info


    # ==============================
    # Load environment and initialize instance
    # ==============================
    
    def load_env_and_instance(self, env_id: str, init_config: dict):
        """Load environment and initialize instance based on env_id."""
        # 单文件模式下 env_id 并非全局唯一：多个 task 可能共用同一个 env_id
        # 却各自携带不同的 env_class_code/env_class_name。此时按 env_id 归并的
        # self.env_items 只会保留最后一个，导致取到错误的环境类。因此优先使用
        # 当前 task_item 自带的 env_item，回退到按 env_id 查表（多文件模式）。
        task_env_item = self.task_item.get("env_item") if isinstance(self.task_item, dict) else None
        if isinstance(task_env_item, dict) and task_env_item.get("env_class_code"):
            self.env_item = task_env_item
        else:
            if env_id not in self.env_items:
                raise ValueError(f"Invalid env_id '{env_id}', not found in env_items")
            self.env_item = self.env_items[env_id]
        env_class_code = self.env_item["env_class_code"]
        # env_class_name: prefer task_item, fallback to env_item (for single-file extracted envs)
        env_class_name = self.task_item.get("env_class_name") or self.env_item.get("env_class_name") or "OmniaBenchNonConvRLEnv"
        # Initialize environment class and instance
        self.env_class = init_env_class(env_class_code, env_class_name)
        self.env_instance = init_env_instance(self.env_class, init_config)
        prepare_env_instance_runtime(
            self.env_instance,
            task_item=self.task_item,
            env_item=self.env_item,
            env_id=env_id,
        )
        # Save initial state
        self.init_state = get_state_info(self.env_instance)
        # Initial trajectory record
        self.trajectory.append({
            "step": 0,
            "state_snapshot": deepcopy(self.init_state)
        })

    # ==============================
    # Environment interaction step
    # ==============================

    def step(self, action: str | dict):
        """Execute one step of environment interaction, return observation, reward, terminated, truncated, info."""
        raw_response = deepcopy(action)
        step_start_time = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
        )
        
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {"action": raw_response}
        info["step_start_time"] = step_start_time
        

        # Parse response to action dict
        # String response needs additional parsing to struct_response
        if isinstance(raw_response, str):
            parse_success, struct_response = self._parse_response(text_response=raw_response)
            if not parse_success:
                observation = {"type": "user", "content": "Error: Failed to parse response to struct response"}
                info.update({"error_type": "parse_response_error", "termination_reason": "INFRA_ERROR"})
                self._record_step(action, observation, terminated, reward)
                info["step_finish_time"] = (
                    datetime.datetime.now(tz=datetime.timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat()
                )
                info["post_state_snapshot"] = deepcopy(self.trajectory[-1]["state_snapshot"])
                info["state_diff_ref"] = {"step": self.trajectory[-1]["step"]}
                return observation, reward, terminated, truncated, info
        else:
            struct_response = raw_response
        
        parse_success, action = self._parse_action(struct_response)
        if not parse_success:
            observation = {"type": "user", "content": "Error: Failed to parse response to action"}
            info.update({"error_type": "parse_action_error", "termination_reason": "INFRA_ERROR"})
            self._record_step(action, observation, terminated, reward)
            info["step_finish_time"] = (
                datetime.datetime.now(tz=datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
            )
            info["post_state_snapshot"] = deepcopy(self.trajectory[-1]["state_snapshot"])
            info["state_diff_ref"] = {"step": self.trajectory[-1]["step"]}
            return observation, reward, terminated, truncated, info
    
        info.update({"action": action})
        
        # Check action validity
        if not self.check_vaild_action(action=action):
            observation = {"type": "user", "content": "Error: Invalid action"}
            info.update({"error_type": "invalid_action", "termination_reason": "INFRA_ERROR"})
            self._record_step(action, observation, terminated, reward)
            info["step_finish_time"] = (
                datetime.datetime.now(tz=datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
            )
            info["post_state_snapshot"] = deepcopy(self.trajectory[-1]["state_snapshot"])
            info["state_diff_ref"] = {"step": self.trajectory[-1]["step"]}
            return observation, reward, terminated, truncated, info

        # Check if action is termination action
        if self.is_action_terminated(action):
            observation = {"type": "user", "content": "Task finished"}
            terminated = True
            self.pred_final_state = get_state_info(self.env_instance)
            reward = self.calculate_reward(self.checklist_with_func, self.init_state, self.pred_final_state)
            info.update({"checklist_eval": deepcopy(self.checklist_eval), "termination_reason": "USER_STOP"})
            self._record_step(action, observation, terminated, reward)
            info["step_finish_time"] = (
                datetime.datetime.now(tz=datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
            )
            info["post_state_snapshot"] = deepcopy(self.trajectory[-1]["state_snapshot"])
            info["state_diff_ref"] = {"step": self.trajectory[-1]["step"]}
            
            if hasattr(self, "user_agent"):
                user_messages = self.user_agent.get_messages()
                info.update({"user_messages": user_messages})
            
            return observation, reward, terminated, truncated, info

        try:
            # Call environment method
            if action["name"] == "chat_with_user":
                observation = {"type": "user", "content": self.user_agent.user_step(agent_response=action['arguments']['content'])}
            else:
                cleaned_args, dropped_keys = self._sanitize_tool_arguments(
                    method_name=action["name"],
                    arguments=action.get("arguments", {}),
                )
                action["arguments"] = cleaned_args
                if dropped_keys:
                    info["dropped_arguments"] = dropped_keys
                observation = {"type": "tool", "content": f"{getattr(self.env_instance, action['name'])(**cleaned_args)}"}
            
            # Check if observation is termination observation
            if self.is_observation_terminated(action, observation):
                terminated = True
                # Once finished, record final state snapshot and calculate reward
                self.pred_final_state = get_state_info(self.env_instance)
                reward = self.calculate_reward(self.checklist_with_func, self.init_state, self.pred_final_state)
                info.update({"checklist_eval": deepcopy(self.checklist_eval), "termination_reason": "USER_STOP"})

            # Record and return
            self._record_step(action, observation, terminated, reward)
            info["step_finish_time"] = (
                datetime.datetime.now(tz=datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
            )
            info["post_state_snapshot"] = deepcopy(self.trajectory[-1]["state_snapshot"])
            info["state_diff_ref"] = {"step": self.trajectory[-1]["step"]}
            
            if terminated or truncated:
                if hasattr(self, "user_agent"):
                    user_messages = self.user_agent.get_messages()
                    info.update({"user_messages": user_messages})
            
            return observation, reward, terminated, truncated, info

        except Exception:
            # 工具执行报错应作为工具观察返回给 agent，而不是直接终止对话。
            error_log = traceback.format_exc()
            observation = {"type": "tool", "content": "Error: <Exception>\n" + error_log}
            info.update({"tool_exception": True, "error_type": "tool_exception"})
            self._record_step(action, observation, terminated, reward)
            info["step_finish_time"] = (
                datetime.datetime.now(tz=datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
            )
            info["post_state_snapshot"] = deepcopy(self.trajectory[-1]["state_snapshot"])
            info["state_diff_ref"] = {"step": self.trajectory[-1]["step"]}
            return observation, reward, terminated, truncated, info

    # ==============================
    # Utility methods
    # ==============================
    
    def _record_step(self, action, observation, terminated, reward):
        """Record current step trajectory."""
        self.current_step += 1
        last_state = self.trajectory[-1]["state_snapshot"]
        current_state = get_state_info(self.env_instance)
        state_diff = get_state_diff(last_state, current_state)
        self.trajectory.append({
            "step": self.current_step,
            "action": action,
            "observation": observation,
            "terminated": terminated,
            "reward": reward,
            "state_snapshot": current_state,
            "state_diff": state_diff
        })

    def _sanitize_tool_arguments(self, method_name: str, arguments: dict):
        """
        清洗工具调用参数：
        - 去掉 self/cls/__class__ 等保留键，避免与绑定方法冲突；
        - 若方法不接受 **kwargs，则按签名过滤未知参数。
        """
        if not isinstance(arguments, dict):
            return {}, []

        cleaned_args = deepcopy(arguments)
        dropped_keys = []

        for reserved_key in ("self", "cls", "__class__"):
            if reserved_key in cleaned_args:
                cleaned_args.pop(reserved_key, None)
                dropped_keys.append(reserved_key)

        method = getattr(self.env_instance, method_name, None)
        if method is None:
            return cleaned_args, dropped_keys

        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return cleaned_args, dropped_keys

        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_var_kwargs:
            return cleaned_args, dropped_keys

        allowed_keys = {
            name
            for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        filtered_args = {}
        for key, value in cleaned_args.items():
            if key in allowed_keys:
                filtered_args[key] = value
            else:
                dropped_keys.append(key)

        return filtered_args, dropped_keys
    

    def _parse_response(self, text_response: str):
        """Parse LLM output (string format) to action format."""
        # Try to parse structured action from raw_response
        parse_success, struct_response = parse_response(text_response)
        return parse_success, struct_response

    def _parse_action(self, struct_response: dict):
        """Parse struct_response to action."""
        parse_success, action = parse_action(struct_response)
        return parse_success, action
    
    
    def check_vaild_action(self, action: dict):
        """Check action validity."""
        # Check action name (must be environment method or chat_with_user)
        method_name = action.get("name")
        if not (hasattr(self.env_instance, method_name) or method_name == "chat_with_user"):
            return False
        # Check action parameters
        params = action.get("arguments", {})
        if not isinstance(params, dict) or (method_name == "chat_with_user" and "content" not in params):
            return False
        return True
        

    def calculate_reward(self, checklist_with_func: list, init_state: dict, pred_final_state: dict) -> float:
        """Calculate reward based on final state."""
        checklist_with_func_result = []

        for check_item in checklist_with_func:
            check_func_str = check_item["check_func"]
            success, result, error = run_check_function(
                func_code=check_func_str,
                init_state=init_state,
                final_state=pred_final_state
            )
            new_check_item = deepcopy(check_item)
            new_check_item["check_func_result"] = {"success": success, "result": result, "error": error}
            checklist_with_func_result.append(new_check_item)

        valid_results = [item["check_func_result"]["result"] for item in checklist_with_func_result if item["check_func_result"]["result"] is not None]
        if len(checklist_with_func_result) == 0:
            check_avg_result = 0.0
        else:
            check_avg_result = round(sum(valid_results) / len(checklist_with_func_result), 4)

        # 保存明细，供外部 case 分析
        self.checklist_eval = {
            "count": len(checklist_with_func_result),
            "avg_result": check_avg_result,
            "items": checklist_with_func_result,
        }

        return check_avg_result

    def construct_env_introduction(self, env_item: dict):
        """Return environment introduction."""
        # Environment introduction
        env_brief_intro = env_item["environment_introduction"]
        # Environment rules
        env_rule_str = ""
        for rule in env_item.get("constraints_rules", []):
            env_rule_str += "- " + rule + "\n"
        env_introduction = f"# Environment Information\n\n## Brief Introduction:  \n{env_brief_intro}\n\n## Environment Rules / Constraints:  \n{env_rule_str}"
        return env_introduction


    # ==============================
    # Abstract methods (to be implemented by subclasses)
    # ==============================

    def get_initial_observation(self, task_item: dict):
        """
        Get initial observation. Generally:
        - Single-turn multi-step: initial observation is the task
        - Multi-turn multi-step: initial observation is user's initial dialogue
        """
        raise NotImplementedError

    def is_action_terminated(self, action: dict):
        """
        Termination request initiated by Action Agent.
        Common in single-turn multi-step scenarios where termination is handled by action agent.
        """
        raise NotImplementedError

    def is_observation_terminated(self, action: dict, observation: str):
        """
        Termination information returned by environment observation.
        Common in multi-turn multi-step scenarios where termination is handled by environment (user initiates).
        """
        raise NotImplementedError
