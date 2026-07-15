"""
Conversational RL environment.
"""
from envscaler_env.utils.user_agent import UserAgent, user_system_prompt, user_system_prompt_en
from .base_env import EnvScalerBaseEnv

class EnvScalerConvRLEnv(EnvScalerBaseEnv):
    """Conversational RL environment that uses UserAgent for multi-turn dialogue."""
    
    def __init__(
        self,
        mode,
        user_model,
        provider,
        env_items_path=None,
        task_items_path=None,
        api_key=None,
        base_url=None,
        user_difficulty_config=None,
        verbose=True,
        lang="cn",
        enable_thinking=False,
    ):
        self.user_agent = UserAgent(
            system_prompt=user_system_prompt,
            model=user_model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            difficulty_config=user_difficulty_config,
            lang=lang,
            enable_thinking=enable_thinking,
        )
        super().__init__(mode=mode, env_items_path=env_items_path, task_items_path=task_items_path, verbose=verbose)

    def get_initial_observation(self, task_item: dict):
        """Get initial observation from user agent's first reply."""
        persona = task_item.get("persona_card_md") or task_item.get("persona") or ""
        return self.user_agent.get_init_reply(task=task_item["task"], persona=persona)

    def is_action_terminated(self, action: dict):
        """Conversation mode does not rely on action for termination."""
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """Terminate when user sends ###STOP### message."""
        return action.get("name") == "chat_with_user" and "###STOP###" in str(observation)
