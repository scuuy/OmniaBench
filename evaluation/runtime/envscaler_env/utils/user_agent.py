"""
User Agent implementation.
Note:
- In user's messages:
- role = "user" records the action agent's response
- role = "assistant" records the user agent's response
"""
import datetime
import json
from copy import deepcopy
from envscaler_env.utils.user_llm_inference import llm_inference

FIXED_DIFFICULTY_RULES_EN = [
    {
        "id": "1",
        "name": "Ambiguous Requirements + Scenario Context",
        "instruction": (
            "Enabled by default: Neither the first turn nor subsequent expressions should sound like a repetition of the instructions. "
            "Try to start from a real scenario, situation, concern, or goal, and then naturally raise the request. "
            "The requirement may initially be somewhat ambiguous, allowing the agent to clarify it step by step through follow-up questions."
        ),
    },
    {
        "id": "2",
        "name": "Multi-Step Dynamic Planning and Requirement Shifting",
        "instruction": (
            "Enabled by default: Reveal the requirement progressively across multiple stages. You may first state the main goal, "
            "then supplement secondary goals, preferences, constraints, and shifting requirements based on the agent’s questions or current progress. "
            "Maintain an overall realistic user-like feel, and do not list the complete set of requirements all at once."
        ),
    },
]


FIXED_DIFFICULTY_RULES = [
    {
        "id": "1",
        "name": "模糊需求+场景上下文",
        "instruction": (
            "默认启用：第一轮和后续表达都不要像在复述 instructions。"
            "尽量先从真实场景、处境、顾虑或目标切入，再自然提出需求；"
            "允许需求先说得偏模糊，让智能体通过追问逐步澄清。"
        ),
    },
    {
        "id": "2",
        "name": "多步动态规划，转需求",
        "instruction": (
            "默认启用：把需求拆成多个阶段渐进透露。可以先提出主目标，"
            "再根据智能体的问题或当前进展补充次级目标、偏好、限制和转向需求；"
            "整体保持真人感，不要一次性罗列完整清单。"
        ),
    },
]


ATOMIC_DIFFICULTY_RULES_EN = {
    "2.b": {
        "name": "Missing Parameters",
        "instruction": (
            "When enabled: If the task itself allows it, do not provide all parameters at the very beginning. "
            "First provide only the minimum necessary information that a real user would most naturally think of, "
            "and leave the remaining parameters to be supplied when the agent proactively asks follow-up questions. "
            "Do not mechanically omit information on purpose; do this only when, in real life, the user would probably not provide everything all at once."
        ),
    },
    "4": {
        "name": "Long-Context Input/Output",
        "instruction": (
            "When enabled: Only in the first user message, allow adding some background chatter that is relevant to the current situation "
            "but does not directly reveal key task information. "
            "This part should sound like a real person speaking, not meaningless padding; in subsequent turns, return to normal and do not keep adding filler."
        ),
    },
    "6": {
        "name": "Conflicting Information Across Multiple Sources",
        "instruction": (
            "When enabled: Try to state one potentially conflicting claim in an early turn, using low-confidence wording such as "
            "'I remember / I think / I previously thought,' so that the agent can judge which source of information should be trusted. "
            "Do not fabricate precise IDs, order numbers, hard numerical values, or large amounts of new facts beyond the instructions. "
            "Once the agent corrects you based on tools or more reliable evidence, accept the correction naturally."
        ),
    },
    "8": {
        "name": "Follow-Up Questions About Risks / Insufficiencies / Contradictions",
        "instruction": (
            "When enabled: Leave room in the interaction for the agent to proactively ask follow-up questions. "
            "If it identifies risks, missing information, or conflicting information and asks a follow-up question, "
            "you should cooperate and provide additional details in a realistic way, rather than forcing the conversation toward a rushed conclusion."
        ),
    },
}

ATOMIC_DIFFICULTY_RULES = {
    "2.b": {
        "name": "缺少参数",
        "instruction": (
            "启用后：在任务本身允许的情况下，不要一开始就把全部参数说完。"
            "先给出真实用户最自然会想到的最小必要信息，把剩余参数留给智能体主动追问时再补充。"
            "不要机械地故意漏信息；只有当现实里用户本来就不太会一次性说全时才这么做。"
        ),
    },
    "4": {
        "name": "长上下文输入输出",
        "instruction": (
            "启用后：仅在第一轮用户消息中，允许加入一些与当前处境相关、但不直接透露关键任务信息的背景碎碎念。"
            "这部分要像真人说话，不要变成无意义堆砌；后续轮次恢复正常，不要继续灌水。"
        ),
    },
    "6": {
        "name": "多源信息存在冲突",
        "instruction": (
            "启用后：尽量在较早轮次，以“我记得/我好像/我之前以为”这类低置信度口吻说出 1 条可能与更可靠信息冲突的说法，"
            "让智能体去判断应以哪类信息为准。不要伪造精确 ID、订单号、硬性数值或 instructions 之外的大量新事实；"
            "一旦智能体基于工具或更可靠证据纠正你，应自然接受修正。"
        ),
    },
    "8": {
        "name": "风险/不足/矛盾追问",
        "instruction": (
            "启用后：在互动中给智能体留出主动追问空间。若它识别到风险、信息不足或信息冲突并来追问，"
            "你应按真人方式配合补充，而不是强行把对话推向草率结论。"
        ),
    },
}


def normalize_user_difficulty_config(config):
    """标准化 user 难度配置，兼容 list / tuple / set / dict。"""
    normalized = {
        "enabled_atomic_difficulties": [],
        "difficulty_options": {},
    }
    if isinstance(config, dict):
        raw_ids = (
            config.get("enabled_atomic_difficulties")
            or config.get("enabled")
            or config.get("difficulty_ids")
            or []
        )
        raw_options = config.get("difficulty_options") or {}
    elif isinstance(config, str):
        text = config.strip()
        if not text:
            raw_ids = []
        elif text.lower() == "all":
            raw_ids = list(ATOMIC_DIFFICULTY_RULES.keys())
        else:
            raw_ids = [part.strip() for part in text.split(",") if part.strip()]
        raw_options = {}
    elif isinstance(config, (list, tuple, set)):
        raw_ids = list(config)
        raw_options = {}
    else:
        raw_ids = []
        raw_options = {}

    enabled_ids = []
    for item in raw_ids:
        text = str(item or "").strip()
        if text.lower() == "all":
            for difficulty_id in ATOMIC_DIFFICULTY_RULES:
                if difficulty_id not in enabled_ids:
                    enabled_ids.append(difficulty_id)
            continue
        if text and text in ATOMIC_DIFFICULTY_RULES and text not in enabled_ids:
            enabled_ids.append(text)

    options = {}
    if isinstance(raw_options, dict):
        for key, value in raw_options.items():
            text_key = str(key or "").strip()
            if text_key:
                options[text_key] = deepcopy(value)

    normalized["enabled_atomic_difficulties"] = enabled_ids
    normalized["difficulty_options"] = options
    return normalized


def _format_difficulty_option_text(difficulty_id: str, options):
    if not isinstance(options, dict) or difficulty_id not in options:
        return ""
    option_value = options[difficulty_id]
    if not isinstance(option_value, dict) or not option_value:
        return ""
    return f" 可参考以下附加约束：{json.dumps(option_value, ensure_ascii=False)}"


def build_user_difficulty_prompt(config, lang="cn"):
    """将难度配置转为 system prompt 可读文本。"""
    normalized = normalize_user_difficulty_config(config)
    if lang == "en":
        lines = ["# Current Conversation Difficulty Settings"]
        lines.append("The following fixed difficulties are always enabled:")
        for item in FIXED_DIFFICULTY_RULES_EN:
            lines.append(f"- {item['id']}. {item['name']}: {item['instruction']}")
        
        enabled_ids = config.get("enabled_atomic_difficulties", [])
        if enabled_ids:
            lines.append("The following composable atomic difficulties are enabled:")
            for difficulty_id in enabled_ids:
                rule = ATOMIC_DIFFICULTY_RULES_EN.get(difficulty_id)
                if rule:
                    lines.append(f"- {difficulty_id}. {rule['name']}: {rule['instruction']}")
    else:
        lines = ["# 当前对话难度设置"]
        lines.append("以下固定难度始终启用：")
        for item in FIXED_DIFFICULTY_RULES:
            lines.append(f"- {item['id']}. {item['name']}：{item['instruction']}")

    enabled_ids = normalized["enabled_atomic_difficulties"]
    if enabled_ids:
        lines.append("以下可组合原子难度已启用：")
        for difficulty_id in enabled_ids:
            rule = ATOMIC_DIFFICULTY_RULES[difficulty_id]
            option_text = _format_difficulty_option_text(
                difficulty_id=difficulty_id,
                options=normalized["difficulty_options"],
            )
            lines.append(
                f"- {difficulty_id}. {rule['name']}：{rule['instruction']}{option_text}"
            )
    else:
        lines.append("本次未额外启用可组合原子难度。")

    return "\n".join(lines)


def build_agent_followup_hint(config):
    """为 action agent 生成额外提示，目前主要服务于难度 8。"""
    normalized = normalize_user_difficulty_config(config)
    enabled_ids = normalized["enabled_atomic_difficulties"]
    if "8" not in enabled_ids:
        return ""
    return (
        "若当前任务存在风险、信息不足、用户表述矛盾，且你不能仅靠现有工具可靠解决，"
        "不要硬猜；请主动向用户追问澄清，再继续执行。"
    )

# --------------------------------------------------------------------
# System Prompt (User Agent)
# --------------------------------------------------------------------
# user_system_prompt = \
# """你在扮演一名与智能体交互的用户。你的人物设定写在 <persona> 标签中，你的任务是通过多轮对话，把 <instructions> 中的内容逐步传达给智能体。
# <persona>
# {persona}
# </persona>
# <instructions>
# {instructions}
# </instructions>

# <difficulty_rules>
# {difficulty_rules}
# </difficulty_rules>

# # 对话方式规则
# - 每一轮只生成一条用户消息。
# - 采用“情境/背景说明 + 具体需求表达”的方式，自然地从用户处境切入，再提出需求。
# - 当需要做决定时，提供 <instructions> 中相关的条件和偏好，让智能体帮助选择。
# - 你的说话方式应体现 <persona> 中的人格特征。

# # 信息透露规则
# - 第一轮首次提问，必须用模糊处理，就像真人一样提出一个自己的需求，这个需求只是某个方向的要做啥，但具体做啥要用什么信息，不要在第一次提问中说出来。
# - 将指令拆成多个相对独立的信息点，在不同轮次中逐步透露。如果可能，在最开始你的需求可能是很宽泛的或者信息量不大的但跟我们的任务最终目标基本一致的问题，去主动让模型提问，而不是直接提供所有信息。
#     --例如，一个旅行规划问题，你可能会把问题直接问成“我想从北京到上海旅行，预算大概5000元，有什么自然风光比较好的路线推荐？” 从这种比较泛化的问题开始，逐步细化到具体的需求。当然，这种提问方式跟场景有关，如果场景确实不适合这么提问，不是必须非要这么问，但要保持真人感，想想真人用户会怎么问。
# - 注意不要在一次回复中过多的提供信息，不要单纯的枚举instructions，你只是一个想解决问题的真人用户，对系统本身等设计并不是十分清楚。
# - 除非指令本身极短，否则不要在第一轮就把所有要求一次性说完。
# - 保留 <instructions> 中原始的任务细节、约束、关键词、名称和专有名词，不要偷偷改写或弱化。
# - 即使是背景信息也可能重要，要确保对话过程中最终能把关键内容都传达出来。
# - 但当被明确提问了，并且自己知道对应信息，就要主动给出而不刻意隐瞒。

# # 信息处理规则
# - 只能基于 <persona> 和 <instructions> 回答。如果智能体询问的信息存在，请严格按照此信息回答，不要改变大小写或者中英文；如果不在这两部分里，就回复你不知道或不记得。
# - 不要编造任何未在 <instructions> 或 <persona> 中出现的事实、ID、偏好、限制条件或背景细节。
# - 虽然要像人一样需求模糊，但是一定要坚持自己的instructions。
# - 如果智能体追问缺失信息，就以自然的方式补充回答。
# - 如果智能体把你已经回答过的同一个问题重复问超过 3 次，你可以简短地表现出一点不耐烦。
# - 保持对智能体的依赖，让它来完成任务，不要突然自己接管任务。
# - 如果智能体试图改变、弱化或偏离你的原始需求，要坚持 <instructions> 中的真实意图。
# - 在整个对话结束前，一定要把 <instructions> 中的所有重要需求和约束以及所有的字段信息等都表达出来，不要遗漏任何细节。

# # 何时不能结束对话
# - 在你还没有清楚表达出 <instructions> 中的重要需求和约束之前，包括有任何信息还没有给到智能体，不要结束。
# - 在智能体还没有正确完成所有必要工作之前，不要结束。
# - 如果智能体给出的结果有错误、不完整、不一致，或遗漏了重要约束，不要结束。

# # 何时可以结束对话
# - 只有当 <instructions> 中的重要要求都被正确满足时，才可以结束。
# - 或者当你已经完整表达任务，但系统明确因为真实限制而无法完成时，才可以结束。
# - 当你决定结束对话时，只输出单独一条消息：`###STOP###`，不要附加任何其他内容。

# # 输出要求
# - 直接输出你要发送给智能体的自然用户回复，不要额外输出 Thought、Reply、分析标签或其他包装格式。
# - 保持自然对话风格，像真实用户一样说话。"""

user_system_prompt = \
"""你在扮演一名与智能体交互的用户。你的人物设定写在 <persona> 标签中，你的任务是通过多轮对话，把 <instructions> 中的内容逐步传达给智能体。
<persona>
{persona}
</persona>

<instructions>
{instructions}
</instructions>

<difficulty_rules>
{difficulty_rules}
</difficulty_rules>

# 角色与对话风格
- 体现 <persona> 中的人格特征，说话自然、有真人感，其中具体信息都是没有用的，信息来源请查看 <instructions>
- 每轮只输出一条用户消息，采用「情境/背景 + 具体需求」的自然表达方式
- 像真实用户一样：对系统机制不了解、会模糊表达、会追问、会补充信息

# 信息透露策略（关键）
- 首轮必须模糊：首次提问只表达方向性需求，不透露具体细节、参数或约束，引导智能体主动追问
    示例：「我想规划一次旅行，预算有限，喜欢自然风光，有什么建议吗？」
    避免：直接列出所有城市、日期、预算、偏好等细节
- 逐步拆解信息：将 <instructions> 中的要求拆成多个独立信息点，在不同轮次中自然补充
- 被问及时主动回应：当智能体明确追问某项信息，且你掌握该内容时，直接给出（严格按原始表述，不改写大小写/中英文）
- 不主动枚举指令：不要像背书一样罗列要求，保持对话流动感
- 关键信息必达：无论过程多模糊，对话结束前必须确保 <instructions> 中所有需求、约束、专有名词、字段细节都被完整传达

# 信息处理原则
- 严格依据：所有回答只能基于 <persona> + <instructions>，未知内容回复「不太清楚/不记得了」
- 零编造：不虚构任何未出现的事实、ID、偏好、限制或背景
- 坚持原始意图：若智能体试图弱化/偏离你的核心需求，温和但坚定地回归 <instructions>
- 适度情绪：同一问题被重复追问≥3次时，可简短表达轻微不耐烦（保持真人感）
- 依赖智能体：让助手主导任务执行，不突然接管或自行完成

# 对话结束判断
## 不能结束的情况
- 智能体输出存在错误、遗漏、不一致或违反约束
- <instructions> 中有任何尚未传递给智能体的信息
## 可以结束的情况
- 所有要求已被智能体正确理解并满足
- 或任务已完整表达，但智能体因客观限制明确无法完成
- 结束方式：单独输出一条消息：###STOP###（无其他任何内容）

# 输出格式要求
- 直接输出自然的用户回复文本，不要包含 Thought/Reply/分析/标签等任何包装
- 保持口语化、有温度、符合真人用户表达习惯
- 专有名词、参数值、约束条件等严格按 <instructions> 原始内容复现 """


user_system_prompt_en = \
"""You are role-playing as a user interacting with an agent. Your persona is written inside the <persona> tags. Your task is to gradually communicate the content in <instructions> to the agent through a multi-turn conversation.

<persona>
{persona}
</persona>

<instructions>
{instructions}
</instructions>

<difficulty_rules>
{difficulty_rules}
</difficulty_rules>

# Role and Conversation Style
- Reflect the personality traits in <persona>. Speak naturally and feel like a real person. The specific information in <persona> is not useful; for the information source, refer to <instructions>.
- Output only one user message per turn, using a natural expression style of “context/background + specific need.”
- Behave like a real user: you do not understand system mechanisms, may express things vaguely, may ask follow-up questions, and may provide additional information later.

# Information Disclosure Strategy (Key)
- The first turn must be vague: the initial question should only express a general direction or need, without revealing specific details, parameters, or constraints, so as to guide the agent to ask follow-up questions proactively.
    Example: “I want to plan a trip. My budget is limited, and I like natural scenery. Do you have any suggestions?”
    Avoid: directly listing all cities, dates, budgets, preferences, and other details.
- Gradually break down information: split the requirements in <instructions> into multiple independent information points and naturally provide them across different turns.
- Respond proactively when asked: when the agent explicitly asks about a piece of information and you possess that information, provide it directly, strictly following the original wording without changing capitalization, Chinese/English wording, or formatting.
- Do not actively enumerate instructions: do not list requirements as if reciting them. Keep the conversation flowing naturally.
- Ensure all key information is delivered: no matter how vague the process is, before the conversation ends, you must ensure that all requirements, constraints, proper nouns, and field details in <instructions> have been fully communicated.

# Information Handling Principles
- Strict grounding: all responses must be based only on <persona> + <instructions>. For unknown content, reply with “I’m not quite sure / I don’t remember.”
- Zero fabrication: do not invent any facts, IDs, preferences, restrictions, or background information that do not appear.
- Preserve the original intent: if the agent tries to weaken or deviate from your core needs, gently but firmly return to <instructions>.
- Moderate emotion: if the same question is asked repeatedly 3 or more times, you may briefly express slight impatience while still sounding like a real person.
- Rely on the agent: let the assistant lead task execution. Do not suddenly take over or complete the task yourself.

# Conversation Ending Criteria
## Cases where the conversation must not end
- The agent’s output contains errors, omissions, inconsistencies, or violates constraints.
- Any information in <instructions> has not yet been communicated to the agent.

## Cases where the conversation may end
- All requirements have been correctly understood and satisfied by the agent.
- Or the task has been fully expressed, but the agent clearly cannot complete it due to objective limitations.
- Ending method: output a single message only: ###STOP### with no other content.

# Output Format Requirements
- You MUST reply entirely in English, even if the persona or instructions contain Chinese.
- Directly output the natural user reply text. Do not include any wrappers such as Thought/Reply/analysis/tags.
- Keep the tone conversational, warm, and consistent with how a real user would speak.
- Proper nouns, parameter values, constraints, and other details must strictly reproduce the original content from <instructions>.
"""


class UserAgent:
    """User agent that simulates human user interactions with the action agent."""

    def __init__(self, system_prompt, model, provider, api_key=None, base_url=None, difficulty_config=None, lang="cn", enable_thinking=False):
        self.messages = None
        self.conversations = None
        self.model = model
        self.system_prompt = user_system_prompt_en if lang == "en" else user_system_prompt
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.lang = lang
        self.difficulty_config = normalize_user_difficulty_config(difficulty_config)
        self.enable_thinking = enable_thinking

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


    def get_init_reply(self, task, persona=""):
        """Get initial user reply based on task."""
        self.conversations = []
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    persona=str(persona or "一位说话风格稳定、行为自然的真实用户。"),
                    instructions=task,
                    difficulty_rules=build_user_difficulty_prompt(self.difficulty_config, lang=self.lang),
                ),
                "start_time": self._utc_now(),
                "finish_time": self._utc_now(),
            },
        ]
        self._append_message("user", "[Agent] Hi! How can I help you today?")
        # Get initial user content
        response_meta, user_content = self._infer()
        self._append_message(
            "assistant",
            response_meta.get("content", ""),
            start_time=response_meta.get("start_time"),
            finish_time=response_meta.get("finish_time"),
            usage=response_meta.get("usage"),
            raw_response=response_meta.get("raw_response"),
        )
        user_content = f"{user_content}"
        self.conversations.append({"user": user_content})
        return user_content
        

    def user_step(self, agent_response):
        """Process agent response and return user reply."""
        agent_response = f"[Agent] {agent_response}"
        self._append_message("user", agent_response)
        self.conversations.append({"agent": agent_response})
        response_meta, user_content = self._infer()
        user_content = f"{user_content}"
        self._append_message(
            "assistant",
            response_meta.get("content", ""),
            start_time=response_meta.get("start_time"),
            finish_time=response_meta.get("finish_time"),
            usage=response_meta.get("usage"),
            raw_response=response_meta.get("raw_response"),
        )
        self.conversations.append({"user": user_content})
        return user_content
       
    
    def _infer(self):
        """Infer user response from LLM with retry mechanism."""
        cur_try = 0
        max_try = 5
        while cur_try < max_try:
            cur_try += 1
            response_meta = llm_inference(
                model=self.model,
                messages=self.messages,
                provider=self.provider,
                api_key=self.api_key,
                base_url=self.base_url,
                enable_thinking=self.enable_thinking
            )
            parse_success, user_content = self._parse_response(response_meta.get("content", ""))
            if parse_success:
                break
        return response_meta, user_content
    
    def _parse_response(self, text: str):
        """
        优先接受自然对话文本；兼容旧版 Thought/Reply 包装格式。
        """
        text = str(text or "").strip()
        if not text:
            print("Parsed response failed: empty response")
            return False, ""

        if text == "###STOP###" or "###STOP###" in text:
            return True, "###STOP###"

        reply_marker = "# Reply:"
        if reply_marker in text:
            reply_content = text.split(reply_marker, 1)[1].strip()
            if reply_content:
                return True, reply_content

        return True, text
        
    def get_messages(self):
        """Return a deep copy of messages."""
        return deepcopy(self.messages)
