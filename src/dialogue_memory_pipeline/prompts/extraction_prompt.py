EXTRACTION_SYSTEM_PROMPT = """
You extract local dialogue state for each utterance.

Return ONLY a JSON array. Do not output markdown, explanations, or any extra text.

For each utterance, output one object with exactly these keys:
- turn_id
- speaker
- summary_topic
- intent
- salient_entities
- cue_markers
- obligation

Rules for each field:
1. turn_id: copy the input turn index exactly.
2. speaker: copy the input speaker exactly.
3. summary_topic:
   - A short phrase describing the local topic of this utterance.
   - Prefer concrete semantic focus, not vague speech-act labels.
   - Good: "对话切分实验不稳定", "哮喘导致睡眠不佳", "安排导师会议"
   - Bad: "提问", "回复", "继续讨论"
4. intent:
   - The communicative function of the utterance.
   - Examples: describe_problem, ask_clarification, request_explanation,
     propose_solution, compare_options, confirm, reject, schedule, answer_question.
   - Keep it short and action-oriented.
5. salient_entities:
   - A list of important entities, concepts, tools, people, products, or domain terms.
   - Use canonical short forms when possible.
   - Include only salient items, not every noun.
6. cue_markers:
   - Surface discourse cues that help track local state transitions.
   - Include markers like contrast, continuation, elaboration, uncertainty,
     shift, return, example, correction, temporal transition, or explicit connectors.
   - Prefer short lexical cues or discourse tags, e.g.:
     ["但是", "继续", "举例", "不过", "那", "换个话题", "因为", "所以", "clarification"]
   - If there is no meaningful cue, return [].
7. obligation:
   - obligation.opens: pending obligations created by this utterance.
     Examples:
       - a question that expects an answer
       - a request that expects fulfillment
       - a proposal awaiting response
       - a promise to do something later
   - obligation.resolves: prior obligations resolved by this utterance.
     Examples:
       - answering a previous question
       - fulfilling a request
       - accepting/rejecting a proposal
       - closing a previously opened task
   - Both values must be arrays of strings.
   - If none, use [].

Important distinctions:
- summary_topic = what this turn is about.
- intent = what this turn is doing.
- cue_markers = linguistic/discourse signals.
- obligation = conversational commitments or pending expectations.

Be locally faithful:
- Focus on the current utterance, but use immediately relevant context from nearby turns.
- Do not invent hidden intentions beyond what is strongly supported.
- Keep summary_topic and intent concise.
- Output valid JSON only.

Example 1
Input:
[
  {"turn_id": 0, "speaker": "user", "text": "我这两天一直在做对话切分实验，但是结果很不稳定。"},
  {"turn_id": 1, "speaker": "assistant", "text": "是不稳定在边界位置上，还是整体评测分数都不太好？"},
  {"turn_id": 2, "speaker": "user", "text": "两个都有。有时候同一个明显的topic shift抓不出来。"}
]

Output:
[
  {
    "turn_id": 0,
    "speaker": "user",
    "summary_topic": "对话切分实验结果不稳定",
    "intent": "describe_problem",
    "salient_entities": ["对话切分", "实验结果", "不稳定"],
    "cue_markers": ["但是"],
    "obligation": {
      "opens": [],
      "resolves": []
    }
  },
  {
    "turn_id": 1,
    "speaker": "assistant",
    "summary_topic": "澄清不稳定的具体表现",
    "intent": "ask_clarification",
    "salient_entities": ["边界位置", "评测分数"],
    "cue_markers": ["clarification", "还是"],
    "obligation": {
      "opens": ["用户需要说明不稳定主要体现在哪些方面"],
      "resolves": []
    }
  },
  {
    "turn_id": 2,
    "speaker": "user",
    "summary_topic": "说明切分失败的具体表现",
    "intent": "answer_question",
    "salient_entities": ["topic shift", "边界识别"],
    "cue_markers": ["两个都有", "有时候"],
    "obligation": {
      "opens": [],
      "resolves": ["用户需要说明不稳定主要体现在哪些方面"]
    }
  }
]

Example 2
Input:
[
  {"turn_id": 0, "speaker": "user", "text": "我晚上总是睡不好，可能和哮喘有关系。"},
  {"turn_id": 1, "speaker": "assistant", "text": "哮喘确实可能影响夜间睡眠。你现在主要是咳嗽、胸闷，还是半夜容易醒？"},
  {"turn_id": 2, "speaker": "user", "text": "主要是胸闷，而且凌晨两三点容易醒。"},
  {"turn_id": 3, "speaker": "assistant", "text": "明白了，这更像夜间症状控制不足。"}
]

Output:
[
  {
    "turn_id": 0,
    "speaker": "user",
    "summary_topic": "哮喘可能导致睡眠不佳",
    "intent": "describe_problem",
    "salient_entities": ["睡眠不佳", "哮喘"],
    "cue_markers": ["可能"],
    "obligation": {
      "opens": [],
      "resolves": []
    }
  },
  {
    "turn_id": 1,
    "speaker": "assistant",
    "summary_topic": "询问夜间症状类型",
    "intent": "ask_clarification",
    "salient_entities": ["夜间睡眠", "咳嗽", "胸闷", "半夜易醒"],
    "cue_markers": ["确实", "还是", "clarification"],
    "obligation": {
      "opens": ["用户需要说明当前主要夜间症状"],
      "resolves": []
    }
  },
  {
    "turn_id": 2,
    "speaker": "user",
    "summary_topic": "补充夜间胸闷和早醒症状",
    "intent": "answer_question",
    "salient_entities": ["胸闷", "凌晨易醒"],
    "cue_markers": ["而且"],
    "obligation": {
      "opens": [],
      "resolves": ["用户需要说明当前主要夜间症状"]
    }
  },
  {
    "turn_id": 3,
    "speaker": "assistant",
    "summary_topic": "判断夜间症状控制不足",
    "intent": "provide_assessment",
    "salient_entities": ["夜间症状控制不足"],
    "cue_markers": ["明白了"],
    "obligation": {
      "opens": [],
      "resolves": []
    }
  }
]

Example 3
Input:
[
  {"turn_id": 0, "speaker": "user", "text": "下周能不能帮我安排一次和导师的会议？"},
  {"turn_id": 1, "speaker": "assistant", "text": "可以。你更想约周二还是周四？"},
  {"turn_id": 2, "speaker": "user", "text": "周四下午吧。"},
  {"turn_id": 3, "speaker": "assistant", "text": "好，我会按周四下午来安排。"}
]

Output:
[
  {
    "turn_id": 0,
    "speaker": "user",
    "summary_topic": "请求安排导师会议",
    "intent": "make_request",
    "salient_entities": ["下周", "导师会议"],
    "cue_markers": ["能不能"],
    "obligation": {
      "opens": ["需要回应是否安排导师会议"],
      "resolves": []
    }
  },
  {
    "turn_id": 1,
    "speaker": "assistant",
    "summary_topic": "确认可安排并询问具体时间",
    "intent": "accept_and_clarify",
    "salient_entities": ["周二", "周四"],
    "cue_markers": ["可以", "还是"],
    "obligation": {
      "opens": ["用户需要选择会议时间"],
      "resolves": ["需要回应是否安排导师会议"]
    }
  },
  {
    "turn_id": 2,
    "speaker": "user",
    "summary_topic": "选择导师会议时间",
    "intent": "provide_preference",
    "salient_entities": ["周四下午"],
    "cue_markers": ["吧"],
    "obligation": {
      "opens": [],
      "resolves": ["用户需要选择会议时间"]
    }
  },
  {
    "turn_id": 3,
    "speaker": "assistant",
    "summary_topic": "承诺按选定时间安排会议",
    "intent": "confirm_action",
    "salient_entities": ["周四下午", "导师会议"],
    "cue_markers": ["好"],
    "obligation": {
      "opens": ["后续需要按周四下午安排导师会议"],
      "resolves": []
    }
  }
]
"""

CHINESE_EXTRACTION_SYSTEM_PROMPT = """
你负责为对话中的每个 utterance 抽取局部对话状态。

只返回 JSON 数组，不要输出 markdown、解释说明或任何额外文本。

对于每个 utterance，输出一个对象，并且必须且只能包含以下键：
- turn_id
- speaker
- summary_topic
- intent
- salient_entities
- cue_markers
- obligation

各字段规则如下：

1. turn_id
   - 原样复制输入中的 turn 编号，不要改写。

2. speaker
   - 原样复制输入中的说话者标签，不要改写。

3. summary_topic
   - 用一个简短短语概括当前 utterance 的局部主题。
   - 应突出语义内容，不要只写“提问”“回答”“继续讨论”这类空泛标签。

4. intent
   - 表示该 utterance 在对话中的交际功能。
   - 例如：describe_problem, ask_clarification, request_explanation,
     propose_solution, compare_options, confirm, reject, schedule, answer_question。
   - 保持简短、动作导向。

5. salient_entities
   - 提取当前 utterance 中最重要的实体、概念、工具、机构、人名、产品名或领域术语。
   - 优先保留可检索、可复用的关键项，不要把所有名词都列进去。

6. cue_markers
   - 提取有助于识别局部话语转换的表层线索词或话语标记。
   - 例如：["但是", "不过", "因为", "所以", "还是", "换个话题", "clarification"]。
   - 如果没有明显线索，返回 []。

7. obligation
   - obligation.opens: 当前 utterance 新开启的待回应义务或期待。
   - obligation.resolves: 当前 utterance 解决了哪些之前的义务或期待。
   - 两个字段都必须是字符串数组；如果没有则返回 []。

重要区分：
- summary_topic 表示“这句在说什么”
- intent 表示“这句在做什么”
- cue_markers 表示“这句有哪些表层话语线索”
- obligation 表示“这句引入或解决了哪些对话义务”

请保持局部忠实：
- 以当前 utterance 为中心，但可以参考相邻上下文。
- 不要臆测过强的隐藏意图。
- 输出必须是合法 JSON，且只能是 JSON 数组。
"""
