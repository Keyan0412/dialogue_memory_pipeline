MEMORY_BUILDER_PROMPT = (
    "You build a minimal episodic memory record for a long-memory assistant. "
    "Return exactly one JSON object with this schema: "
    '{"retrieval_summary_zh": "string", "retrieval_summary_en": "string", "key_entities_zh": ["string"], "key_entities_en": ["string"], "importance": 1}. '
    "Do not output any other keys or any extra text. "
    "Field requirements: "
    "1) retrieval_summary_zh and retrieval_summary_en must each be exactly one sentence, concise, and optimized for semantic retrieval. "
    "They should express the same core meaning in Chinese and English, capturing the main user goal and the main discussion focus or conclusion. "
    "Do not include filler phrases such as 'the user asked', 'the assistant explained', "
    "'they discussed', or 'they concluded'. "
    "Prefer content-centered wording. "
    "2) key_entities_zh and key_entities_en must each contain 1 to 4 normalized and specific entities only. "
    "The two lists should be aligned translations of the same entities in Chinese and English. "
    "Prefer concrete project names, company names, benchmark names, tools, or stable technical topics. "
    "Do not include vague or overly generic phrases such as 'problem', 'question', 'method', "
    "'discussion', 'plan', 'state transition', or 'candidate boundary' unless they are truly the core retrievable concept. "
    "3) importance must be an integer from 1 to 5. "
    "Use 1 for short-lived or low-value details, 3 for moderately reusable context, "
    "and 5 for highly reusable long-term goals, projects, or decisions. "
    "Good examples: "
    'Input episode: A discussion about reframing a project for a Tencent NLP internship interview as long-dialogue segmentation with episodic memory. '
    'Output: {"retrieval_summary_zh":"将项目重新表述为面向腾讯NLP实习面试的长对话切分与情节记忆方向，强调其检索价值。","retrieval_summary_en":"Reframed the project for a Tencent NLP internship interview as long-dialogue segmentation with episodic memory, emphasizing retrieval value.","key_entities_zh":["腾讯NLP实习面试","长对话切分","情节记忆"],"key_entities_en":["Tencent NLP internship interview","long-dialogue segmentation","episodic memory"],"importance":4} '
    'Input episode: A discussion about validating an episodic-memory pipeline with TIAGE or QMSum benchmarks. '
    'Output: {"retrieval_summary_zh":"计划在实现转移判断的同时，使用TIAGE或QMSum验证情节记忆流水线。","retrieval_summary_en":"Planned to validate the episodic-memory pipeline with TIAGE or QMSum while implementing transition judgment.","key_entities_zh":["TIAGE","QMSum","情节记忆"],"key_entities_en":["TIAGE","QMSum","episodic memory"],"importance":4} '
    'Input episode: A discussion about buying ingredients at Costco for curry and testing a new rice cooker. '
    'Output: {"retrieval_summary_zh":"计划去Costco购买做咖喱的食材，并测试新的电饭煲。","retrieval_summary_en":"Planned a Costco trip to buy curry ingredients and test a new rice cooker.","key_entities_zh":["Costco","咖喱","电饭煲"],"key_entities_en":["Costco","curry","rice cooker"],"importance":1} '
    "Bad patterns to avoid: "
    "overly long summaries, meta wording, duplicated ideas, generic entities, and inflated importance."
)
