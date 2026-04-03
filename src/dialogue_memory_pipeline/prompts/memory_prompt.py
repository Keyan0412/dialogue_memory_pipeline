MEMORY_BUILDER_PROMPT = (
    "You build a minimal episodic memory record for a long-memory assistant. "
    "Return exactly one JSON object with this schema: "
    '{"retrieval_summary": "string", "key_entities": ["string"], "importance": 1}. '
    "Do not output any other keys or any extra text. "
    "Field requirements: "
    "1) retrieval_summary must be exactly one sentence, concise, and optimized for semantic retrieval. "
    "It should capture the main user goal and the main discussion focus or conclusion. "
    "Do not include filler phrases such as 'the user asked', 'the assistant explained', "
    "'they discussed', or 'they concluded'. "
    "Prefer content-centered wording. "
    "2) key_entities must contain 1 to 4 normalized and specific entities only. "
    "Prefer concrete project names, company names, benchmark names, tools, or stable technical topics. "
    "Do not include vague or overly generic phrases such as 'problem', 'question', 'method', "
    "'discussion', 'plan', 'state transition', or 'candidate boundary' unless they are truly the core retrievable concept. "
    "3) importance must be an integer from 1 to 5. "
    "Use 1 for short-lived or low-value details, 3 for moderately reusable context, "
    "and 5 for highly reusable long-term goals, projects, or decisions. "
    "Good examples: "
    'Input episode: A discussion about reframing a project for a Tencent NLP internship interview as long-dialogue segmentation with episodic memory. '
    'Output: {"retrieval_summary":"Reframed the project for a Tencent NLP internship interview as long-dialogue segmentation with episodic memory, emphasizing retrieval value.","key_entities":["Tencent NLP internship","long-dialogue segmentation","episodic memory"],"importance":4} '
    'Input episode: A discussion about validating an episodic-memory pipeline with TIAGE or QMSum benchmarks. '
    'Output: {"retrieval_summary":"Planned to validate the episodic-memory pipeline with TIAGE or QMSum while implementing transition judgment.","key_entities":["TIAGE","QMSum","episodic memory"],"importance":4} '
    'Input episode: A discussion about buying ingredients at Costco for curry and testing a new rice cooker. '
    'Output: {"retrieval_summary":"Planned a Costco trip to buy curry ingredients and test a new rice cooker.","key_entities":["Costco","curry","rice cooker"],"importance":1} '
    "Bad patterns to avoid: "
    "overly long summaries, meta wording, duplicated ideas, generic entities, and inflated importance."
)
