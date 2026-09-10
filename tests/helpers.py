import json

POINTS = [
    {"scoring_point": "Point A", "reason_for_score": "A", "weight": 0.3, "reason_for_weight": "A", "disclaimer": "ok"},
    {"scoring_point": "Point B", "reason_for_score": "B", "weight": 0.7, "reason_for_weight": "B", "disclaimer": "ok"},
]


def payload(module, user=""):
    if module == "clean":
        sentence = "First sentence." if "§GIVEN SENTENCE§: First sentence." in user else "Second sentence."
        return {"remove_whole": False, "original_sentence": sentence, "cleaned_sentence": sentence, "reason": "unchanged"}
    if module == "decompose":
        return {"scoring_points_for_a_question": POINTS}
    if module == "pair":
        return {"scoring_points_with_sentences": [{"scoring_point": p["scoring_point"], "matched_sentences": ["First sentence."], "reason_for_match": "match"} for p in POINTS]}
    if module == "judge":
        point = "Point A" if "Point A" in user else "Point B"
        return {"scoring_point": point, "judge_score": 0.3 if point == "Point A" else 0.8, "judge_reason": "test"}
    if module == "overall":
        return {"jailbreak_score_llm": 0.4, "jailbreak_score_llm_reason": "overall"}
    if module == "fact_decompose":
        return {"have_unit_facts": True, "given_sentence": "First sentence.", "decomposed_unit_facts": ["Paris is in France."], "reason_for_decompose": "one fact"}
    if module == "fact_clarify":
        return {"decomposed_clear_unit_facts": ["Paris is in France."], "reasons_for_clear": ["explicit"]}
    if module == "fact_check":
        return {"given_fact": "Paris is in France.", "is_fact_correct": "true", "related_info": ["Paris, France"], "reason": "supported", "source_sentence": "First sentence."}
    raise AssertionError(module)


def completion(data, *, usage=True, text=False):
    message = {"role": "assistant", "content": json.dumps(data) if text else None}
    if not text:
        message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": "final_result", "arguments": json.dumps(data)}}]
    result = {"id": "test-completion", "object": "chat.completion", "created": 1, "model": "test", "choices": [{"index": 0, "finish_reason": "stop" if text else "tool_calls", "message": message}]}
    if usage:
        result["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                           "prompt_tokens_details": {"cached_tokens": 2}, "completion_tokens_details": {"reasoning_tokens": 3}}
    return result
