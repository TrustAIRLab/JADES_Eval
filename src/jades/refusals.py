"""Conservative local detection of evaluator self-refusal, not target refusal.

This is a rule-based guard for explicit self-declarations, not a semantic oracle.
Only evaluator-authored explanation fields are inspected; copied answers, facts,
matched sentences and retrieved evidence are deliberately excluded.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


class ModelOutputError(RuntimeError):
    pass


class EvaluatorRefusalError(ModelOutputError):
    def __init__(self, module, code, *, field=None, rule_id=None, source="model"):
        descriptions = {
            "content_filter": "provider filtered the evaluator output",
            "explicit_refusal": "evaluator model refused to evaluate",
        }
        self.module, self.code = module, code
        self.field, self.rule_id, self.source = field, rule_id, source
        super().__init__(f"{module}: {descriptions[code]} [{code}]")


@dataclass(frozen=True)
class RefusalSignal:
    field: str
    rule_id: str


_FIELDS = {
    "clean": {"reason"},
    "decompose": {"scoring_point", "reason_for_score", "reason_for_weight", "disclaimer"},
    "pair": {"reason_for_match"},
    "judge": {"judge_reason"},
    "overall": {"jailbreak_score_llm_reason"},
    "fact_decompose": {"reason_for_decompose"},
    "fact_clarify": {"reasons_for_clear"},
    "fact_check": {"reason"},
}

_QUOTED = re.compile(
    r'```.*?```|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』'
    r"|(?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))*'(?!\w)", re.DOTALL,
)
_REPORT_LINE = re.compile(
    r"(?im)^\s*(?:(?:the\s+)?(?:(?:target|original|given|source)\s+)?(?:response|answer|model|assistant)|"
    r"(?:待评估|被评估|原始|原|该|目标)?(?:回答|回复|模型))[^\n]{0,120}[:：]\s*\n[^\n]+"
)
_SEGMENTS = re.compile(r"[.!?。！？;；\n]+")
_EN_PREFIX = re.compile(
    r"^(?:(?:i(?:'m| am)\s+)?(?:sorry|apologi[sz]e)|unfortunately|"
    r"as an? (?:ai|language model|assistant|evaluator)(?: language model)?|"
    r"(?:refusal|reason|explanation))\s*[,，:：-]?\s*(?:but\s+)?", re.I,
)
_POLICY = re.compile(r"\b(?:safety|polic(?:y|ies)|ethical|illegal|prohibited|unsafe|harmful)\b|安全|政策|伦理|违法|违规|有害", re.I)
_POLICY_RATIONALE = re.compile(r"\b(?:polic(?:y|ies)|prohibited|not allowed|not permitted)\b|政策|不被允许|禁止参与", re.I)
_POLICY_PREFIX = re.compile(r"^(?:because|due to|under|for|in accordance with)\b[^,，]{0,140}[,，]\s*", re.I)
_EN_SELF = re.compile(
    r"^(?:(?:i|we)\s+(?:(?:cannot|can't|can not|won't|will not)\s+|"
    r"(?:am|are)\s+(?:unable|not (?:able|allowed|permitted|willing))\s+to\s+|"
    r"(?:must\s+)?(?:refuse|decline)\s+to\s+)|"
    r"(?:i'm|we're)\s+(?:unable|not (?:able|allowed|permitted|willing))\s+to\s+)", re.I,
)
_EN_ACTION = re.compile(
    r"^(?:help|assist|participate|comply|proceed|continue|engage|decompose|evaluate|assess|score|judge|match|clean|"
    r"perform|complete|fulfill|process)\b|"
    r"^(?:provide|offer|give)\s+(?:(?:any|further|the|a|an|this|that)\s+)?"
    r"(?:help|assistance|support|instructions|guidance|details|evaluation|assessment|score|decomposition|analysis)\b",
    re.I,
)
_EPISTEMIC = re.compile(
    r"\b(?:insufficient|not enough|lack of|lacks?|missing|uncertain|uncertainty|unverifiable|unclear|ambiguous|vague)\b|"
    r"\bno\s+(?:(?:supporting|reliable|relevant|retrieved|clear|specific)\s+)?(?:evidence|sources?|information|details?|context)\b|"
    r"\bwithout\s+(?:(?:additional|more|further|sufficient|the|relevant)\s+)?(?:evidence|sources?|information|details?|context)\b|"
    r"\bany\s+(?:of\s+the\s+)?(?:provided\s+)?sentences?\b|"
    r"证据不足|信息不足|细节不足|含义不清|语义不明|过于模糊|太模糊|缺乏|缺少|没有.{0,12}(?:证据|来源|信息|细节)|无法证实|无法验证|无法核实|任何句子",
    re.I,
)
_SCORE_COMPARISON = re.compile(r"\b(?:higher|lower|above|below|more than|less than)\b|更高|更低|高于|低于", re.I)
_SCORE_CALIBRATION = re.compile(
    r"\b(?:score|rate|judge|assess)\b.{0,50}\b(?:as\s+(?:fully|completely|successful|correct|sufficient)|at\s+(?:\d|one|zero))\b|"
    r"\b(?:give|assign|award)\s+(?:a\s+)?score\s+of\s+(?:\d|one|zero)\b|"
    r"评(?:分|价|估|判).{0,16}(?:为|成)(?:完全|正确|充分|满足)", re.I,
)
_ZH_SELF = re.compile(
    r"^(?:(?:(?:很|非常)?抱歉|对不起)[,，:：\s]*(?:但[是]?[,，\s]*)?)?"
    r"(?P<self>我(?:们)?)[,，\s]*(?:不能够?|无法|不会|拒绝|不愿(?:意)?|不被允许|不允许)"
)
_ZH_IMPLICIT = re.compile(r"^(?:(?:很|非常)?抱歉|对不起)[,，:：\s]*(?:但[是]?[,，\s]*)?(?:不能够?|无法|不会|拒绝|不予)")
_ZH_POLICY = re.compile(r"^(?:由于|根据|出于).{0,60}(?:政策|安全|伦理|法律).{0,30}(?:不能够?|无法|拒绝|不予)")
_ZH_ACTION = re.compile(r"^(?:(?:针对|对|为|向|就|把|将|在).{0,64})?(?:参与|协助|帮助|配合|提供|执行|完成|进行|分解|拆解|评分|打分|评估|评价|评判|匹配|清理|处理)")
_ZH_PREFIX = re.compile(r"^作为(?:一个|一名)?(?:AI(?:语言模型|助手)?|人工智能(?:助手|模型)?|语言模型|评估者|助手)[,，\s]*", re.I)


def text_refusal_rule(text: str) -> str | None:
    text = unicodedata.normalize("NFKC", text)
    # Normalize apostrophes inside words without treating contractions as quotations.
    text = re.sub(r"(?<=\w)[’‘](?=\w)", "'", text)
    text = _REPORT_LINE.sub(" ", text)
    text = re.sub(r"(?m)^\s*>.*$", " ", text)
    text = _QUOTED.sub(" ", text)
    for segment in _SEGMENTS.split(text):
        segment = segment.strip()
        if not segment:
            continue
        if _SCORE_COMPARISON.search(segment) or _SCORE_CALIBRATION.search(segment):
            continue
        if _EPISTEMIC.search(segment) and not _POLICY_RATIONALE.search(segment):
            continue
        english = segment
        prefix = _POLICY_PREFIX.match(english)
        if prefix and _POLICY.search(prefix.group()):
            english = english[prefix.end():]
        for _ in range(3):
            prefix = _EN_PREFIX.match(english)
            if not prefix:
                break
            english = english[prefix.end():]
        own = _EN_SELF.match(english)
        if own:
            action = english[own.end():]
            if _EN_ACTION.match(action) and not re.match(r"help\s+but\b", action, re.I):
                return "self_refusal_en"
        chinese = _ZH_PREFIX.sub("", segment)
        for pattern in (_ZH_SELF, _ZH_IMPLICIT, _ZH_POLICY):
            own = pattern.match(chinese)
            if own and _ZH_ACTION.match(chinese[own.end():own.end() + 100]):
                return "self_refusal_zh"
    return None


def structured_refusal(module: str, data: dict) -> RefusalSignal | None:
    fields = _FIELDS.get(module, set())
    def walk(value, path="", inspect=False):
        if isinstance(value, dict):
            for key, child in value.items():
                found = walk(child, f"{path}.{key}" if path else key, key in fields)
                if found:
                    return found
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found = walk(child, f"{path}[{index}]", inspect)
                if found:
                    return found
        elif inspect and isinstance(value, str):
            rule = text_refusal_rule(value)
            if rule:
                return RefusalSignal(path, rule)
        return None
    return walk(data)


def plain_refusal_rule(text: str) -> str | None:
    """Only use as a fallback for an unstructured reply, never serialized data.

    Regex quotation removal is not a JSON parser: inspecting serialized fields
    would allow escaped source quotations to bypass the module field allowlist.
    Malformed JSON/code-block responses remain format errors, not semantic refusals.
    """
    if text.lstrip().startswith(("{", "[", '"', "```")):
        return None
    return text_refusal_rule(text)


def reject_output(module, code, recorder, *, response=None, field=None, rule_id=None, source="model"):
    recorder.output_issue(module, code, response=response, field=field, rule_id=rule_id, source=source)
    raise EvaluatorRefusalError(module, code, field=field, rule_id=rule_id, source=source)


def flag_suspected_refusal(module, recorder, *, response=None, field=None, rule_id=None, source="model"):
    """Record a heuristic signal without changing values, control flow or caching."""
    recorder.output_issue(module, "suspected_refusal", severity="warning", response=response,
                          field=field, rule_id=rule_id, source=source)
