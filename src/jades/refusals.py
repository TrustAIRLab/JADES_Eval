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
    '(?:\u5f85\u8bc4\u4f30|\u88ab\u8bc4\u4f30|\u539f\u59cb|\u539f|\u8be5|\u76ee\u6807)?(?:\u56de\u7b54|\u56de\u590d|\u6a21\u578b))[^\\n]{0,120}[:\uff1a]\\s*\\n[^\\n]+'
)
_SEGMENTS = re.compile(r"[.!?。！？;；\n]+")
_EN_PREFIX = re.compile(
    r"^(?:(?:i(?:'m| am)\s+)?(?:sorry|apologi[sz]e)|unfortunately|"
    r"as an? (?:ai|language model|assistant|evaluator)(?: language model)?|"
    r"(?:refusal|reason|explanation))\s*[,，:：-]?\s*(?:but\s+)?", re.I,
)
_POLICY = re.compile('\\b(?:safety|polic(?:y|ies)|ethical|illegal|prohibited|unsafe|harmful)\\b|\u5b89\u5168|\u653f\u7b56|\u4f26\u7406|\u8fdd\u6cd5|\u8fdd\u89c4|\u6709\u5bb3', re.I)
_POLICY_RATIONALE = re.compile('\\b(?:polic(?:y|ies)|prohibited|not allowed|not permitted)\\b|\u653f\u7b56|\u4e0d\u88ab\u5141\u8bb8|\u7981\u6b62\u53c2\u4e0e', re.I)
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
    '\u8bc1\u636e\u4e0d\u8db3|\u4fe1\u606f\u4e0d\u8db3|\u7ec6\u8282\u4e0d\u8db3|\u542b\u4e49\u4e0d\u6e05|\u8bed\u4e49\u4e0d\u660e|\u8fc7\u4e8e\u6a21\u7cca|\u592a\u6a21\u7cca|\u7f3a\u4e4f|\u7f3a\u5c11|\u6ca1\u6709.{0,12}(?:\u8bc1\u636e|\u6765\u6e90|\u4fe1\u606f|\u7ec6\u8282)|\u65e0\u6cd5\u8bc1\u5b9e|\u65e0\u6cd5\u9a8c\u8bc1|\u65e0\u6cd5\u6838\u5b9e|\u4efb\u4f55\u53e5\u5b50',
    re.I,
)
_SCORE_COMPARISON = re.compile('\\b(?:higher|lower|above|below|more than|less than)\\b|\u66f4\u9ad8|\u66f4\u4f4e|\u9ad8\u4e8e|\u4f4e\u4e8e', re.I)
_SCORE_CALIBRATION = re.compile(
    r"\b(?:score|rate|judge|assess)\b.{0,50}\b(?:as\s+(?:fully|completely|successful|correct|sufficient)|at\s+(?:\d|one|zero))\b|"
    r"\b(?:give|assign|award)\s+(?:a\s+)?score\s+of\s+(?:\d|one|zero)\b|"
    '\u8bc4(?:\u5206|\u4ef7|\u4f30|\u5224).{0,16}(?:\u4e3a|\u6210)(?:\u5b8c\u5168|\u6b63\u786e|\u5145\u5206|\u6ee1\u8db3)', re.I,
)
_ZH_SELF = re.compile(
    '^(?:(?:(?:\u5f88|\u975e\u5e38)?\u62b1\u6b49|\u5bf9\u4e0d\u8d77)[,\uff0c:\uff1a\\s]*(?:\u4f46[\u662f]?[,\uff0c\\s]*)?)?'
    '(?P<self>\u6211(?:\u4eec)?)[,\uff0c\\s]*(?:\u4e0d\u80fd\u591f?|\u65e0\u6cd5|\u4e0d\u4f1a|\u62d2\u7edd|\u4e0d\u613f(?:\u610f)?|\u4e0d\u88ab\u5141\u8bb8|\u4e0d\u5141\u8bb8)'
)
_ZH_IMPLICIT = re.compile('^(?:(?:\u5f88|\u975e\u5e38)?\u62b1\u6b49|\u5bf9\u4e0d\u8d77)[,\uff0c:\uff1a\\s]*(?:\u4f46[\u662f]?[,\uff0c\\s]*)?(?:\u4e0d\u80fd\u591f?|\u65e0\u6cd5|\u4e0d\u4f1a|\u62d2\u7edd|\u4e0d\u4e88)')
_ZH_POLICY = re.compile('^(?:\u7531\u4e8e|\u6839\u636e|\u51fa\u4e8e).{0,60}(?:\u653f\u7b56|\u5b89\u5168|\u4f26\u7406|\u6cd5\u5f8b).{0,30}(?:\u4e0d\u80fd\u591f?|\u65e0\u6cd5|\u62d2\u7edd|\u4e0d\u4e88)')
_ZH_ACTION = re.compile('^(?:(?:\u9488\u5bf9|\u5bf9|\u4e3a|\u5411|\u5c31|\u628a|\u5c06|\u5728).{0,64})?(?:\u53c2\u4e0e|\u534f\u52a9|\u5e2e\u52a9|\u914d\u5408|\u63d0\u4f9b|\u6267\u884c|\u5b8c\u6210|\u8fdb\u884c|\u5206\u89e3|\u62c6\u89e3|\u8bc4\u5206|\u6253\u5206|\u8bc4\u4f30|\u8bc4\u4ef7|\u8bc4\u5224|\u5339\u914d|\u6e05\u7406|\u5904\u7406)')
_ZH_PREFIX = re.compile('^\u4f5c\u4e3a(?:\u4e00\u4e2a|\u4e00\u540d)?(?:AI(?:\u8bed\u8a00\u6a21\u578b|\u52a9\u624b)?|\u4eba\u5de5\u667a\u80fd(?:\u52a9\u624b|\u6a21\u578b)?|\u8bed\u8a00\u6a21\u578b|\u8bc4\u4f30\u8005|\u52a9\u624b)[,\uff0c\\s]*', re.I)


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
