"""Public-output projection; internal checkpoint provenance remains intact."""

FINGERPRINT_FIELDS = ("prompt_fingerprint", "config_fingerprint")
FINGERPRINT_WARNINGS = frozenset({"legacy_prompt_fingerprint_incomplete"})


def project_result(data, *, include_fingerprints=False):
    result = dict(data)
    if not include_fingerprints and isinstance(result.get("metadata"), dict):
        metadata = dict(result["metadata"])
        for key in FINGERPRINT_FIELDS:
            metadata.pop(key, None)
        result["metadata"] = metadata
    return result


def compatibility_fields(warnings, *, include_fingerprints=False):
    visible = [warning for warning in warnings if include_fingerprints or warning not in FINGERPRINT_WARNINGS]
    return {"compatibility_warnings": visible} if visible or include_fingerprints else {}
