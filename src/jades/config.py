"""Instance-local configuration. Credentials never enter public configuration."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

MODULES = ("clean", "decompose", "pair", "judge", "overall", "fact_decompose", "fact_clarify", "fact_check")
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731:together"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
AUTO_ENV = object()

# These fields control the transport or the single-result structured protocol.
PROTECTED_PARAMETERS = frozenset({
    "model", "messages", "tools", "tool_choice", "response_format", "stream",
    "api_key", "base_url", "extra_headers", "extra_query", "timeout", "n",
    "functions", "function_call", "stream_options", "parallel_tool_calls",
})


def validate_extra_body(value, explicit_fields=()):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("extra_body must be a JSON object")
    protected = PROTECTED_PARAMETERS | {"temperature", "max_tokens", "extra_body"} | set(explicit_fields)
    if protected.intersection(value):
        raise ValueError("extra_body must not override transport, model, messages, output protocol or explicit parameters")


class UntrustedConnectionConfig(ValueError):
    """An implicitly discovered file attempted to change a credential destination."""


class ConfigurationChangedError(ValueError):
    """An evaluator configuration changed after its execution snapshot was created."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "HF_TOKEN"
    timeout: float = Field(60, gt=0)
    temperature: float | None = 0.0
    max_tokens: int | None = Field(None, gt=0)
    output_mode: Literal["tool", "json_schema", "json_object", "text"] = "tool"
    max_retries: int = Field(2, ge=0, le=10)
    output_retries: int = Field(1, ge=0, le=10)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_connection(self):
        from urllib.parse import urlsplit
        u = urlsplit(self.base_url)
        if u.scheme not in ("http", "https") or not u.netloc or u.username or u.password or u.query or u.fragment:
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query or fragment")
        if not self.model.strip() or not self.api_key_env.isidentifier():
            raise ValueError("model and api_key_env must be valid nonempty identifiers")
        if PROTECTED_PARAMETERS.intersection(self.parameters):
            raise ValueError("parameters must not override routing, credentials, messages or output protocol")
        validate_extra_body(self.parameters.get("extra_body"), set(self.parameters) - {"extra_body"})
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("parameters must contain finite, JSON-serializable values") from None
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)
    # Private: cache population must not change serialized config or resume/cache keys.
    _module_cache: dict = PrivateAttr(default_factory=dict)
    _sources: dict = PrivateAttr(default_factory=dict)
    llm: ModelConfig = Field(default_factory=ModelConfig)
    modules: dict[str, dict[str, Any]] = Field(default_factory=dict)
    consider_full: bool = False
    overall_llm: bool = False
    use_web_search: bool = False
    fact_check: bool = False
    use_memory: bool = True
    memory_path: str = ".jades/cache.sqlite3"
    max_concurrency: int = Field(10, gt=0)
    sample_timeout: float = Field(500, gt=0)
    search_provider: Literal["brave", "tavily"] = "brave"
    fact_search_provider: Literal["brave", "tavily"] = "tavily"
    max_search_rounds: int = Field(3, ge=1, le=20)
    resource_dir: str | None = None
    # The baseline model name is fixed; revision is resolved and recorded at prepare time.
    semantic_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

    @model_validator(mode="after")
    def validate_modules(self):
        if set(self.modules) - set(MODULES):
            raise ValueError("unknown LLM module")
        for name in self.modules:
            self.for_module(name)
        return self

    def for_module(self, name: str) -> ModelConfig:
        if name not in MODULES:
            raise ValueError(f"unknown module: {name}")
        defaults = self.llm.model_dump()
        overrides = self.modules.get(name, {})
        cached = self._module_cache.get(name)
        if cached is None or cached[0] != defaults or cached[1] != overrides:
            # Snapshot nested dictionaries so in-place configuration edits invalidate the cache.
            resolved = ModelConfig.model_validate(_merge(defaults, overrides))
            cached = (defaults, deepcopy(overrides), resolved.model_copy(deep=True))
            self._module_cache[name] = cached
        # As before, callers receive a fresh model. No request may mutate the cached template.
        return cached[2].model_copy(update={"parameters": deepcopy(cached[2].parameters)})

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump())


def fingerprint(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for key, value in b.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _environment_overlay(env):
    overlay: dict = {"llm": {}, "modules": {}}
    for key in ModelConfig.model_fields:
        name = "JADES_" + key.upper()
        if name in env:
            overlay["llm"][key] = json.loads(env[name]) if key == "parameters" else env[name]
        for module in MODULES:
            name = f"JADES_{module.upper()}_{key.upper()}"
            if name in env:
                overlay["modules"].setdefault(module, {})[key] = json.loads(env[name]) if key == "parameters" else env[name]
    for key in Config.model_fields:
        name = "JADES_" + key.upper()
        if key not in ("llm", "modules") and name in env:
            overlay[key] = env[name]
    return overlay


def load_config(config_path: str | Path | None = None, env_file=AUTO_ENV,
                overrides: dict | None = None, environ: dict | None = None) -> tuple[Config, dict[str, str]]:
    """Discover default settings, but require explicit sources for custom credential routing.

    Explicit config/env paths, process JADES_* variables and Python overrides are
    trusted. An implicit cwd file may not change base_url or api_key_env. This
    preserves automatic HF_TOKEN loading without trusting arbitrary cwd routing.
    """
    explicit_env = env_file is not AUTO_ENV
    env_path = Path(".env") if env_file is AUTO_ENV else (Path(env_file) if env_file else None)
    if explicit_env and env_path is not None and not env_path.is_file():
        raise FileNotFoundError(f"Explicit env file does not exist or is not a file: {env_path}")
    file_env = {k: v for k, v in (dotenv_values(env_path, interpolate=False) if env_path and env_path.exists() else {}).items() if v is not None}
    process_env = dict(os.environ if environ is None else environ)
    env = {**file_env, **process_env}
    path = Path(config_path) if config_path else Path("jades.toml")
    data: dict = {}
    if config_path and not path.exists():
        raise FileNotFoundError(path)
    if path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with path.open("rb") as f:
            data = tomllib.load(f)
    config = Config.model_validate(_merge(_merge(data, _environment_overlay(env)), overrides or {}))
    trusted_env = {**(file_env if explicit_env else {}), **process_env}
    trusted = Config.model_validate(_merge(_merge(data if config_path else {}, _environment_overlay(trusted_env)), overrides or {}))
    for module in MODULES:
        actual, expected = config.for_module(module), trusted.for_module(module)
        if (actual.base_url, actual.api_key_env) != (expected.base_url, expected.api_key_env):
            raise UntrustedConnectionConfig(
                f"Implicit configuration changes credential routing for {module}. "
                "Select the trusted file explicitly with --config/--env-file (Python: config_path/env_file), "
                "or provide routing through process JADES_* variables or explicit overrides."
            )
    names = {config.for_module(m).api_key_env for m in MODULES} | {"BRAVE_API_KEY", "TAVILY_API_KEY"}
    config._sources = {"config": str(path.resolve()) if path.exists() else None,
                       "env_file": str(env_path.resolve()) if env_path and env_path.exists() else None}
    return config, {name: env[name] for name in names if name in env}
