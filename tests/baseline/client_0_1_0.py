"""One observable OpenAI-compatible transport for every LLM operation."""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ValidationError

from .config import Config
from .metrics import Recorder, utc_now


class ModelOutputError(RuntimeError):
    pass


class LLMCallError(RuntimeError):
    """Deliberately excludes raw response bodies, request URLs and headers."""


class LLMClient:
    def __init__(self, config: Config, credentials: dict, transport=None):
        self.config, self.credentials = config, credentials
        self.transport = transport
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.clients = {}

    def validate_credentials(self, modules):
        for module in modules:
            cfg = self.config.for_module(module)
            if not self.credentials.get(cfg.api_key_env):
                raise ValueError(f"Missing credential variable {cfg.api_key_env} for module {module}")

    def _client(self, cfg):
        import httpx
        key = (cfg.base_url, cfg.api_key_env, cfg.timeout)
        if key not in self.clients:
            secret = self.credentials.get(cfg.api_key_env)
            if not secret:
                raise ValueError(f"Missing credential variable {cfg.api_key_env}")
            self.clients[key] = AsyncOpenAI(base_url=cfg.base_url, api_key=secret, max_retries=0, timeout=cfg.timeout,
                                           http_client=httpx.AsyncClient(transport=self.transport) if self.transport else None)
        return self.clients[key]

    async def close(self):
        for client in self.clients.values():
            await client.close()
        self.clients.clear()

    async def request(self, module, messages, recorder: Recorder, purpose="evaluation", **kwargs):
        cfg = self.config.for_module(module)
        client = self._client(cfg)
        settings = dict(cfg.parameters)
        if cfg.temperature is not None:
            settings["temperature"] = cfg.temperature
        if cfg.max_tokens is not None:
            settings["max_tokens"] = cfg.max_tokens
        settings.update(kwargs)
        logical_id = uuid.uuid4().hex
        for attempt in range(cfg.max_retries + 1):
            queued = time.perf_counter()
            record = {"request_id": uuid.uuid4().hex, "logical_request_id": logical_id, "module": module, "model": cfg.model, "base_url": cfg.base_url,
                      "kind": "llm", "purpose": purpose, "attempt": attempt + 1, "started_at": utc_now(),
                      "queue_wait_seconds": 0.0, "api_time_seconds": 0.0, "usage": None,
                      "status": "cancelled", "sent": False}
            acquired = False
            api_start = None
            retry = False
            try:
                await self.semaphore.acquire()
                acquired = True
                record["queue_wait_seconds"] = time.perf_counter() - queued
                record["sent"] = True
                api_start = time.perf_counter()
                recorder.event({"event": "request_started", **record, "status": "in_flight"})
                response = await client.chat.completions.create(model=cfg.model, messages=messages, **settings)
                record["status"] = "ok"
                record["server_request_id"] = getattr(response, "_request_id", None)
                record["completion_id"] = response.id
                record["usage"] = response.usage.model_dump() if response.usage else None
                return response
            except asyncio.CancelledError:
                raise
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                record.update(status="error", error_type=type(exc).__name__, http_status=status)
                retry = (status is None or status in (408, 409, 429) or status >= 500) and attempt < cfg.max_retries
                if not retry:
                    raise LLMCallError(f"{module}: {type(exc).__name__}" + (f" (HTTP {status})" if status else "")) from None
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                raise LLMCallError(f"{module}: {type(exc).__name__}") from None
            finally:
                if api_start is not None:
                    record["api_time_seconds"] = time.perf_counter() - api_start
                if not acquired:
                    record["queue_wait_seconds"] = time.perf_counter() - queued
                if acquired:
                    self.semaphore.release()
                recorder.request(record)
            if retry:
                start = time.perf_counter()
                try:
                    await asyncio.sleep(min(2 ** attempt, 8))
                finally:
                    recorder.retry_wait(module, time.perf_counter() - start)

    async def structured(self, module: str, system: str, user: str, schema: type[BaseModel], recorder: Recorder,
                         search=None):
        """Default tool output mirrors PydanticAI's final_result protocol, with no protocol fallback."""
        cfg = self.config.for_module(module)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        output_tool = {"type": "function", "function": {"name": "final_result", "description": "The final response which ends this conversation",
                         "parameters": schema.model_json_schema()}}
        search_tool = {"type": "function", "function": {"name": "web_search", "description": "Search the web for information to verify facts. Only use when unsure about your judgement. Use as few searches as possible.",
                         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
        failures = 0
        searches = 0
        while True:
            options = {}
            tools = []
            if cfg.output_mode == "tool":
                tools.append(output_tool)
            if search and searches < self.config.max_search_rounds:
                tools.append(search_tool)
            if tools:
                options.update(tools=tools, tool_choice="required" if cfg.output_mode == "tool" else "auto")
            if cfg.output_mode == "json_schema":
                options["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
            elif cfg.output_mode == "json_object":
                options["response_format"] = {"type": "json_object"}
            if cfg.output_mode in ("text", "json_object") and not any(m.get("jades_schema") for m in messages):
                # Explicit opt-in modes need schema instructions; default mode leaves baseline prompts unchanged.
                messages.append({"role": "user", "content": "Return only a JSON object conforming to this schema: " + json.dumps(schema.model_json_schema()), "jades_schema": True})
            wire_messages = [{k: v for k, v in m.items() if k != "jades_schema"} for m in messages]
            response = await self.request(module, wire_messages, recorder, purpose="output_repair" if failures else ("after_search" if searches else "evaluation"), **options)
            if not response.choices:
                raise ModelOutputError(f"{module}: response has no choices")
            choice = response.choices[0]
            msg = choice.message
            if msg.refusal:
                raise ModelOutputError(f"{module}: evaluator model refused to evaluate")
            if choice.finish_reason == "length":
                raise ModelOutputError(f"{module}: model output truncated")
            tool_calls = msg.tool_calls or []
            search_calls = [t for t in tool_calls if t.function.name == "web_search"]
            if search_calls and search:
                if searches >= self.config.max_search_rounds:
                    raise ModelOutputError(f"{module}: search round limit exceeded")
                messages.append(msg.model_dump(exclude_none=True))
                for call in tool_calls:
                    if call.function.name == "web_search" and searches < self.config.max_search_rounds:
                        searches += 1
                        try:
                            query = json.loads(call.function.arguments)["query"]
                            if not isinstance(query, str) or not query.strip():
                                raise ValueError()
                        except (ValueError, KeyError, TypeError):
                            content = "Invalid search query; supply a nonempty query string."
                        else:
                            content = await search(query)
                    else:
                        content = "Complete the final_result after processing the search results; search limit may have been reached."
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                # Malformed queries must not allow an unbounded tool loop.
                if all(t.function.name == "web_search" for t in tool_calls) and not searches:
                    failures += 1
                    if failures > cfg.output_retries:
                        raise ModelOutputError(f"{module}: invalid search tool arguments")
                continue
            final = next((t for t in tool_calls if t.function.name == "final_result"), None)
            raw = final.function.arguments if final else (msg.content or "")
            try:
                if cfg.output_mode == "tool" and final is None:
                    raise ValueError("expected final_result tool")
                return schema.model_validate_json(raw)
            except (ValidationError, ValueError):
                failures += 1
                if failures > cfg.output_retries:
                    raise ModelOutputError(f"{module}: invalid structured output after {failures} attempts") from None
                # Do not include raw provider errors or credentials in repair prompts.
                messages.append(msg.model_dump(exclude_none=True))
                if tool_calls:
                    messages.extend({"role": "tool", "tool_call_id": t.id, "content": "Output does not conform to the required schema. Return a valid final_result."} for t in tool_calls)
                else:
                    messages.append({"role": "user", "content": "Return a valid result using the required output schema and protocol."})
