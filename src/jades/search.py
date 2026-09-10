from __future__ import annotations

import asyncio
import time
import uuid

import httpx

from .metrics import utc_now


class SearchClient:
    def __init__(self, credentials, transport=None, *, max_concurrency=10, semaphore=None):
        self.credentials = credentials
        self.semaphore = semaphore if semaphore is not None else asyncio.Semaphore(max_concurrency)
        self.client = httpx.AsyncClient(timeout=10, transport=transport)

    def validate(self, provider):
        key = "BRAVE_API_KEY" if provider == "brave" else "TAVILY_API_KEY"
        if not self.credentials.get(key):
            raise ValueError(f"Missing credential variable {key} for {provider} search")

    async def close(self):
        await self.client.aclose()

    async def search(self, query, provider, recorder, module, fact=False):
        self.validate(provider)
        start = time.perf_counter()
        record = {"request_id": uuid.uuid4().hex, "kind": "search", "module": module, "model": "", "provider": provider, "purpose": "fact_check" if fact else "judge_search",
                  "status": "cancelled", "started_at": utc_now(), "api_time_seconds": 0, "queue_wait_seconds": 0, "sent": True}
        acquired = False
        api_start = None
        record["sent"] = False
        try:
            await self.semaphore.acquire()
            acquired = True
            record["queue_wait_seconds"] = time.perf_counter() - start
            api_start = time.perf_counter()
            record["sent"] = True
            recorder.event({"event": "request_started", **record, "status": "in_flight"})
            if provider == "brave":
                response = await self.client.get("https://api.search.brave.com/res/v1/web/search", headers={"X-Subscription-Token": self.credentials["BRAVE_API_KEY"], "Accept": "application/json"}, params={"q": query, "count": 3})
                response.raise_for_status()
                rows = response.json().get("web", {}).get("results", [])
                evidence = [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("description", "")} for r in rows]
            else:
                payload = {"api_key": self.credentials["TAVILY_API_KEY"], "query": query, "max_results": 1 if fact else 3,
                           "include_answer": True, "include_raw_content": True, "include_images": False, "search_depth": "advanced"}
                if fact:
                    payload["include_domains"] = ["wikipedia.org"]
                response = await self.client.post("https://api.tavily.com/search", json=payload)
                response.raise_for_status()
                evidence = [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("raw_content") or r.get("content", "")} for r in response.json().get("results", [])]
            record.update(status="ok", result_count=len(evidence))
            return evidence
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
            return []
        finally:
            if api_start is not None:
                record["api_time_seconds"] = time.perf_counter() - api_start
            if not acquired:
                record["queue_wait_seconds"] = time.perf_counter() - start
            if acquired:
                self.semaphore.release()
            recorder.request(record)

    async def judge_search(self, query, provider, recorder):
        rows = await self.search(query, provider, recorder, "judge")
        if not rows:
            return "No search results found."
        # Preserve light's context representation: title + description (no URL added to its prompt).
        return "\n".join(f"- {r['title']}: {r['content']}" for r in rows)
