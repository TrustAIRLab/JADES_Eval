from __future__ import annotations

import asyncio
import json
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from . import core, messages, prompts, resources
from .cache import DecompositionCache
from .client import LLMClient
from .config import Config, load_config, AUTO_ENV, ConfigurationChangedError
from .contracts import prompt_fingerprint, decomposition_fingerprint
from .numeric import require_finite, NonFiniteOutputError
from .models import ScoringPoints
from .metrics import Metrics, Recorder
from .models import CleanSentence, ScoringPointJudgement, DecomposedUnitFacts, ClearUnitFacts, PipelineState
from .search import SearchClient
from .refusals import structured_refusal, flag_suspected_refusal
from .presentation import project_result


class FactCheckResult(BaseModel):
    given_fact: str
    is_fact_correct: Literal["true", "false", "unknown"]
    related_info: list[str]
    reason: str
    source_sentence: str


class EvaluationResult(BaseModel):
    _fingerprints: dict = PrivateAttr(default_factory=dict)
    state: PipelineState
    metrics: Metrics
    metadata: dict = Field(default_factory=dict)
    all_decomposed_clear_unit_facts: list[dict] | None = None
    all_fact_check_results: list[dict] | None = None

    @property
    def score(self):
        return self.state.jailbreak_score_weighted

    def to_dict(self, *, include_fingerprints=False):
        data = self.model_dump()
        data["metadata"].update(self._fingerprints)
        return project_result(data, include_fingerprints=include_fingerprints)

    def to_checkpoint_dict(self):
        """Internal persistence always retains provenance, regardless of display options."""
        return self.to_dict(include_fingerprints=True)

    def to_legacy_dict(self):
        result = self.state.to_output_dict()
        if self.all_fact_check_results is not None:
            result.update(all_fact_check_results=self.all_fact_check_results,
                          all_decomposed_clear_unit_facts=self.all_decomposed_clear_unit_facts)
        return result


class EvaluationError(RuntimeError):
    def __init__(self, message: str, metrics: Metrics, state: PipelineState, *, error_code=None, module=None):
        super().__init__(message)
        self.metrics = metrics
        self.state = state
        self.error_code = error_code
        self.module = module


class _MemoryAdapter:
    def __init__(self, config: Config, recorder):
        self.config, self.recorder = config, recorder
        self.cache = DecompositionCache(config.memory_path)

    def _key(self, question):
        return self.cache.key(question, self.config.for_module("decompose"), prompts.sys_prompt_question_decompose_v1_original,
                              decomposition_fingerprint())

    def load_memory_variables(self, inputs):
        question = inputs.get("question")
        if question:
            value = self.cache.get(self._key(question))
            if value is not None:
                # Validate cached numeric values too; do not let old NaN/Inf entries
                # bypass the model-output validation and crash persistence later.
                ScoringPoints.model_validate({"scoring_points_for_a_question": value})
                refusal = structured_refusal("decompose", {"scoring_points_for_a_question": value})
                if refusal:
                    flag_suspected_refusal("decompose", self.recorder, field=refusal.field,
                                           rule_id=refusal.rule_id, source="cache")
                self.recorder.metrics.modules["decompose"].cache_hits += 1
                return {"scoring_points_for_a_question": value}
        return {}

    def save_context(self, inputs, outputs):
        if inputs.get("question") and outputs.get("scoring_points_for_a_question"):
            self.cache.put(self._key(inputs["question"]), outputs["scoring_points_for_a_question"])


class _Runtime:
    @staticmethod
    async def gather(coroutines):
        return await _gather_cancel(coroutines)

    def __init__(self, evaluator, recorder):
        self.config, self.client, self.search = evaluator._execution_config, evaluator.client, evaluator.search
        self.recorder = recorder
        self.facts = None
        self.units = None

    async def call(self, module, system, user, schema, search=None):
        return await self.client.structured(module, system, user, schema, self.recorder, search)

    async def clean(self, sentence, full_response, question):
        return (await self.call("clean", prompts.CLEAN, messages.clean(sentence, full_response, question), CleanSentence)).model_dump()

    async def judge(self, point, sentences):
        if self.facts is None:
            prompt, user = prompts.sys_prompt_single_scoring_point_judgement_v1_original, messages.judge(point, sentences)
        else:
            # Same simplified evidence representation passed to extension's scoring node.
            simple = [{"given_sentence": item["given_sentence"], "sentence_fact_check_results": [
                {"given_fact": f["given_fact"], "is_fact_correct": f["is_fact_correct"]}
                for f in item["sentence_fact_check_results"]]} for item in self.facts]
            prompt, user = prompts.FACT_JUDGE, messages.fact_judge(point, sentences, simple)
        search = None
        if self.config.use_web_search:
            search = lambda query: self.search.judge_search(query, self.config.search_provider, self.recorder)
        return (await self.call("judge", prompt, user, ScoringPointJudgement, search)).model_dump()

    async def extension(self, state):
        sentences = [s["cleaned_sentence"] for s in state.cleaned_sentences if s.get("remove_whole") == False]
        full = state.full_response if self.config.consider_full else "Not Given"
        with self.recorder.module("fact_decompose"):
            decomposed = await _gather_cancel([self.call("fact_decompose", prompts.FACT_DECOMPOSE, messages.fact_decompose(s), DecomposedUnitFacts) for s in sentences])
        with self.recorder.module("fact_clarify"):
            clarified = await _gather_cancel([self.call("fact_clarify", prompts.FACT_CLARIFY, messages.fact_clarify(d.decomposed_unit_facts, state.question, full), ClearUnitFacts) for d in decomposed])
        self.units = [{**d.model_dump(), "given_sentence": s, **c.model_dump()} for s, d, c in zip(sentences, decomposed, clarified)]
        with self.recorder.module("fact_check"):
            async def check_sentence(item):
                checks = await _gather_cancel([self.check_fact(item["given_sentence"], f) for f in item["decomposed_clear_unit_facts"]]) if item["have_unit_facts"] else []
                return {"given_sentence": item["given_sentence"], "sentence_fact_check_results": checks}
            self.facts = await _gather_cancel([check_sentence(item) for item in self.units])

    async def check_fact(self, sentence, fact):
        evidence = await self.search.search(fact, self.config.fact_search_provider, self.recorder, "fact_check", fact=True)
        if not evidence:
            return {"given_fact": fact, "is_fact_correct": "unknown", "related_info": [], "reason": "No retrieved evidence available.", "source_sentence": sentence, "sources": []}
        user = messages.fact_check(sentence, fact) + "\nRetrieved information:\n" + json.dumps(evidence, ensure_ascii=False)
        result = (await self.call("fact_check", prompts.FACT_CHECK, user, FactCheckResult)).model_dump()
        result["sources"] = evidence
        return result


async def _gather_cancel(coroutines):
    tasks = [asyncio.create_task(c) for c in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class AsyncEvaluator:
    def __init__(self, config: Config | None = None, credentials: dict | None = None, *, transport=None, search_transport=None):
        if config is None or credentials is None:
            loaded, env = load_config(overrides=config.model_dump() if config is not None else None)
            config = config if config is not None else loaded
            credentials = credentials if credentials is not None else env
        self.config = config
        self._execution_config = Config.model_validate(config.model_dump())
        self._config_fingerprint = self._execution_config.fingerprint()
        self.client = LLMClient(self._execution_config, dict(credentials), transport)
        self.search = SearchClient(dict(credentials), search_transport, max_concurrency=config.max_concurrency, semaphore=self.client.semaphore)
        self._closed = False
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls, env_file=AUTO_ENV, config_path=None, **overrides):
        config, credentials = load_config(config_path, env_file, overrides)
        return cls(config, credentials)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        if not self._closed:
            await self.client.close()
            await self.search.close()
            self._closed = True

    def _check_configuration(self):
        try:
            unchanged = self.config.fingerprint() == self._config_fingerprint
        except Exception:
            unchanged = False
        if not unchanged:
            raise ConfigurationChangedError("Evaluator configuration changed; create a new Evaluator or AsyncEvaluator with the new configuration.")

    def validate_configuration(self):
        """Pure preflight; no network requests and no resource downloads."""
        self._check_configuration()
        active = ["clean", "decompose", "pair", "judge"]
        if self.config.overall_llm:
            active.append("overall")
        if self.config.fact_check:
            active.extend(["fact_decompose", "fact_clarify", "fact_check"])
            self.search.validate(self.config.fact_search_provider)
        if self.config.use_web_search:
            self.search.validate(self.config.search_provider)
        self.client.validate_credentials(active)

    async def aevaluate(self, question: str, response: str, *, metrics_path=None, run_id=None, sample_id=None,
                        include_fingerprints=False):
        if self._closed:
            raise RuntimeError("Evaluator is closed")
        # Same as light: samples are sequential; within-sample clean/judge/facts are concurrent.
        async with self._lock:
            config = self._execution_config
            recorder = Recorder(run_id, sample_id, metrics_path)
            state = PipelineState(question=question, full_response=response)
            rt = _Runtime(self, recorder)

            async def execute():
                self.validate_configuration()
                def preprocess():
                    resources.configure(config.resource_dir, config.semantic_revision, recorder)
                    return core.preprocess_node(state)
                with recorder.module("preprocess"):
                    # NLP loading/encoding cannot be force-cancelled safely in a thread.
                    # Wait for its cleanup on cancellation, so no resource events appear after finish.
                    work = asyncio.create_task(asyncio.to_thread(preprocess))
                    try:
                        rejection = await asyncio.shield(work)
                    except asyncio.CancelledError:
                        await work
                        raise
                if rejection:
                    return
                with recorder.module("clean"):
                    await core.run_clean_node(state, rt)
                if config.fact_check:
                    await rt.extension(state)
                with recorder.module("decompose"):
                    await core.run_question_decompose_node(state, rt, _MemoryAdapter(config, recorder), config.use_memory)
                with recorder.module("pair"):
                    await core.run_pair_node(state, rt)
                with recorder.module("judge"):
                    await core.run_scoring_points_judgement_node(state, rt)
                with recorder.module("overall"):
                    await core.run_overall_evaluation_node(state, rt)
                    require_finite(state.model_dump())

            try:
                await asyncio.wait_for(execute(), config.sample_timeout)
                self._check_configuration()
            except asyncio.CancelledError:
                recorder.finish()
                raise
            except Exception as exc:
                # Error text is local, never a raw provider response or secret-bearing URL.
                from .client import LLMCallError, ModelOutputError
                reason = str(exc) if isinstance(exc, (LLMCallError, ModelOutputError, ConfigurationChangedError)) else type(exc).__name__
                if isinstance(exc, asyncio.TimeoutError):
                    reason = f"Evaluation timed out after {config.sample_timeout:g} seconds"
                elif isinstance(exc, ValueError) and str(exc).startswith("Missing credential variable "):
                    reason = str(exc)
                elif isinstance(exc, NonFiniteOutputError):
                    reason = str(exc)
                raise EvaluationError(reason, recorder.finish(), state, error_code=getattr(exc, "code", None),
                                      module=getattr(exc, "module", None)) from None
            metrics = recorder.finish()
            fingerprints = {"config_fingerprint": config.fingerprint(), "prompt_fingerprint": prompt_fingerprint()}
            metadata = {"config": config.model_dump(), "baseline": "JADES_light", "version": "0.1.3"}
            if include_fingerprints:
                metadata.update(fingerprints)
            result = EvaluationResult(state=state, metrics=metrics, all_decomposed_clear_unit_facts=rt.units,
                                      all_fact_check_results=rt.facts, metadata=metadata)
            result._fingerprints = fingerprints
            return result

    async def aevaluate_many(self, items, *, include_fingerprints=False):
        results = []
        for item in items:
            try:
                results.append(await self.aevaluate(item["question"], item["response"], include_fingerprints=include_fingerprints))
            except EvaluationError as exc:
                results.append(exc)
        return results


class Evaluator:
    """Synchronous owner of one event loop. In async applications use AsyncEvaluator."""
    def __init__(self, config=None, credentials=None, **kwargs):
        self._loop = asyncio.new_event_loop()
        self._async = AsyncEvaluator(config, credentials, **kwargs)

    @classmethod
    def from_env(cls, env_file=AUTO_ENV, config_path=None, **overrides):
        config, credentials = load_config(config_path, env_file, overrides)
        return cls(config, credentials)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _run(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coroutine.close()
            raise RuntimeError("Use AsyncEvaluator inside an existing event loop")
        if self._loop.is_closed():
            coroutine.close()
            raise RuntimeError("Evaluator is closed")
        return self._loop.run_until_complete(coroutine)

    def evaluate(self, question, response, **kwargs):
        return self._run(self._async.aevaluate(question, response, **kwargs))

    def evaluate_many(self, items, *, include_fingerprints=False):
        return self._run(self._async.aevaluate_many(items, include_fingerprints=include_fingerprints))

    def close(self):
        if not self._loop.is_closed():
            self._run(self._async.close())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()
