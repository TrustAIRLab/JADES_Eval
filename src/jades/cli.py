from __future__ import annotations

import argparse
import asyncio
import importlib.resources
import json
from pathlib import Path

from .batch import run_batch
from .config import load_config, AUTO_ENV
from .evaluator import AsyncEvaluator
from .metrics import Recorder


def parser():
    p = argparse.ArgumentParser(prog="jades", description="JADES evaluation, preserving JADES_light core logic")
    p.add_argument("--version", action="version", version="jades-eval 0.1.4")
    subs = p.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init", help="Create configuration templates without overwriting existing files")
    init.add_argument("--directory", default=".")
    prep = subs.add_parser("prepare-resources", help="Download/cache the reference NLTK and MiniLM resources")
    evaluate = subs.add_parser("evaluate", help="Evaluate JSON/JSONL or JailbreakBench samples")
    for command in (prep, evaluate):
        command.add_argument("--config")
        command.add_argument("--env-file", help="Explicitly select a trusted .env file (default: discover .env for the default connection)")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--response-field", choices=["response", "truncated_response"], default="response")
    evaluate.add_argument("--fact-check", action="store_true", default=None)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--legacy-checkpoint", help="Explicitly import a v0.1.0 checkpoint with incomplete historical template provenance; requires --resume")
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument("--legacy", action="store_true")
    evaluate.add_argument("--include-fingerprints", action="store_true", help="Include prompt/config fingerprints, batch signatures and fingerprint compatibility details in public output")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--model")
    evaluate.add_argument("--base-url")
    evaluate.add_argument("--api-key-env")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            root = Path(args.directory)
            root.mkdir(parents=True, exist_ok=True)
            for template, name in (("jades.toml", "jades.toml"), ("env.example", ".env.example")):
                target = root / name
                if target.exists():
                    print(f"Preserved existing {target}")
                    continue
                with target.open("x", encoding="utf-8") as f:
                    f.write(importlib.resources.files("jades").joinpath("templates", template).read_text())
                print(f"Created {target}")
            return 0
        overrides = {}
        if args.command == "evaluate":
            if args.fact_check is not None:
                overrides["fact_check"] = args.fact_check
            overrides["llm"] = {k: getattr(args, k) for k in ("model", "base_url", "api_key_env") if getattr(args, k) is not None}
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            if args.resume and args.overwrite:
                raise ValueError("--resume and --overwrite are mutually exclusive")
        config, env = load_config(args.config, args.env_file if args.env_file is not None else AUTO_ENV, overrides)
        if args.command == "prepare-resources":
            from .resources import prepare
            recorder = Recorder()
            prepare(config.resource_dir, config.semantic_revision, recorder)
            print(json.dumps(recorder.finish().model_dump(), indent=2))
            return 0
        async def run():
            async with AsyncEvaluator(config, env) as evaluator:
                return await run_batch(evaluator, args.input, args.output, response_field=args.response_field, resume=args.resume,
                                       overwrite=args.overwrite, legacy=args.legacy, limit=args.limit,
                                       legacy_checkpoint=args.legacy_checkpoint, include_fingerprints=args.include_fingerprints)
        print(f"Configuration: {config._sources.get('config') or 'defaults'}; env file: {config._sources.get('env_file') or 'none'}")
        routes = {}
        for module in ("clean", "decompose", "pair", "judge", "overall", "fact_decompose", "fact_clarify", "fact_check"):
            model = config.for_module(module)
            routes.setdefault((model.model, model.base_url, model.api_key_env), []).append(module)
        for (model, endpoint, key_name), modules in routes.items():
            print(f"{', '.join(modules)}: {model} @ {endpoint} (credential variable: {key_name})")
        summary = asyncio.run(run())
        tokens = summary["current_run"]["tokens"]
        print(f"Completed: {summary['success_count']} successful, {summary['failure_count']} failed, {summary['skipped_count']} resumed")
        print(f"Wall time: {summary['wall_time_seconds']:.3f}s; known tokens: {tokens['known_total_tokens']}; unknown-usage requests: {tokens['unknown_usage_requests']}")
        print(f"Results: {args.output}")
        if args.include_fingerprints:
            print(f"Batch signature: {summary['signature']}")
            print(f"Prompt fingerprint: {summary['prompt_fingerprint']}")
            print(f"Config fingerprint: {summary['config_fingerprint']}")
        for warning in summary.get("accounting_warnings", []) + summary.get("compatibility_warnings", []):
            print(f"Warning: {warning}")
        return 1 if summary["failure_count"] else 0
    except KeyboardInterrupt:
        print("Interrupted. Saved progress can be resumed with --resume.")
        return 130
    except Exception as exc:
        # ValidationError may contain arbitrary user-provided secrets in input_value.
        from pydantic import ValidationError
        if isinstance(exc, ValidationError):
            fields = [".".join(map(str, e["loc"])) for e in exc.errors(include_input=False)]
            print("Configuration validation failed for: " + ", ".join(fields))
        else:
            print(f"{type(exc).__name__}: {exc}")
        return 2
