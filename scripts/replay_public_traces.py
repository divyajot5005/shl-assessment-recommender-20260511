from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import sys
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.engine import SHLAgentService
from app.schemas import ChatMessage


@dataclass(slots=True)
class ParsedTrace:
    name: str
    user_turns: list[str]
    expected_shortlist: list[str]


TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|", re.MULTILINE)
EXPECTED_NAME_OVERRIDES = {
    "Microsoft Excel 365 (New)": "Microsoft Excel 365 - Essentials (New)",
    "SVAR Spoken English (US) (New)": "SVAR - Spoken English (US) (New)",
}


def ensure_traces_dir(settings_path: Path) -> None:
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Public traces directory not found at {settings_path}. "
            "Run `python scripts/fetch_reference_data.py --skip-if-present` first."
        )


def parse_trace(path: Path) -> ParsedTrace:
    content = path.read_text(encoding="utf-8")
    turn_sections = re.split(r"### Turn \d+", content)
    user_turns: list[str] = []
    for section in turn_sections:
        match = re.search(r"\*\*User\*\*\s*(.*?)\s*\*\*Agent\*\*", section, re.DOTALL)
        if not match:
            continue
        raw = match.group(1)
        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith(">"):
                lines.append(stripped[1:].strip())
        joined = " ".join(lines).strip()
        if joined:
            user_turns.append(joined)

    tables = re.split(r"_`end_of_conversation`:", content)
    last_table_source = tables[-2] if len(tables) > 1 else content
    expected = [row.group(1).strip() for row in TABLE_ROW_RE.finditer(last_table_source)]

    return ParsedTrace(name=path.stem, user_turns=user_turns, expected_shortlist=expected)


def canonicalize_expected_names(expected_names: list[str], service: SHLAgentService) -> list[str]:
    canonical: list[str] = []
    catalog_names = list(service.catalog.by_name)
    for name in expected_names:
        if name in EXPECTED_NAME_OVERRIDES:
            canonical.append(EXPECTED_NAME_OVERRIDES[name])
            continue
        if name in service.catalog.by_name:
            canonical.append(name)
            continue
        matches = get_close_matches(name, catalog_names, n=1, cutoff=0.75)
        canonical.append(matches[0] if matches else name)
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="Evaluate a deployed /chat endpoint instead of the local service.")
    parser.add_argument("--output-name", default="public_trace_metrics.json")
    return parser.parse_args()


def run_local_chat(service: SHLAgentService, messages: list[ChatMessage]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = service.chat(messages)
    latency = time.perf_counter() - started
    return (
        {
            "reply": result.reply,
            "recommendations": [item.model_dump() for item in result.recommendations],
            "end_of_conversation": result.end_of_conversation,
            "debug": result.debug,
        },
        latency,
    )


def run_remote_chat(base_url: str, messages: list[ChatMessage]) -> tuple[dict[str, Any], float]:
    payload = {"messages": [item.model_dump() for item in messages]}
    started = time.perf_counter()
    with httpx.Client(timeout=30.0) as client:
        response = client.post(base_url.rstrip("/") + "/chat", json=payload)
        response.raise_for_status()
    latency = time.perf_counter() - started
    body = response.json()
    return (
        {
            "reply": str(body["reply"]),
            "recommendations": list(body.get("recommendations", [])),
            "end_of_conversation": bool(body["end_of_conversation"]),
            "debug": {},
        },
        latency,
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_traces_dir(settings.public_traces_dir)
    service = SHLAgentService(settings)

    results = []
    total_recall = 0.0
    total_candidate_hit = 0.0
    candidate_hit_samples = 0
    latency_samples: list[float] = []

    for path in sorted(settings.public_traces_dir.glob("C*.md")):
        trace = parse_trace(path)
        expected_names = canonicalize_expected_names(trace.expected_shortlist, service)
        messages: list[ChatMessage] = []
        final_result: dict[str, Any] | None = None
        trace_latencies: list[float] = []

        for user_turn in trace.user_turns:
            messages.append(ChatMessage(role="user", content=user_turn))
            if args.base_url:
                result, latency = run_remote_chat(args.base_url, messages)
            else:
                result, latency = run_local_chat(service, messages)
            messages.append(ChatMessage(role="assistant", content=result["reply"]))
            final_result = result
            trace_latencies.append(latency)
            if result["end_of_conversation"]:
                break

        predicted_names = [item["name"] for item in (final_result["recommendations"] if final_result else [])]
        hits = len(set(predicted_names).intersection(expected_names))
        recall = hits / len(expected_names) if expected_names else 0.0
        total_recall += recall

        candidate_names = final_result["debug"].get("candidate_names", []) if final_result else []
        candidate_hit_rate = None
        if candidate_names:
            candidate_hits = len(set(candidate_names).intersection(expected_names))
            candidate_hit_rate = candidate_hits / len(expected_names) if expected_names else 0.0
            total_candidate_hit += candidate_hit_rate
            candidate_hit_samples += 1

        latency_samples.extend(trace_latencies)
        results.append(
            {
                "trace": trace.name,
                "turns_until_shortlist": len(messages),
                "within_turn_budget": len(messages) <= settings.max_turns,
                "expected_shortlist": expected_names,
                "predicted_shortlist": predicted_names,
                "recall_at_10": recall,
                "candidate_hit_rate": candidate_hit_rate,
                "schema_ok": bool(final_result),
                "all_urls_catalog_backed": all(
                    item["url"] in service.catalog.by_url for item in (final_result["recommendations"] if final_result else [])
                ),
                "mean_chat_latency_seconds": (sum(trace_latencies) / len(trace_latencies)) if trace_latencies else None,
            }
        )

    summary = {
        "trace_count": len(results),
        "execution_mode": "remote" if args.base_url else "local",
        "base_url": args.base_url,
        "mean_recall_at_10": total_recall / len(results) if results else 0.0,
        "mean_candidate_hit_rate": (total_candidate_hit / candidate_hit_samples) if candidate_hit_samples else None,
        "mean_chat_latency_seconds": (sum(latency_samples) / len(latency_samples)) if latency_samples else None,
        "results": results,
    }

    output_dir = settings.artifacts_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
