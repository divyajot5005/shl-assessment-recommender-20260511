from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import httpx

from app.config import Settings

TEST_TYPE_CODE_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

TEST_TYPE_ORDER = ("A", "B", "C", "D", "E", "K", "P", "S")

STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "new",
    "report",
    "assessment",
    "questionnaire",
    "occupational",
    "interactive",
    "verify",
}

MANUAL_ALIASES = {
    "Occupational Personality Questionnaire OPQ32r": ["opq32r", "opq"],
    "Global Skills Assessment": ["gsa"],
    "Dependability and Safety Instrument (DSI)": ["dsi"],
    "SHL Verify Interactive G+": ["verify g+", "g+"],
    "Amazon Web Services (AWS) Development (New)": ["aws development", "aws"],
    "Contact Center Call Simulation (New)": ["contact centre call simulation"],
    "Entry Level Customer Serv-Retail & Contact Center": [
        "entry level customer service retail and contact center",
        "entry level customer service contact center",
    ],
}

SHORT_ALIAS_ALLOWLIST = {
    "aws",
    "dsi",
    "g+",
    "gsa",
    "hipaa",
    "opq",
    "opq32r",
    "sql",
    "svar",
    "ucf",
}


def normalize_text(value: str) -> str:
    lowered = value.lower().replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9+]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_duration_minutes(duration: str) -> int | None:
    match = re.search(r"(\d+)", duration or "")
    return int(match.group(1)) if match else None


def infer_report_flag(name: str, description: str) -> bool:
    lowered_name = name.lower()
    lowered_description = description.lower().strip()
    return (
        "report" in lowered_name
        or lowered_name.endswith("development")
        or lowered_description.startswith("this report")
        or lowered_description.startswith("report that")
    )


def derive_test_type(keys: Iterable[str]) -> str:
    codes = {
        TEST_TYPE_CODE_MAP[key]
        for key in keys
        if key in TEST_TYPE_CODE_MAP
    }
    return ",".join(code for code in TEST_TYPE_ORDER if code in codes)


def generate_aliases(name: str, link: str) -> set[str]:
    aliases = {normalize_text(name)}

    cleaned_name = re.sub(r"\([^)]*\)", "", name).strip()
    aliases.add(normalize_text(cleaned_name))

    slug = link.rstrip("/").split("/")[-1].replace("-", " ")
    aliases.add(normalize_text(slug))

    parts = re.findall(r"[A-Za-z0-9+]+", name)
    filtered = [part for part in parts if normalize_text(part) and normalize_text(part) not in STOPWORDS]
    if filtered:
        acronym = "".join(part[0] for part in filtered if part[0].isalnum())
        if len(acronym) >= 3:
            aliases.add(normalize_text(acronym))

    for part in parts:
        if any(char.isdigit() for char in part) or part.isupper():
            token = normalize_text(part)
            if len(token) >= 2:
                aliases.add(token)

    for alias in MANUAL_ALIASES.get(name, []):
        aliases.add(normalize_text(alias))

    return {alias for alias in aliases if len(alias) >= 2}


@dataclass(slots=True)
class CatalogRecord:
    entity_id: str
    name: str
    url: str
    description: str
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    duration_text: str
    duration_minutes: int | None
    remote: bool
    adaptive: bool
    keys: tuple[str, ...]
    test_type: str
    is_report: bool
    aliases: tuple[str, ...]
    compact_aliases: tuple[str, ...]
    search_text: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["job_levels"] = list(self.job_levels)
        data["languages"] = list(self.languages)
        data["keys"] = list(self.keys)
        data["aliases"] = list(self.aliases)
        data["compact_aliases"] = list(self.compact_aliases)
        return data


class CatalogRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.records: list[CatalogRecord] = []
        self.by_name: dict[str, CatalogRecord] = {}
        self.by_url: dict[str, CatalogRecord] = {}
        self.alias_lookup: dict[str, list[CatalogRecord]] = {}

    def ensure_catalog_file(self) -> Path:
        path = self.settings.raw_catalog_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path

        with httpx.Client(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            response = client.get(self.settings.catalog_url)
            response.raise_for_status()
            path.write_bytes(response.content)
        return path

    def load(self) -> list[CatalogRecord]:
        raw_path = self.ensure_catalog_file()
        payload = json.loads(raw_path.read_text(encoding="utf-8"), strict=False)
        records: list[CatalogRecord] = []

        for item in payload:
            required_fields = {"entity_id", "name", "link", "description", "keys"}
            if not required_fields.issubset(item):
                missing = required_fields.difference(item)
                raise ValueError(f"Catalog item missing fields: {sorted(missing)}")

            record = CatalogRecord(
                entity_id=str(item["entity_id"]),
                name=item["name"].strip(),
                url=item["link"].strip(),
                description=item["description"].strip(),
                job_levels=tuple(level.strip() for level in item.get("job_levels", []) if level.strip()),
                languages=tuple(language.strip() for language in item.get("languages", []) if language.strip()),
                duration_text=(item.get("duration") or "").strip(),
                duration_minutes=extract_duration_minutes(item.get("duration") or ""),
                remote=str(item.get("remote", "")).lower() == "yes",
                adaptive=str(item.get("adaptive", "")).lower() == "yes",
                keys=tuple(key.strip() for key in item.get("keys", []) if key.strip()),
                test_type=derive_test_type(item.get("keys", [])),
                is_report=infer_report_flag(item["name"], item["description"]),
                aliases=tuple(sorted(generate_aliases(item["name"], item["link"]))),
                compact_aliases=tuple(sorted({compact_text(alias) for alias in generate_aliases(item["name"], item["link"])})),
                search_text=" ".join(
                    part
                    for part in (
                        item["name"],
                        item["description"],
                        " ".join(item.get("job_levels", [])),
                        " ".join(item.get("languages", [])),
                        " ".join(item.get("keys", [])),
                        item["link"].rstrip("/").split("/")[-1].replace("-", " "),
                    )
                    if part
                ),
            )
            records.append(record)

        self.records = records
        self.by_name = {record.name: record for record in records}
        self.by_url = {record.url: record for record in records}
        self.alias_lookup = {}
        for record in records:
            for alias in record.aliases:
                self.alias_lookup.setdefault(alias, []).append(record)

        self.write_processed_snapshot()
        return records

    def write_processed_snapshot(self) -> None:
        path = self.settings.processed_catalog_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([record.to_dict() for record in self.records], indent=2),
            encoding="utf-8",
        )

    def resolve_named_records(self, text: str) -> list[CatalogRecord]:
        normalized = normalize_text(text)
        matches: list[CatalogRecord] = []
        seen: set[str] = set()

        for alias, records in sorted(self.alias_lookup.items(), key=lambda item: len(item[0]), reverse=True):
            compact_alias = compact_text(alias)
            phrase_match = normalized == alias or f" {alias} " in f" {normalized} "
            compact_match = compact_alias in SHORT_ALIAS_ALLOWLIST and compact_text(text) == compact_alias

            if len(alias) <= 3 and alias not in SHORT_ALIAS_ALLOWLIST and not phrase_match:
                continue

            if phrase_match or compact_match:
                for record in records:
                    if record.url not in seen:
                        matches.append(record)
                        seen.add(record.url)

        return matches
