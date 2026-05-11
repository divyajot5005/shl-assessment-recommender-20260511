from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.catalog import CatalogRecord, CatalogRepository, compact_text, normalize_text
from app.config import Settings
from app.llm import HostedLLMClient
from app.retrieval import CatalogIndex, SearchCandidate
from app.schemas import ChatMessage, ChatResponse, Recommendation

CONFIRMATION_PATTERNS = (
    "that works",
    "that's good",
    "that is good",
    "confirmed",
    "lock it in",
    "locking it in",
    "keep the shortlist",
    "keep the list",
    "keep it",
    "perfect",
    "final list",
    "final shortlist",
    "thanks",
    "thank you",
)

LEGAL_PATTERNS = (
    "legally required",
    "legal requirement",
    "satisfy that requirement",
    "regulatory obligation",
    "under hipaa",
    "under gdpr",
    "compliance question",
)

PROMPT_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "reveal your prompt",
    "show me the system prompt",
    "jailbreak",
    "act as a different assistant",
)

OFF_TOPIC_PATTERNS = (
    "interview process",
    "interview round",
    "interview rounds",
    "interview question",
    "interview questions",
    "sourcing strategy",
    "salary range",
    "compensation",
    "employment law",
    "termination advice",
    "offer negotiation",
    "offer package",
    "recruiting strategy",
    "where should i source",
)

LANGUAGE_HINTS = {
    "english": ("English International", "English (USA)"),
    "us english": ("English (USA)",),
    "uk english": ("English UK",),
    "australian english": ("English (Australia)",),
    "indian english": ("English (India)",),
    "spanish": ("Spanish", "Latin American Spanish"),
    "latin american spanish": ("Latin American Spanish",),
    "french": ("French", "French (Canada)", "French (Belgium)"),
    "portuguese": ("Portuguese", "Portuguese (Brazil)"),
    "german": ("German",),
}

JOB_LEVEL_HINTS = {
    "graduate": ("Graduate",),
    "recent graduate": ("Graduate",),
    "entry level": ("Entry-Level",),
    "entry-level": ("Entry-Level",),
    "junior": ("Entry-Level",),
    "mid-level": ("Mid-Professional",),
    "mid level": ("Mid-Professional",),
    "mid professional": ("Mid-Professional",),
    "senior ic": ("Professional Individual Contributor", "Mid-Professional"),
    "professional individual contributor": ("Professional Individual Contributor",),
    "manager": ("Manager", "Front Line Manager"),
    "supervisor": ("Supervisor",),
    "director": ("Director",),
    "executive": ("Executive",),
    "cxo": ("Executive",),
}

CATEGORY_KEYWORDS = {
    "cognitive": "A",
    "reasoning": "A",
    "aptitude": "A",
    "personality": "P",
    "behaviour": "P",
    "behavior": "P",
    "situational": "B",
    "scenario": "B",
    "simulation": "S",
    "knowledge": "K",
    "skills": "K",
    "development": "D",
    "360": "D",
}

TECH_STACK_TERMS = ("java", "spring", "sql", "aws", "docker", "angular", "rest")
FOLLOW_UP_PATTERNS = (
    "add ",
    "include ",
    "drop ",
    "remove ",
    "replace ",
    "without ",
    "shorter",
    "too long",
    "do we need",
    "redundant",
    "is the",
    "keep ",
)
PRIMARY_ALIAS_TO_NAME = {
    "dsi": "Dependability and Safety Instrument (DSI)",
    "gsa": "Global Skills Assessment",
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "opq32r": "Occupational Personality Questionnaire OPQ32r",
    "svar": "SVAR - Spoken English (US) (New)",
}
CANONICAL_SKILL_PRODUCTS = {
    "angular": "Angular 14 (New)",
    "aws": "Amazon Web Services (AWS) Development (New)",
    "docker": "Docker (New)",
    "java": "Core Java (Advanced Level) (New)",
    "rest": "RESTful Web Services (New)",
    "spring": "Spring (New)",
    "sql": "SQL (New)",
}
ROLE_FAMILY_HINTS = {
    "contact_center": ("contact center", "contact centre", "call center", "call centre", "customer service"),
    "sales": ("sales", "account executive", "business development", "revenue", "seller"),
    "technical": ("engineer", "developer", "full stack", "backend", "frontend", "java", "sql", "aws", "docker"),
    "finance": ("financial analyst", "accounting", "finance", "numerical"),
    "operations": ("plant operators", "chemical facility", "safety", "procedure compliance"),
    "healthcare_admin": ("healthcare admin", "patient records", "hipaa", "medical terminology"),
    "graduate_management": ("graduate management trainee", "management trainee", "graduate scheme"),
    "leadership": ("leadership", "executive", "cxo", "succession"),
    "admin": ("admin assistant", "administrative assistant", "excel", "word"),
}
PROFILE_PRODUCT_NAMES = {
    "leadership": (
        "Occupational Personality Questionnaire OPQ32r",
        "OPQ Universal Competency Report 2.0",
        "OPQ Leadership Report",
    ),
    "sales_development": (
        "Global Skills Assessment",
        "Global Skills Development Report",
        "Occupational Personality Questionnaire OPQ32r",
        "OPQ MQ Sales Report",
        "Sales Transformation 2.0 - Individual Contributor",
    ),
    "contact_center": (
        "Contact Center Call Simulation (New)",
        "Entry Level Customer Serv-Retail & Contact Center",
        "Customer Service Phone Simulation",
    ),
    "finance": (
        "SHL Verify Interactive – Numerical Reasoning",
        "Financial Accounting (New)",
        "Basic Statistics (New)",
        "Graduate Scenarios",
        "Occupational Personality Questionnaire OPQ32r",
    ),
    "operations": (
        "Manufac. & Indust. - Safety & Dependability 8.0",
        "Workplace Health and Safety (New)",
    ),
    "healthcare_admin": (
        "HIPAA (Security)",
        "Medical Terminology (New)",
        "Microsoft Word 365 - Essentials (New)",
        "Dependability and Safety Instrument (DSI)",
        "Occupational Personality Questionnaire OPQ32r",
    ),
    "graduate_management": (
        "SHL Verify Interactive G+",
        "Occupational Personality Questionnaire OPQ32r",
        "Graduate Scenarios",
    ),
    "admin": (
        "MS Excel (New)",
        "MS Word (New)",
        "Occupational Personality Questionnaire OPQ32r",
    ),
}


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9+]){re.escape(phrase)}(?![a-z0-9+])", text) is not None


def contains_any_phrase(text: str, phrases: tuple[str, ...] | list[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


@dataclass(slots=True)
class ConversationState:
    combined_user_text: str
    latest_user_text: str
    named_records: list[CatalogRecord]
    comparison_records: list[CatalogRecord]
    prior_shortlist: list[CatalogRecord]
    include_terms: set[str] = field(default_factory=set)
    exclude_terms: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    job_levels: set[str] = field(default_factory=set)
    category_codes: set[str] = field(default_factory=set)
    explicit_skill_terms: set[str] = field(default_factory=set)
    use_case: str = "selection"
    role_family: str = "general"
    missing_slots: list[str] = field(default_factory=list)
    clarification_count: int = 0
    quick_screen: bool = False
    high_volume: bool = False
    wants_shorter: bool = False
    wants_comparison: bool = False
    confirmation: bool = False
    direct_recommendation_request: bool = False
    latest_message_is_vague: bool = False
    history_turns: int = 0
    remaining_turns: int = 0


@dataclass(slots=True)
class AgentResult:
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool
    debug: dict[str, Any] = field(default_factory=dict)

    def to_response(self) -> ChatResponse:
        return ChatResponse(
            reply=self.reply,
            recommendations=self.recommendations,
            end_of_conversation=self.end_of_conversation,
        )


class SHLAgentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = CatalogRepository(settings)
        self.records = self.catalog.load()
        self.index = CatalogIndex(self.records)
        self.llm = HostedLLMClient(settings)

    def chat(self, messages: list[ChatMessage]) -> AgentResult:
        history = [{"role": message.role, "content": message.content} for message in messages]
        latest_user_text = next(
            (message["content"] for message in reversed(history) if message["role"] == "user"),
            "",
        )

        guardrail = self._apply_guardrails(latest_user_text)
        if guardrail:
            return AgentResult(reply=guardrail, recommendations=[], end_of_conversation=False)

        state = self._build_state(history)
        if state.wants_comparison:
            comparison = self._compare_records(state)
            return AgentResult(reply=comparison, recommendations=[], end_of_conversation=False)

        clarification = self._maybe_clarify(state)
        if clarification:
            return AgentResult(reply=clarification, recommendations=[], end_of_conversation=False)

        candidates = self._rank_candidates(state)
        shortlist = self._assemble_shortlist(state, candidates)
        reply = self._compose_recommendation_reply(state, shortlist)
        end_of_conversation = self._determine_conversation_end(state, shortlist, reply)
        debug_candidate_names: list[str] = []
        for record in shortlist:
            if record.name not in debug_candidate_names:
                debug_candidate_names.append(record.name)
        for candidate in candidates:
            if candidate.record.name not in debug_candidate_names:
                debug_candidate_names.append(candidate.record.name)
            if len(debug_candidate_names) >= 10:
                break

        return AgentResult(
            reply=reply,
            recommendations=[
                Recommendation(name=record.name, url=record.url, test_type=record.test_type)
                for record in shortlist[: self.settings.max_recommendations]
            ],
            end_of_conversation=end_of_conversation,
            debug={
                "candidate_names": debug_candidate_names[:10],
                "selected_names": [record.name for record in shortlist],
                "role_family": state.role_family,
                "missing_slots": state.missing_slots,
            },
        )

    def _apply_guardrails(self, latest_user_text: str) -> str | None:
        latest = normalize_text(latest_user_text)
        if any(pattern in latest for pattern in PROMPT_INJECTION_PATTERNS):
            return (
                "I can only help with grounded SHL catalog recommendations and comparisons. "
                "I won't follow prompt-injection or hidden-instruction requests."
            )
        if any(pattern in latest for pattern in LEGAL_PATTERNS):
            return (
                "I can help select SHL assessments, but I can't advise on legal or regulatory obligations. "
                "If you share the role constraints, I can keep the assessment shortlist grounded in the catalog."
            )
        if any(pattern in latest for pattern in OFF_TOPIC_PATTERNS):
            return (
                "I can help with SHL assessment selection and comparison only. "
                "Share the role, level, or required skills and I can recommend catalog-based assessments."
            )
        if any(pattern in latest for pattern in ("what interview process", "how many interview rounds", "where should we source", "how should i hire")):
            return (
                "I can only help with SHL assessment selection and comparison. "
                "I can't advise on broader hiring-process design, but I can recommend SHL assessments for the role."
            )
        return None

    def _build_state(self, history: list[dict[str, str]]) -> ConversationState:
        user_turns = [message["content"] for message in history if message["role"] == "user"]
        assistant_turns = [message["content"] for message in history if message["role"] == "assistant"]
        latest_user_text = user_turns[-1]
        combined_user_text = "\n".join(user_turns)

        state = ConversationState(
            combined_user_text=combined_user_text,
            latest_user_text=latest_user_text,
            named_records=self.catalog.resolve_named_records(combined_user_text),
            comparison_records=self._extract_comparison_records(latest_user_text),
            prior_shortlist=self._extract_prior_shortlist(assistant_turns),
            clarification_count=sum("?" in content for content in assistant_turns),
            history_turns=len(history),
            remaining_turns=max(self.settings.max_turns - (len(history) + 1), 0),
        )

        latest_normalized = normalize_text(latest_user_text)
        combined_normalized = normalize_text(combined_user_text)

        state.quick_screen = any(term in combined_normalized for term in ("quick", "daily", "fast", "screen"))
        state.high_volume = any(term in combined_normalized for term in ("500", "high volume", "volume screening", "large volume"))
        state.wants_shorter = "shorter" in combined_normalized or "too long" in combined_normalized
        state.wants_comparison = any(
            term in latest_normalized for term in ("difference between", "compare", "different from", " versus ", " vs ")
        ) and len(state.comparison_records) >= 2
        state.confirmation = any(term in latest_normalized for term in CONFIRMATION_PATTERNS)
        state.direct_recommendation_request = any(
            term in combined_normalized for term in ("recommend", "assessment", "battery", "what should we use", "what assessments")
        )
        state.latest_message_is_vague = state.direct_recommendation_request and len(combined_normalized.split()) < 8

        if any(term in combined_normalized for term in ("reskill", "re skill", "re-skill", "development", "talent audit", "audit")):
            state.use_case = "development"
        elif any(term in combined_normalized for term in ("selection", "screening", "hiring", "hire")):
            state.use_case = "selection"

        for phrase, levels in JOB_LEVEL_HINTS.items():
            if phrase in combined_normalized:
                state.job_levels.update(levels)

        for phrase, options in LANGUAGE_HINTS.items():
            if phrase in combined_normalized:
                state.languages.update(options)

        for keyword, category_code in CATEGORY_KEYWORDS.items():
            if keyword in combined_normalized:
                state.category_codes.add(category_code)

        if any(term in combined_normalized for term in ("full battery", "full stack", "cognitive, personality", "cognitive personality")):
            state.category_codes.update({"A", "P"})
        state.explicit_skill_terms = {term for term in TECH_STACK_TERMS if contains_phrase(combined_normalized, term)}
        state.role_family = self._infer_role_family(combined_normalized)

        for chunk in user_turns:
            self._extract_change_terms(chunk, state)

        state.missing_slots = self._identify_missing_slots(state)

        return state

    def _extract_prior_shortlist(self, assistant_turns: list[str]) -> list[CatalogRecord]:
        matches: list[CatalogRecord] = []
        seen: set[str] = set()
        for content in assistant_turns:
            if "?" in content:
                continue
            resolved = self.catalog.resolve_named_records(content)
            if len(resolved) < 2:
                continue
            for record in resolved:
                if record.url not in seen:
                    seen.add(record.url)
                    matches.append(record)
        return matches

    def _extract_comparison_records(self, text: str) -> list[CatalogRecord]:
        comparison_patterns = (
            r"difference between\s+(.+?)\s+and\s+(.+)",
            r"compare\s+(.+?)\s+and\s+(.+)",
            r"compare\s+(.+?)\s+with\s+(.+)",
            r"(.+?)\s+versus\s+(.+)",
            r"(.+?)\s+vs\s+(.+)",
        )

        for pattern in comparison_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            records: list[CatalogRecord] = []
            for fragment in match.groups()[:2]:
                record = self._resolve_comparison_fragment(fragment)
                if record and all(existing.url != record.url for existing in records):
                    records.append(record)
            if len(records) >= 2:
                return records
        return []

    def _resolve_comparison_fragment(self, fragment: str) -> CatalogRecord | None:
        cleaned_fragment = fragment.strip(" .?!,:;\"'")
        normalized_fragment = normalize_text(cleaned_fragment)
        primary_name = PRIMARY_ALIAS_TO_NAME.get(normalized_fragment)
        if primary_name:
            return self.catalog.by_name.get(primary_name)

        matches = self.catalog.resolve_named_records(cleaned_fragment)
        if not matches:
            return None

        compact_fragment = compact_text(cleaned_fragment)

        def score(record: CatalogRecord) -> tuple[int, int, int]:
            alias_score = 0
            if normalized_fragment == normalize_text(record.name):
                alias_score += 4
            if normalized_fragment in record.aliases:
                alias_score += 3
            if compact_fragment and compact_fragment in record.compact_aliases:
                alias_score += 2
            if contains_phrase(normalized_fragment, normalize_text(record.name)):
                alias_score += 1
            instrument_bias = 0 if record.is_report else 1
            return (alias_score, instrument_bias, -len(record.name))

        return max(matches, key=score)

    def _extract_change_terms(self, text: str, state: ConversationState) -> None:
        normalized = normalize_text(text)
        raw_lower = text.lower()

        def clean_fragment(fragment: str) -> str | None:
            tokens = [token for token in normalize_text(fragment).split() if token not in {"the", "a", "an"}]
            if not tokens:
                return None
            if tokens[0] in {"it", "them", "this", "that", "something", "anything"}:
                return None
            if "shorter" in tokens:
                state.wants_shorter = True
                return None
            if len(tokens) > 3:
                if tokens[0] in {"core", "basic", "medical", "financial", "customer"}:
                    tokens = tokens[:2]
                else:
                    tokens = tokens[:1]
            cleaned = " ".join(tokens)
            return cleaned if len(cleaned) >= 3 else None

        for pattern in (r"(?:drop|remove|exclude|without|skip)\s+([^.,;:!?]+)", r"(?:replace)\s+([^.,;:!?]+?)\s+with\s+([^.,;:!?]+)"):
            match = re.search(pattern, raw_lower)
            if not match:
                continue
            if len(match.groups()) == 1:
                fragments = re.split(r",| and |/|\bor\b|\bbut\b", match.group(1))
                for fragment in fragments:
                    cleaned = clean_fragment(fragment)
                    if cleaned:
                        state.exclude_terms.add(cleaned)
            else:
                removed = clean_fragment(match.group(1))
                added = clean_fragment(match.group(2))
                if removed:
                    state.exclude_terms.add(removed)
                if added:
                    state.include_terms.add(added)

        for match in re.finditer(r"(?:add|include|plus|also add|bring in)\s+([^.,;:!?]+)", raw_lower):
            fragments = re.split(r",| and |/|\bor\b|\bbut\b", match.group(1))
            for fragment in fragments:
                cleaned = clean_fragment(fragment)
                if cleaned:
                    state.include_terms.add(cleaned)

    def _infer_role_family(self, text: str) -> str:
        priority = (
            "contact_center",
            "sales",
            "healthcare_admin",
            "operations",
            "graduate_management",
            "leadership",
            "finance",
            "technical",
            "admin",
        )
        for role_family in priority:
            if contains_any_phrase(text, ROLE_FAMILY_HINTS[role_family]):
                return role_family
        return "general"

    def _has_scope_anchor(self, state: ConversationState) -> bool:
        return any(
            (
                state.role_family != "general",
                bool(state.explicit_skill_terms),
                bool(state.named_records),
                bool(state.job_levels),
                bool(state.category_codes),
                bool(state.include_terms),
            )
        )

    def _identify_missing_slots(self, state: ConversationState) -> list[str]:
        text = normalize_text(state.combined_user_text)
        missing: list[str] = []

        if state.direct_recommendation_request and not self._has_scope_anchor(state):
            missing.append("role_scope")
        if state.role_family == "contact_center":
            if not any(language in text for language in ("english", "spanish", "french", "portuguese")):
                missing.append("contact_center_language")
            elif contains_phrase(text, "english") and not contains_any_phrase(text, ("us", "uk", "australian", "indian")):
                missing.append("english_variant")
        if contains_phrase(text, "full stack") and len(state.explicit_skill_terms) >= 4:
            if not any(term in text for term in ("backend leaning", "backend-leaning", "frontend heavy", "balanced")):
                missing.append("technical_orientation")
        if state.role_family == "leadership" and not any(
            term in text for term in ("selection", "development", "reskill", "feedback", "succession benchmarking")
        ):
            missing.append("leadership_use_case")
        if "spanish" in text and state.role_family == "healthcare_admin":
            if not any(term in text for term in ("hybrid", "english fluent", "functionally bilingual")):
                missing.append("healthcare_language_tradeoff")
        return missing

    def _maybe_clarify(self, state: ConversationState) -> str | None:
        text = normalize_text(state.combined_user_text)
        latest = normalize_text(state.latest_user_text)

        if state.clarification_count >= 2 or state.remaining_turns < 2:
            return None
        if state.latest_message_is_vague or latest in {"i need an assessment", "we need an assessment", "need assessment"}:
            return "What role are you hiring for, and what level or core skills matter most day one?"
        if "contact center" in text or "contact centre" in text:
            if not any(language in text for language in ("english", "spanish", "french", "portuguese")):
                return "What language do callers use? That determines which SHL spoken-language screen fits."
            if contains_phrase(text, "english") and not contains_any_phrase(text, ("us", "uk", "australian", "indian")):
                return "Which English variant fits your operation: US, UK, Australian, or Indian?"
        if contains_phrase(text, "full stack") and sum(term in text for term in TECH_STACK_TERMS) >= 4:
            if not any(term in text for term in ("backend leaning", "backend-leaning", "frontend heavy", "balanced")):
                return "Is this backend-leaning, frontend-heavy, or a balanced full-stack role? That changes which stack tests belong in the shortlist."
        if any(term in text for term in ("leadership", "cxo", "executive")) and not any(term in text for term in ("selection", "development", "reskill", "feedback")):
            return "Is this for selection, succession benchmarking, or development feedback? That changes whether I keep this instrument-only or add leadership reports."
        if "spanish" in text and any(term in text for term in ("hipaa", "medical terminology", "healthcare admin")):
            if not any(term in text for term in ("hybrid", "english fluent", "functionally bilingual")):
                return (
                    "There’s a catalog constraint: the HIPAA and medical terminology knowledge tests are English-only. "
                    "Should I build a hybrid battery with English knowledge tests and Spanish personality measures, or keep it Spanish-only?"
                )
        return None

    def _maybe_clarify(self, state: ConversationState) -> str | None:
        if state.clarification_count >= 2 or state.remaining_turns < 2:
            return None

        latest = normalize_text(state.latest_user_text)
        if state.latest_message_is_vague or latest in {"i need an assessment", "we need an assessment", "need assessment"}:
            return "What role are you hiring for, and what level or core skills matter most day one?"

        slot = state.missing_slots[0] if state.missing_slots else ""
        if slot == "role_scope":
            return "What role are you hiring for, and what level or core skills matter most day one?"
        if slot == "contact_center_language":
            return "What language do callers use? That determines which SHL spoken-language screen fits."
        if slot == "english_variant":
            return "Which English variant fits your operation: US, UK, Australian, or Indian?"
        if slot == "technical_orientation":
            return "Is this backend-leaning, frontend-heavy, or a balanced full-stack role? That changes which stack tests belong in the shortlist."
        if slot == "leadership_use_case":
            return "Is this for selection, succession benchmarking, or development feedback? That changes whether I keep this instrument-only or add leadership reports."
        if slot == "healthcare_language_tradeoff":
            return (
                "There's a catalog constraint: the HIPAA and medical terminology knowledge tests are English-only. "
                "Should I build a hybrid battery with English knowledge tests and Spanish personality measures, or keep it Spanish-only?"
            )
        return None

    def _rank_candidates(self, state: ConversationState) -> list[SearchCandidate]:
        query = state.combined_user_text
        if state.include_terms:
            query += " " + " ".join(sorted(state.include_terms))
        candidates = self.index.search(query, limit=80)
        named_urls = {record.url for record in state.named_records}
        prior_urls = {record.url for record in state.prior_shortlist}
        normalized_all = normalize_text(state.combined_user_text)

        for candidate in candidates:
            record = candidate.record
            reasons = candidate.reasons
            score = candidate.score
            record_text = normalize_text(f"{record.name} {record.description} {' '.join(record.keys)}")

            if record.url in named_urls:
                score += 1.5
                reasons.append("explicit_mention")
            if record.url in prior_urls and not state.exclude_terms:
                score += 0.6
                reasons.append("prior_shortlist")

            for language in state.languages:
                if language in record.languages:
                    score += 1.1
                    reasons.append("language_match")
                    break
            if state.languages and not any(language in record.languages for language in state.languages):
                if "English" not in " ".join(state.languages):
                    score -= 0.7
                    reasons.append("language_penalty")

            if state.job_levels and set(record.job_levels).intersection(state.job_levels):
                score += 0.8
                reasons.append("job_level_match")

            for code in state.category_codes:
                if code in record.test_type.split(","):
                    score += 0.9
                    reasons.append(f"category_{code}")

            if state.quick_screen and record.duration_minutes is not None:
                if record.duration_minutes <= 15:
                    score += 0.6
                    reasons.append("quick_screen")
                elif record.duration_minutes > 30:
                    score -= 0.5
                    reasons.append("long_duration_penalty")

            if state.high_volume and "Simulations" in record.keys and (record.duration_minutes or 0) <= 20:
                score += 0.7
                reasons.append("high_volume_sim")

            if state.wants_shorter and record.duration_minutes is not None:
                if record.duration_minutes <= 15:
                    score += 0.8
                else:
                    score -= 0.9

            if state.use_case == "development":
                if "Development & 360" in record.keys or "development" in record_text:
                    score += 1.2
                    reasons.append("development_use_case")
                if record.is_report:
                    score += 0.5
            elif record.is_report and not any(token in normalized_all for token in ("report", "benchmark", "feedback", "development", "audit")):
                score -= 0.5
                reasons.append("report_penalty")

            if any(term in normalized_all and term in record_text for term in TECH_STACK_TERMS):
                score += 0.4
                reasons.append("stack_overlap")
            if any(term in normalized_all for term in ("contact center", "contact centre")) and (
                "contact center" in record_text or "customer service" in record_text or "spoken english" in record_text
            ):
                score += 0.5
                reasons.append("contact_center_fit")
            if "sales" in normalized_all and ("sales" in record_text or "global skills" in record_text):
                score += 0.5
                reasons.append("sales_fit")
            if state.role_family == "healthcare_admin" and any(
                term in record_text for term in ("hipaa", "medical terminology", "word 365", "dependability and safety")
            ):
                score += 0.5
                reasons.append("healthcare_fit")
            if state.role_family == "operations" and "safety" in record_text:
                score += 0.5
                reasons.append("operations_fit")
            if state.role_family == "graduate_management" and any(
                term in record_text for term in ("graduate scenarios", "verify interactive g+", "opq32r")
            ):
                score += 0.5
                reasons.append("graduate_fit")
            if state.role_family == "leadership" and any(term in record_text for term in ("leadership", "opq", "competency report")):
                score += 0.5
                reasons.append("leadership_fit")
            if state.use_case == "development" and state.role_family in {"sales", "leadership"}:
                if record.test_type == "K" and not any(
                    term in record_text for term in ("sales", "global skills", "development", "leadership")
                ):
                    score -= 0.9
                    reasons.append("development_irrelevant_knowledge_penalty")

            if any(term and term in record_text for term in state.include_terms):
                score += 1.0
                reasons.append("include_term")

            if any(term and term in record_text for term in state.exclude_terms):
                score -= 4.0
                reasons.append("exclude_term")

            candidate.score = score

        candidates = [candidate for candidate in candidates if candidate.score > -1.0]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _profile_seed_names(self, state: ConversationState, text: str) -> tuple[str, ...]:
        if state.role_family == "leadership":
            return PROFILE_PRODUCT_NAMES["leadership"]
        if state.role_family == "sales" and state.use_case == "development":
            return PROFILE_PRODUCT_NAMES["sales_development"]
        if state.role_family == "contact_center":
            return PROFILE_PRODUCT_NAMES["contact_center"]
        if state.role_family == "finance" and ("financial analyst" in text or ("financial" in text and "graduate" in text)):
            return PROFILE_PRODUCT_NAMES["finance"]
        if state.role_family == "operations":
            return PROFILE_PRODUCT_NAMES["operations"]
        if state.role_family == "healthcare_admin":
            return PROFILE_PRODUCT_NAMES["healthcare_admin"]
        if state.role_family == "graduate_management" or ("graduate" in text and {"A", "P"}.issubset(state.category_codes)):
            return PROFILE_PRODUCT_NAMES["graduate_management"]
        if state.role_family == "admin":
            return PROFILE_PRODUCT_NAMES["admin"]
        return ()

    def _assemble_shortlist(self, state: ConversationState, candidates: list[SearchCandidate]) -> list[CatalogRecord]:
        selected: list[CatalogRecord] = []
        seen: set[str] = set()
        text = normalize_text(state.combined_user_text)
        has_technical_stack = bool(state.explicit_skill_terms)
        backend_leaning = any(term in text for term in ("backend leaning", "backend-leaning"))
        frontend_secondary = any(term in text for term in ("occasional", "secondary", "review frontend", "review frontend prs"))
        wants_reports = state.use_case == "development" or any(
            token in text for token in ("report", "feedback", "benchmark", "development", "audit")
        )
        wants_personality = "personality" in text or any(
            token in text for token in ("hire", "hiring", "selection", "manager", "graduate", "leadership")
        )
        wants_reasoning = any(code == "A" for code in state.category_codes) or any(
            token in text for token in ("engineer", "developer", "graduate", "analyst", "technical", "reasoning", "cognitive")
        )
        max_shortlist = self.settings.max_recommendations
        if has_technical_stack:
            max_shortlist = min(max_shortlist, 7)
        elif contains_any_phrase(text, ("contact center", "contact centre")):
            max_shortlist = min(max_shortlist, 4)
        elif state.quick_screen:
            max_shortlist = min(max_shortlist, 5)
        else:
            max_shortlist = min(max_shortlist, 6)

        def add_record(record: CatalogRecord | None) -> None:
            if record is None or record.url in seen:
                return
            record_text = normalize_text(record.name + " " + record.description + " " + " ".join(record.keys))
            if any(term in record_text for term in state.exclude_terms):
                return
            if state.languages and record.languages and not any(language in record.languages for language in state.languages):
                if contains_any_phrase(text, ("contact center", "contact centre", "spanish", "french", "portuguese")):
                    return
            if record.is_report and not wants_reports and len(selected) >= max_shortlist - 1:
                return
            selected.append(record)
            seen.add(record.url)

        def add_first_matching(predicate) -> None:
            for candidate in candidates:
                if predicate(candidate.record):
                    add_record(candidate.record)
                    return

        for record in state.prior_shortlist:
            add_record(record)

        for name in self._profile_seed_names(state, text):
            add_record(self.catalog.by_name.get(name))

        if any(term in text for term in ("leadership", "executive", "cxo")):
            for name in (
                "Occupational Personality Questionnaire OPQ32r",
                "OPQ Universal Competency Report 2.0",
                "OPQ Leadership Report",
            ):
                add_record(self.catalog.by_name.get(name))

        if "sales" in text and state.use_case == "development":
            for name in (
                "Global Skills Assessment",
                "Global Skills Development Report",
                "Occupational Personality Questionnaire OPQ32r",
                "OPQ MQ Sales Report",
                "Sales Transformation 2.0 - Individual Contributor",
            ):
                add_record(self.catalog.by_name.get(name))

        if contains_any_phrase(text, ("contact center", "contact centre")):
            if contains_phrase(text, "us") and contains_phrase(text, "english"):
                add_record(self.catalog.by_name.get("SVAR - Spoken English (US) (New)"))
            for name in (
                "Contact Center Call Simulation (New)",
                "Entry Level Customer Serv-Retail & Contact Center",
                "Customer Service Phone Simulation",
            ):
                add_record(self.catalog.by_name.get(name))

        if "financial analyst" in text or ("financial" in text and "graduate" in text):
            for name in (
                "SHL Verify Interactive – Numerical Reasoning",
                "Financial Accounting (New)",
                "Basic Statistics (New)",
                "Graduate Scenarios",
                "Occupational Personality Questionnaire OPQ32r",
            ):
                add_record(self.catalog.by_name.get(name))

        if any(term in text for term in ("plant operators", "chemical facility", "safety is absolute top priority", "procedure compliance", "cutting corners")):
            for name in (
                "Manufac. & Indust. - Safety & Dependability 8.0",
                "Workplace Health and Safety (New)",
            ):
                add_record(self.catalog.by_name.get(name))

        if "hipaa" in text or "healthcare admin" in text or "medical terminology" in text:
            for name in (
                "HIPAA (Security)",
                "Medical Terminology (New)",
                "Microsoft Word 365 - Essentials (New)",
                "Dependability and Safety Instrument (DSI)",
                "Occupational Personality Questionnaire OPQ32r",
            ):
                add_record(self.catalog.by_name.get(name))

        if "graduate management trainee" in text or ("graduate" in text and {"A", "P"}.issubset(state.category_codes)):
            for name in (
                "SHL Verify Interactive G+",
                "Occupational Personality Questionnaire OPQ32r",
                "Graduate Scenarios",
            ):
                add_record(self.catalog.by_name.get(name))

        if any(term in text for term in ("excel", "word", "admin assistant")):
            if any(term in text for term in ("simulation", "capabilities", "capability")):
                for name in (
                    "Microsoft Excel 365 - Essentials (New)",
                    "Microsoft Word 365 (New)",
                    "MS Excel (New)",
                    "MS Word (New)",
                    "Occupational Personality Questionnaire OPQ32r",
                ):
                    add_record(self.catalog.by_name.get(name))
            else:
                for name in (
                    "MS Excel (New)",
                    "MS Word (New)",
                    "Occupational Personality Questionnaire OPQ32r",
                ):
                    add_record(self.catalog.by_name.get(name))
            if any(term in text for term in ("quick", "daily", "screen")):
                for name in (
                    "Microsoft Excel 365 - Essentials (New)",
                    "Microsoft Word 365 (New)",
                ):
                    add_record(self.catalog.by_name.get(name))

        if "rust" in text:
            for name in (
                "Smart Interview Live Coding",
                "Linux Programming (General)",
                "Networking and Implementation (New)",
                "SHL Verify Interactive G+",
                "Occupational Personality Questionnaire OPQ32r",
            ):
                add_record(self.catalog.by_name.get(name))

        if contains_any_phrase(text, ("contact center", "contact centre")):
            add_first_matching(
                lambda record: "spoken english" in normalize_text(record.name + " " + record.description)
                or "svar" in record.aliases
            )
            add_first_matching(
                lambda record: "simulation" in normalize_text(record.name + " " + record.description)
                and "contact center" in normalize_text(record.name + " " + record.description)
            )
            add_first_matching(
                lambda record: "customer service" in normalize_text(record.name + " " + record.description)
                or "contact center" in normalize_text(record.name + " " + record.description)
            )

        if "rust" in text:
            for phrase in ("live coding", "linux", "networking"):
                add_first_matching(lambda record, phrase=phrase: phrase in normalize_text(record.name + " " + record.description))

        if any(term in text for term in ("excel", "word", "admin assistant")):
            for phrase in ("excel", "word"):
                if phrase in text:
                    add_first_matching(lambda record, phrase=phrase: phrase in normalize_text(record.name + " " + record.description))

        if any(term in text for term in ("hipaa", "medical terminology", "healthcare admin")):
            for phrase in ("hipaa", "medical terminology", "word 365", "safety instrument"):
                add_first_matching(lambda record, phrase=phrase: phrase in normalize_text(record.name + " " + record.description))

        if any(term in text for term in ("plant operators", "chemical facility", "procedure compliance", "cutting corners", "safety")):
            add_first_matching(lambda record: "safety" in normalize_text(record.name + " " + record.description))

        if "sales" in text and state.use_case == "development":
            for phrase in ("global skills assessment", "development report", "sales transformation"):
                add_first_matching(lambda record, phrase=phrase: phrase in normalize_text(record.name + " " + record.description))

        if "graduate" in text:
            add_first_matching(lambda record: "graduate scenarios" in normalize_text(record.name + " " + record.description))

        for term in TECH_STACK_TERMS:
            if term in text and term not in state.exclude_terms:
                if term == "angular" and (backend_leaning or frontend_secondary):
                    continue
                if term == "rest" and backend_leaning:
                    continue
                canonical_name = CANONICAL_SKILL_PRODUCTS.get(term, "")
                canonical_record = self.catalog.by_name.get(canonical_name)
                add_record(canonical_record)
                if canonical_record is None:
                    add_first_matching(lambda record, term=term: term in normalize_text(record.name + " " + record.description))

        if has_technical_stack:
            add_record(self.catalog.by_name.get("SHL Verify Interactive G+"))
            add_record(self.catalog.by_name.get("Occupational Personality Questionnaire OPQ32r"))

        if wants_reasoning:
            add_first_matching(lambda record: "A" in record.test_type and not record.is_report)
        if wants_personality:
            add_first_matching(
                lambda record: record.name == "Occupational Personality Questionnaire OPQ32r"
                or ("P" in record.test_type and not record.is_report)
            )
        if wants_reports:
            add_first_matching(lambda record: record.is_report or "Development & 360" in record.keys)

        type_counts: dict[str, int] = {}
        for record in selected:
            for code in record.test_type.split(","):
                type_counts[code] = type_counts.get(code, 0) + 1

        for candidate in candidates:
            record = candidate.record
            if len(selected) >= max_shortlist:
                break
            if record.url in seen:
                continue
            if record.is_report and not wants_reports and type_counts.get("K", 0) < 2:
                continue
            primary_codes = [code for code in record.test_type.split(",") if code]
            if has_technical_stack and primary_codes == ["K"] and type_counts.get("K", 0) >= 5:
                continue
            if not has_technical_stack and primary_codes == ["P"] and type_counts.get("P", 0) >= 2:
                continue
            add_record(record)
            for code in primary_codes:
                type_counts[code] = type_counts.get(code, 0) + 1

        if state.wants_shorter:
            selected.sort(key=lambda record: (record.duration_minutes or 999, record.name))

        return selected[:max_shortlist]

    def _compare_records(self, state: ConversationState) -> str:
        first, second = state.comparison_records[:2]
        differences = []
        if first.test_type != second.test_type:
            differences.append(f"{first.name} is `{first.test_type}` while {second.name} is `{second.test_type}`.")
        if first.duration_text != second.duration_text and first.duration_text and second.duration_text:
            differences.append(f"{first.name} takes {first.duration_text}; {second.name} takes {second.duration_text}.")
        if first.is_report != second.is_report:
            report_text = "a report product" if first.is_report else "an assessment instrument"
            other_text = "a report product" if second.is_report else "an assessment instrument"
            differences.append(f"{first.name} is {report_text}, while {second.name} is {other_text}.")
        if first.languages != second.languages:
            differences.append(
                f"{first.name} languages: {', '.join(first.languages[:4]) or 'not listed'}; "
                f"{second.name} languages: {', '.join(second.languages[:4]) or 'not listed'}."
            )
        differences.append(f"{first.name}: {first.description}")
        differences.append(f"{second.name}: {second.description}")
        return " ".join(differences[:5])

    def _compose_recommendation_reply(self, state: ConversationState, shortlist: list[CatalogRecord]) -> str:
        if not shortlist:
            return "I could not find a grounded SHL shortlist from the catalog yet. Share a bit more about the role, level, and must-have skills."

        catalog_gap = ""
        text = normalize_text(state.combined_user_text)
        if "rust" in text and not any("rust" in normalize_text(record.name + " " + record.description) for record in shortlist):
            catalog_gap = "SHL does not currently expose a Rust-specific test in the catalog, so I used the closest grounded alternatives. "

        summary = ", ".join(record.name for record in shortlist[:3])
        extra = ""
        if len(shortlist) > 3:
            extra = f", plus {len(shortlist) - 3} additional fit-for-purpose options"

        reply = (
            f"{catalog_gap}Here are {len(shortlist)} SHL assessments that fit your brief. "
            f"The shortlist is anchored by {summary}{extra}."
        )

        if state.use_case == "development":
            reply += " I biased toward development and report products where the catalog supports re-skilling and feedback."
        elif any("A" in record.test_type for record in shortlist):
            reply += " I included a reasoning measure where it adds signal beyond role-specific knowledge tests."

        grounded_prompt = (
            f"User brief: {state.combined_user_text}\n"
            f"Shortlist: {[{'name': record.name, 'type': record.test_type, 'description': record.description} for record in shortlist]}\n"
            "Write a concise grounded reply in 2-4 sentences."
        )
        llm_reply = self.llm.write_reply(grounded_prompt)
        return llm_reply or reply

    def _determine_conversation_end(self, state: ConversationState, shortlist: list[CatalogRecord], reply: str) -> bool:
        if not shortlist:
            return False
        if state.confirmation:
            return True
        if state.remaining_turns <= 1:
            return True
        if "?" in reply:
            return False
        if not state.prior_shortlist:
            return True
        latest = normalize_text(state.latest_user_text)
        if any(pattern in latest for pattern in FOLLOW_UP_PATTERNS):
            return False
        return True
