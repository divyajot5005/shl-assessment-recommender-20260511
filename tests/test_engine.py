from app.config import Settings, get_settings
from app.engine import SHLAgentService
from app.schemas import ChatMessage


service = SHLAgentService(get_settings())


def test_vague_request_clarifies() -> None:
    result = service.chat([ChatMessage(role="user", content="I need an assessment.")])
    assert result.recommendations == []
    assert "?" in result.reply


def test_legal_request_refuses() -> None:
    result = service.chat(
        [ChatMessage(role="user", content="Are we legally required under HIPAA to test everyone?")]
    )
    assert result.recommendations == []
    assert "legal" in result.reply.lower()


def test_java_role_recommends_catalog_items() -> None:
    result = service.chat(
        [
            ChatMessage(
                role="user",
                content="Hiring a senior Java backend engineer with Spring, SQL, AWS and Docker. Recommend a shortlist.",
            )
        ]
    )
    names = {item.name for item in result.recommendations}
    assert "Core Java (Advanced Level) (New)" in names
    assert "Spring (New)" in names
    assert "SQL (New)" in names


def test_general_hiring_advice_refuses() -> None:
    result = service.chat([ChatMessage(role="user", content="What interview process should I use for a sales hire?")])
    assert result.recommendations == []
    assert "assessment" in result.reply.lower()


def test_compare_uses_explicit_records() -> None:
    result = service.chat([ChatMessage(role="user", content="Compare OPQ with Java 8 (New).")])
    assert result.recommendations == []
    assert "Occupational Personality Questionnaire OPQ32r" in result.reply
    assert "Java 8 (New)" in result.reply
    assert "Count Out The Money" not in result.reply


def test_contact_center_english_prompts_for_variant_before_recommending() -> None:
    result = service.chat(
        [
            ChatMessage(
                role="user",
                content="We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
            ),
            ChatMessage(role="assistant", content="What language do callers use? That determines which SHL spoken-language screen fits."),
            ChatMessage(role="user", content="English."),
        ]
    )
    assert result.recommendations == []
    assert "Which English variant" in result.reply


def test_sales_development_shortlist_stays_grounded() -> None:
    result = service.chat(
        [
            ChatMessage(
                role="user",
                content="We need to reskill sales reps and want assessments plus development reports for coaching.",
            )
        ]
    )
    names = {item.name for item in result.recommendations}
    assert "Global Skills Assessment" in names
    assert "Global Skills Development Report" in names
    assert "Sales Transformation 2.0 - Individual Contributor" in names
    assert "RESTful Web Services (New)" not in names


def test_hosted_state_extraction_is_disabled_by_default(monkeypatch) -> None:
    hosted_settings = Settings(llm_api_key="test-key")
    hosted_service = SHLAgentService(hosted_settings)

    def fail_if_called(_messages):
        raise AssertionError("hosted state extraction should not be used for shortlist state by default")

    monkeypatch.setattr(type(hosted_service.llm), "extract_state", lambda self, messages: fail_if_called(messages))
    result = hosted_service.chat(
        [
            ChatMessage(
                role="user",
                content="Hiring a senior Java backend engineer with Spring, SQL, AWS and Docker. Recommend a shortlist.",
            )
        ]
    )
    names = {item.name for item in result.recommendations}
    assert "Core Java (Advanced Level) (New)" in names
    assert "Spring (New)" in names


def test_full_stack_role_clarifies_orientation() -> None:
    result = service.chat(
        [
            ChatMessage(
                role="user",
                content="Hiring a full stack engineer with Java, Spring, SQL, AWS and Angular. Recommend a shortlist.",
            )
        ]
    )
    assert result.recommendations == []
    assert "backend-leaning" in result.reply
