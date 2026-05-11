from app.config import get_settings
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
