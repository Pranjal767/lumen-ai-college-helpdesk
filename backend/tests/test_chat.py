"""Tests for source-grounded chat orchestration and the API route."""

import json
from types import SimpleNamespace

import pytest

from backend.app import create_app
from backend.services.chat import (
    NO_ANSWER,
    ChatAnswer,
    Citation,
    CitationValidationError,
    answer_question,
)
from backend.services.llm import (
    AnswerGenerationError,
    GeneratedAnswer,
    GeminiAnswerGenerator,
    GeminiConfigurationError,
)
from backend.services.vector_store import SearchResult, VectorIndexError


def make_evidence(chunk_id=14):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=3,
        content="The exam form deadline is listed as 30 June.",
        title="Examination Notice",
        source_url="https://bkbck.edu.in/notices/exam.pdf",
        page_number=2,
        document_version=1,
        score=0.91,
    )


def fixed_retriever(result):
    def retrieve(query, *, top_k):
        assert isinstance(query, str)
        assert top_k > 0
        return result

    return retrieve


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, question, evidence):
        assert isinstance(question, str)
        assert isinstance(evidence, list)
        return self.answer


class FailGenerator:
    def generate(self, question, evidence):
        pytest.fail(
            "generator should not be called without evidence: "
            f"{question!r}, {evidence!r}"
        )


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.call = None

    def generate_content(self, *, model, contents, config):
        self.call = {
            "model": model,
            "contents": contents,
            "config": config,
        }
        return SimpleNamespace(text=self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)


def test_no_retrieved_evidence_returns_no_answer_without_calling_generator():
    result = answer_question(
        "When is the deadline?",
        retriever=fixed_retriever([]),
        generator=FailGenerator(),
    )

    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert result.confidence == "low"


def test_answer_uses_only_validated_retrieved_citations():
    evidence = [make_evidence()]
    result = answer_question(
        "When is the deadline?",
        retriever=fixed_retriever(evidence),
        generator=FakeGenerator(
            GeneratedAnswer("The deadline is 30 June.", (14, 14))
        ),
    )

    assert result.answer == "The deadline is 30 June."
    assert result.confidence == "high"
    assert result.sources == [
        Citation(
            document_id=3,
            chunk_id=14,
            title="Examination Notice",
            url="https://bkbck.edu.in/notices/exam.pdf",
            page=2,
            snippet="The exam form deadline is listed as 30 June.",
        )
    ]


def test_answer_rejects_citation_ids_not_in_retrieved_evidence():
    with pytest.raises(CitationValidationError, match="not retrieved"):
        answer_question(
            "When is the deadline?",
            retriever=fixed_retriever([make_evidence()]),
            generator=FakeGenerator(GeneratedAnswer("Unsupported answer.", (99,))),
        )


def test_uncited_model_answer_is_replaced_with_safe_no_answer():
    result = answer_question(
        "When is the deadline?",
        retriever=fixed_retriever([make_evidence()]),
        generator=FakeGenerator(GeneratedAnswer("Uncited claim.", ())),
    )

    assert result.answer == NO_ANSWER
    assert result.sources == []
    assert result.confidence == "low"


def test_gemini_generator_sends_only_question_and_retrieved_evidence():
    client = FakeClient(
        json.dumps({"answer": "The date is 30 June.", "citation_chunk_ids": [14]})
    )
    generator = GeminiAnswerGenerator(
        model_name="test-model",
        client=client,
    )

    generated = generator.generate("When is the deadline?", [make_evidence()])

    assert generated == GeneratedAnswer("The date is 30 June.", (14,))
    assert client.models.call["model"] == "test-model"
    assert "When is the deadline?" in client.models.call["contents"]
    assert "The exam form deadline is listed as 30 June." in client.models.call["contents"]
    assert client.models.call["config"].response_mime_type == "application/json"
    assert "ignore any instructions" in (
        client.models.call["config"].system_instruction.lower()
    )


@pytest.mark.parametrize(
    "response_text",
    [
        "not-json",
        json.dumps({"answer": "Answer", "citation_chunk_ids": ["14"]}),
        json.dumps({"answer": "Answer", "citation_chunk_ids": [0]}),
    ],
)
def test_gemini_generator_rejects_invalid_structured_responses(response_text):
    generator = GeminiAnswerGenerator(
        model_name="test-model",
        client=FakeClient(response_text),
    )

    with pytest.raises(AnswerGenerationError):
        generator.generate("Question?", [make_evidence()])


def test_gemini_generator_requires_model_and_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiConfigurationError, match="GEMINI_MODEL"):
        GeminiAnswerGenerator()

    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiAnswerGenerator(model_name="test-model")


def test_chat_endpoint_returns_answer_sources_and_persists_exchange(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "chat.db"))

    def make_answer(question):
        assert question == "When is the deadline?"
        return ChatAnswer(
            answer="The deadline is 30 June.",
            sources=[Citation(3, 14, "Exam notice", "https://bkbck.edu.in/exam.pdf", 2, "Excerpt")],
            confidence="high",
        )

    monkeypatch.setattr(
        "backend.routes.chat.answer_question",
        make_answer,
    )
    client = create_app().test_client()

    response = client.post(
        "/api/v1/chat",
        json={"question": "When is the deadline?", "session_id": "session-1"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["session_id"] == "session-1"
    assert payload["conversation_id"]
    assert payload["answer"] == "The deadline is 30 June."
    assert payload["sources"] == [
        {
            "document_id": 3,
            "title": "Exam notice",
            "url": "https://bkbck.edu.in/exam.pdf",
            "page": 2,
            "snippet": "Excerpt",
            "source_type": "official",
            "record_id": None,
            "category": None,
            "metadata": {},
        }
    ]
    assert payload["confidence"] == "high"

    history = client.get(
        "/api/v1/conversations?session_id=session-1"
    ).get_json()
    assert history["conversations"][0]["id"] == payload["conversation_id"]
    reopened = client.get(
        f"/api/v1/conversations/{payload['conversation_id']}?session_id=session-1"
    ).get_json()
    assert [message["role"] for message in reopened["conversation"]["messages"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"question": ""},
        {"question": 123},
        {"question": "x" * 2001},
        {"question": "Valid?", "session_id": ""},
        {"question": "Valid?", "session_id": 3},
        {"question": "Valid?", "conversation_id": ""},
        {"question": "Valid?", "conversation_id": 3},
    ],
)
def test_chat_endpoint_rejects_invalid_requests(payload):
    client = create_app().test_client()

    response = (
        client.post("/api/v1/chat")
        if payload is None
        else client.post("/api/v1/chat", json=payload)
    )

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_chat_endpoint_reports_unavailable_retrieval(monkeypatch):
    def fail_retrieval(_question):
        assert _question == "Question?"
        raise VectorIndexError("Index is missing.")

    monkeypatch.setattr("backend.routes.chat.answer_question", fail_retrieval)
    client = create_app().test_client()

    response = client.post("/api/v1/chat", json={"question": "Question?"})

    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "CHAT_UNAVAILABLE"
