"""Tests for anonymous conversation history."""

from backend.app import create_app
from backend.services.chat import ChatAnswer


def test_history_is_limited_to_the_owning_session(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.db"))
    monkeypatch.setattr(
        "backend.routes.chat.answer_question",
        lambda question: ChatAnswer(
            answer=f"Answer to: {question}",
            sources=[],
            confidence="low",
        ),
    )
    client = create_app().test_client()

    first = client.post(
        "/api/v1/chat",
        json={"question": "Where is the exam schedule?", "session_id": "session-a"},
    )
    assert first.status_code == 200
    conversation_id = first.get_json()["conversation_id"]

    second = client.post(
        "/api/v1/chat",
        json={
            "question": "And the fee schedule?",
            "session_id": "session-a",
            "conversation_id": conversation_id,
        },
    )
    assert second.status_code == 200

    listing = client.get("/api/v1/conversations?session_id=session-a")
    assert listing.status_code == 200
    assert len(listing.get_json()["conversations"]) == 1
    assert listing.get_json()["conversations"][0]["message_count"] == 4

    reopened = client.get(
        f"/api/v1/conversations/{conversation_id}?session_id=session-a"
    )
    assert reopened.status_code == 200
    assert [
        message["role"] for message in reopened.get_json()["conversation"]["messages"]
    ] == ["user", "assistant", "user", "assistant"]

    other_session_list = client.get("/api/v1/conversations?session_id=session-b")
    other_session_detail = client.get(
        f"/api/v1/conversations/{conversation_id}?session_id=session-b"
    )
    assert other_session_list.get_json()["conversations"] == []
    assert other_session_detail.status_code == 404


def test_conversation_endpoints_require_a_session_id():
    client = create_app().test_client()

    listing = client.get("/api/v1/conversations")
    detail = client.get("/api/v1/conversations/conversation-1")

    assert listing.status_code == 400
    assert detail.status_code == 400


def test_chat_reports_history_storage_failure(monkeypatch, tmp_path):
    database_parent = tmp_path / "not-a-directory"
    database_parent.write_text("blocker", encoding="utf-8")
    monkeypatch.setenv("DATABASE_PATH", str(database_parent / "chat.db"))
    monkeypatch.setattr(
        "backend.routes.chat.answer_question",
        lambda question: ChatAnswer(
            answer="A grounded answer.",
            sources=[],
            confidence="low",
        ),
    )
    client = create_app().test_client()

    response = client.post(
        "/api/v1/chat",
        json={"question": "Question?", "session_id": "session-a"},
    )

    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "HISTORY_UNAVAILABLE"
