"""Student chat API endpoint."""

import sqlite3
from uuid import uuid4

from flask import Blueprint, jsonify, request

from backend.services.chat import (
    MAX_QUESTION_LENGTH,
    CitationValidationError,
    answer_question,
)
from backend.services.conversation_history import (
    ConversationNotFoundError,
    get_conversation,
    list_conversations,
    save_exchange,
)
from backend.services.embeddings import EmbeddingError
from backend.services.llm import AnswerGenerationError, GeminiConfigurationError
from backend.services.vector_store import VectorIndexError

chat_blueprint = Blueprint("chat", __name__)
MAX_SESSION_ID_LENGTH = 128


@chat_blueprint.post("/chat")
def post_chat():
    """Return a source-grounded answer for a student's question."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error_response(
            "INVALID_REQUEST",
            "Send a JSON object containing a question.",
            400,
        )

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return _error_response(
            "INVALID_QUESTION",
            "Question must be a non-empty string.",
            400,
        )
    if len(question.strip()) > MAX_QUESTION_LENGTH:
        return _error_response(
            "INVALID_QUESTION",
            f"Question must be at most {MAX_QUESTION_LENGTH} characters.",
            400,
        )

    session_id = payload.get("session_id")
    if session_id is not None and (
        not isinstance(session_id, str)
        or not session_id.strip()
        or len(session_id) > MAX_SESSION_ID_LENGTH
    ):
        return _error_response(
            "INVALID_SESSION_ID",
            "Session ID must be a non-empty string of at most 128 characters.",
            400,
        )
    session_id = session_id.strip() if isinstance(session_id, str) else uuid4().hex

    conversation_id = payload.get("conversation_id")
    if conversation_id is not None and (
        not isinstance(conversation_id, str)
        or not conversation_id.strip()
        or len(conversation_id) > MAX_SESSION_ID_LENGTH
    ):
        return _error_response(
            "INVALID_CONVERSATION_ID",
            "Conversation ID must be a non-empty string of at most 128 characters.",
            400,
        )
    conversation_id = (
        conversation_id.strip() if isinstance(conversation_id, str) else None
    )

    if conversation_id is not None:
        try:
            if get_conversation(session_id, conversation_id) is None:
                return _error_response(
                    "CONVERSATION_NOT_FOUND",
                    "Conversation was not found for this session.",
                    404,
                )
        except (OSError, sqlite3.Error):
            return _error_response(
                "HISTORY_UNAVAILABLE",
                "Conversation history is temporarily unavailable.",
                503,
            )

    try:
        result = answer_question(question)
    except (VectorIndexError, EmbeddingError, GeminiConfigurationError) as error:
        return _error_response("CHAT_UNAVAILABLE", str(error), 503)
    except (AnswerGenerationError, CitationValidationError) as error:
        return _error_response("ANSWER_GENERATION_FAILED", str(error), 502)

    try:
        conversation_id = save_exchange(
            session_id,
            conversation_id,
            question.strip(),
            result.answer,
        )
    except ConversationNotFoundError:
        return _error_response(
            "CONVERSATION_NOT_FOUND",
            "Conversation was not found for this session.",
            404,
        )
    except (OSError, sqlite3.Error):
        return _error_response(
            "HISTORY_UNAVAILABLE",
            "The answer was generated but could not be saved to conversation history.",
            503,
        )

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "answer": result.answer,
            "sources": [
                {
                    "document_id": source.document_id,
                    "title": source.title,
                    "url": source.url,
                    "page": source.page,
                    "snippet": source.snippet,
                    "source_type": source.source_type,
                    "record_id": source.record_id,
                    "category": source.category,
                    "metadata": source.metadata,
                }
                for source in result.sources
            ],
            "confidence": result.confidence,
        }
    )


@chat_blueprint.get("/conversations")
def get_conversations():
    """List conversations belonging to an anonymous browser session."""
    session_id = request.args.get("session_id", "").strip()
    if not session_id or len(session_id) > MAX_SESSION_ID_LENGTH:
        return _error_response(
            "INVALID_SESSION_ID",
            "A session_id query parameter of at most 128 characters is required.",
            400,
        )

    try:
        conversations = list_conversations(session_id)
    except (OSError, sqlite3.Error):
        return _error_response(
            "HISTORY_UNAVAILABLE",
            "Conversation history is temporarily unavailable.",
            503,
        )

    return jsonify({"success": True, "conversations": conversations})


@chat_blueprint.get("/conversations/<conversation_id>")
def get_conversation_details(conversation_id: str):
    """Load one conversation belonging to an anonymous browser session."""
    session_id = request.args.get("session_id", "").strip()
    if not session_id or len(session_id) > MAX_SESSION_ID_LENGTH:
        return _error_response(
            "INVALID_SESSION_ID",
            "A session_id query parameter of at most 128 characters is required.",
            400,
        )
    if not conversation_id or len(conversation_id) > MAX_SESSION_ID_LENGTH:
        return _error_response(
            "INVALID_CONVERSATION_ID",
            "Conversation ID must be at most 128 characters.",
            400,
        )

    try:
        conversation = get_conversation(session_id, conversation_id)
    except sqlite3.Error:
        return _error_response(
            "HISTORY_UNAVAILABLE",
            "Conversation history is temporarily unavailable.",
            503,
        )
    if conversation is None:
        return _error_response(
            "CONVERSATION_NOT_FOUND",
            "Conversation was not found for this session.",
            404,
        )

    return jsonify({"success": True, "conversation": conversation})


def _error_response(code: str, message: str, status_code: int):
    return (
        jsonify(
            {
                "success": False,
                "error": {"code": code, "message": message},
            }
        ),
        status_code,
    )
