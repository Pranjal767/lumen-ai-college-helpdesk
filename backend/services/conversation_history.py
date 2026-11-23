"""Persistence for anonymous conversations and their messages."""

from uuid import uuid4

from backend.database.db import get_connection, initialize_database
from backend.database.models import (
    get_conversation as get_conversation_record,
    insert_conversation,
    insert_message,
    list_conversations as list_conversation_records,
    list_messages,
    update_conversation_activity,
)

MAX_TITLE_LENGTH = 100


class ConversationNotFoundError(LookupError):
    """Raised when a conversation does not belong to the requested session."""


def save_exchange(
    session_id: str,
    conversation_id: str | None,
    question: str,
    answer: str,
) -> str:
    """Store a question and answer, creating a conversation when needed."""
    initialize_database()
    with get_connection() as connection:
        if conversation_id is None:
            conversation_id = uuid4().hex
            insert_conversation(
                connection,
                conversation_id=conversation_id,
                session_id=session_id,
                title=question[:MAX_TITLE_LENGTH],
            )
        elif get_conversation_record(
            connection,
            conversation_id=conversation_id,
            session_id=session_id,
        ) is None:
            raise ConversationNotFoundError

        insert_message(
            connection,
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        insert_message(
            connection,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )
        update_conversation_activity(connection, conversation_id)

    return conversation_id


def list_conversations(session_id: str) -> list[dict[str, object]]:
    """List conversations owned by an anonymous session."""
    initialize_database()
    with get_connection() as connection:
        rows = list_conversation_records(connection, session_id)
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]


def get_conversation(
    session_id: str, conversation_id: str
) -> dict[str, object] | None:
    """Load a conversation and its messages, scoped to its anonymous session."""
    initialize_database()
    with get_connection() as connection:
        conversation = get_conversation_record(
            connection,
            conversation_id=conversation_id,
            session_id=session_id,
        )
        if conversation is None:
            return None

        messages = list_messages(connection, conversation_id)
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": [
                {
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                }
                for message in messages
            ],
        }
