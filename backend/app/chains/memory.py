"""
Conversation memory management.
Keeps a simple in-memory store of chat history per session_id so the chatbot
supports multi-turn conversational QA. For a production system this would be
backed by Redis/Postgres, but in-memory is fine for an internship demo.
"""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# session_id -> list of messages
_SESSION_STORE: dict[str, list[BaseMessage]] = {}

MAX_HISTORY_MESSAGES = 12  # keep last N messages (6 turns) to bound prompt size


def get_history(session_id: str) -> list[BaseMessage]:
    return _SESSION_STORE.get(session_id, [])


def add_turn(session_id: str, user_message: str, ai_message: str) -> None:
    """
    Append a user/AI message pair to the session history, trimming
    older messages beyond MAX_HISTORY_MESSAGES.
    """
    history = _SESSION_STORE.setdefault(session_id, [])
    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=ai_message))

    if len(history) > MAX_HISTORY_MESSAGES:
        _SESSION_STORE[session_id] = history[-MAX_HISTORY_MESSAGES:]


def clear_history(session_id: str) -> None:
    _SESSION_STORE.pop(session_id, None)


def format_history_for_prompt(session_id: str) -> str:
    """
    Formats the session's chat history into a readable transcript
    for injection into the prompt.
    """
    history = get_history(session_id)
    if not history:
        return "No prior conversation."

    lines = []
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)