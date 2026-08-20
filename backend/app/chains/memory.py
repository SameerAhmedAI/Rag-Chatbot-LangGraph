"""
Conversation memory management.

Previously a module-level dict (_SESSION_STORE) mutated by free functions —
that's global mutable state shared across every request the process
handles, with no encapsulation. Wrapping it in a class doesn't just
satisfy the OOP requirement: it makes the ownership of that state
explicit and gives it a name, instead of a bare dict floating at import
time that any code in the process could reach into directly.

Still in-memory (not Redis/Postgres) — that trade-off is unchanged and
is called out in README's Future Improvements section.
"""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class SessionMemory:
    """
    Keeps chat history per session_id so the chatbot supports multi-turn
    conversational QA. One instance is shared process-wide via the
    module-level `session_memory` object below (still a single in-memory
    store, but now it's an object with a defined interface instead of a
    bare dict any module could mutate directly).
    """

    MAX_HISTORY_MESSAGES = 12  # keep last N messages (6 turns) to bound prompt size

    def __init__(self):
        self._store: dict[str, list[BaseMessage]] = {}

    def get_history(self, session_id: str) -> list[BaseMessage]:
        return self._store.get(session_id, [])

    def add_turn(self, session_id: str, user_message: str, ai_message: str) -> None:
        """
        Append a user/AI message pair to the session history, trimming
        older messages beyond MAX_HISTORY_MESSAGES.
        """
        history = self._store.setdefault(session_id, [])
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=ai_message))

        if len(history) > self.MAX_HISTORY_MESSAGES:
            self._store[session_id] = history[-self.MAX_HISTORY_MESSAGES:]

    def clear_history(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def format_history_for_prompt(self, session_id: str) -> str:
        """Formats the session's chat history into a readable transcript for prompting."""
        history = self.get_history(session_id)
        if not history:
            return "No prior conversation."

        lines = []
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)


# Single shared instance for the process (equivalent scope to the old
# module-level dict, but now encapsulated behind SessionMemory's interface).
session_memory = SessionMemory()


# Module-level convenience wrappers so existing call sites don't all
# need to change their import/call shape in the same commit.
def get_history(session_id: str) -> list[BaseMessage]:
    return session_memory.get_history(session_id)


def add_turn(session_id: str, user_message: str, ai_message: str) -> None:
    session_memory.add_turn(session_id, user_message, ai_message)


def clear_history(session_id: str) -> None:
    session_memory.clear_history(session_id)


def format_history_for_prompt(session_id: str) -> str:
    return session_memory.format_history_for_prompt(session_id)