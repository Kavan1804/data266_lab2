"""PLACEHOLDER so p1_rag.ipynb can run before Muhammed's memory module lands.

It keeps turns per conversation but does not resolve follow-ups, detect user
values or refuse permanent changes. src/p1_conversation.py replaces it (Task 1.3).
"""
from p1_engine import SessionFact, Turn  # noqa: F401


class Conversation:
    def __init__(self, conversation_id: str, engine, mode: str = "hybrid"):
        self.conversation_id, self.engine, self.mode = conversation_id, engine, mode
        self.history: list[Turn] = []
        self.session_facts: list[SessionFact] = []

    def answer(self, question: str) -> dict:
        out = self.engine.answer(question, self.history, self.session_facts, self.mode)
        self.history.append(Turn(question, question, out))
        return out
