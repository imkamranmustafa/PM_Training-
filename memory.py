# memory.py
import time
import uuid
from typing import List, Dict, Any
from vector_store import InMemoryVectorStore

class MemoryAgent:
    """
    Manages:
     - conversation memory (history)
     - knowledge base (structured records with provenance)
     - agent state memory (what each agent did per task)
     - provides keyword/topic search + vector similarity via InMemoryVectorStore
    """
    def __init__(self, name="MemoryAgent", dim=128):
        self.name = name
        self.conversation = []  # list of messages with ts, role, text
        self.knowledge = {}  # id -> record
        self.agent_state = {}  # task_id -> agent notes
        self.vec_store = InMemoryVectorStore(dim=dim)

    def log_conversation(self, role: str, text: str):
        rec = {"ts": time.time(), "role": role, "text": text}
        self.conversation.append(rec)
        return rec

    def store_knowledge(self, topic: str, content: str, source: str, agent: str, confidence: float=0.8):
        id = str(uuid.uuid4())
        metadata = {"topic": topic, "source": source, "agent": agent, "confidence": confidence}
        record = {"id": id, "topic": topic, "content": content, "source": source, "agent": agent, "confidence": confidence, "ts": time.time()}
        self.knowledge[id] = record
        # also add to vector store
        self.vec_store.add(id, content, metadata)
        return record

    def retrieve_by_topic(self, topic_keyword: str, top_k=5):
        # keyword scan on metadata topic
        matched = [r for r in self.knowledge.values() if topic_keyword.lower() in r["topic"].lower() or topic_keyword.lower() in r["content"].lower()]
        # vector similarity
        vec_results = self.vec_store.query(topic_keyword, top_k=top_k)
        return {"keyword_matches": matched, "vector_matches": vec_results}

    def store_agent_state(self, task_id: str, agent: str, note: str):
        ts = time.time()
        if task_id not in self.agent_state:
            self.agent_state[task_id] = []
        self.agent_state[task_id].append({"ts": ts, "agent": agent, "note": note})
        return {"task_id": task_id, "agent": agent, "note": note, "ts": ts}
