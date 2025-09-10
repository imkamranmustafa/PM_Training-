# agents.py
import time
import random
from typing import List, Dict, Any

class BaseAgent:
    def __init__(self, name: str, coordinator=None):
        self.name = name
        self.coordinator = coordinator

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{self.name}] {msg}")

class ResearchAgent(BaseAgent):
    """
    Simulates information retrieval using a pre-loaded knowledge base (a simple dict).
    """
    def __init__(self, name="ResearchAgent", coordinator=None, knowledge_base: Dict[str, List[Dict]] = None):
        super().__init__(name, coordinator)
        # knowledge_base: topic -> list of fact dicts {title, content, source}
        self.kb = knowledge_base or self._default_kb()

    def _default_kb(self):
        return {
            "neural networks": [
                {"title": "NN Types", "content": "Feedforward, CNN, RNN, LSTM, Transformer", "source": "kb:nn_overview"},
                {"title": "CNN", "content": "Convolutional Neural Networks are good for images", "source": "kb:cnn"}
            ],
            "transformers": [
                {"title": "Transformer Overview", "content": "Self-attention based models; BERT, GPT", "source": "kb:transformer"},
                {"title": "Efficiency", "content": "Transformers scale well with GPUs but can be compute-heavy", "source": "kb:transformer_eff"}
            ],
            "reinforcement learning": [
                {"title": "RL challenges", "content": "Sample efficiency, reward design, stability", "source": "kb:rl_challenges"},
            ],
            "optimizers": [
                {"title": "SGD", "content": "Stochastic Gradient Descent with/without momentum", "source": "kb:sgd"},
                {"title": "Adam", "content": "Adaptive optimizer combining momentum and RMSprop ideas", "source": "kb:adam"}
            ],
            "papers:rl:recent": [
                {"title": "RL Paper A 2023", "content": "Methodology: model-based RL with curiosity-driven exploration", "source": "arxiv:2023A"},
                {"title": "RL Paper B 2024", "content": "Methodology: offline RL with conservative objectives", "source": "arxiv:2024B"}
            ]
        }

    def search(self, query: str, top_k=5) -> Dict[str, Any]:
        """Return simulated search results and a confidence score"""
        self.log(f"Searching KB for '{query}'")
        results = []
        query_l = query.lower()
        for key, facts in self.kb.items():
            if query_l in key or any(query_l in f["content"].lower() or query_l in f["title"].lower() for f in facts):
                for f in facts:
                    results.append({
                        "topic": key,
                        "title": f["title"],
                        "content": f["content"],
                        "source": f["source"],
                        "confidence": round(random.uniform(0.6, 0.95), 2)
                    })
        if not results:
            # fallback: return nearest keys by token overlap
            for key, facts in self.kb.items():
                overlap = set(query_l.split()).intersection(set(key.split()))
                if overlap:
                    for f in facts:
                        results.append({
                            "topic": key,
                            "title": f["title"],
                            "content": f["content"],
                            "source": f["source"],
                            "confidence": 0.5
                        })
        # sort by confidence
        results = sorted(results, key=lambda r: r["confidence"], reverse=True)[:top_k]
        return {"agent": self.name, "query": query, "results": results, "confidence": round(sum(r["confidence"] for r in results)/max(1,len(results)), 2)}

class AnalysisAgent(BaseAgent):
    """
    Perform comparisons, simple calculations, reasoning.
    """
    def __init__(self, name="AnalysisAgent", coordinator=None):
        super().__init__(name, coordinator)

    def compare_optimizers(self, items: List[Dict]) -> Dict[str, Any]:
        """
        Example comparator: takes items from research agent (each with title/content)
        and returns a simple scored comparison.
        """
        self.log("Comparing optimizers...")
        scores = {}
        for item in items:
            title = item.get("title", "unknown")
            content = item.get("content", "")
            # heuristic scoring
            score = 0.5
            if "adaptive" in content.lower() or "adam" in title.lower():
                score += 0.3
            if "stochastic" in content.lower() or "sgd" in title.lower():
                score += 0.1
            # penalize vague
            if len(content) < 20:
                score -= 0.1
            scores[title] = round(min(max(score, 0.0), 1.0), 2)
        # produce ranking
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        confidence = round(sum(scores.values())/max(1,len(scores)), 2)
        return {"agent": self.name, "scores": scores, "ranking": ranking, "confidence": confidence}

    def analyze_transformer_efficiency(self, facts: List[Dict]) -> Dict[str, Any]:
        self.log("Analyzing transformer efficiency...")
        impacts = []
        for f in facts:
            txt = f.get("content", "").lower()
            reason = "unknown"
            if "compute" in txt or "heavy" in txt:
                reason = "High compute; good parallelism"
            elif "scale" in txt:
                reason = "Scales with data and hardware"
            else:
                reason = "Mixed evidence"
            impacts.append({"title": f.get("title"), "reason": reason, "confidence": 0.8})
        confidence = round(sum(i["confidence"] for i in impacts)/max(1,len(impacts)), 2)
        return {"agent": self.name, "impacts": impacts, "confidence": confidence}
