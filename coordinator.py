# coordinator.py
import time
import uuid
import traceback
from typing import Dict, Any, List

from agents import ResearchAgent, AnalysisAgent
from memory import MemoryAgent

class Coordinator:
    def __init__(self):
        self.memory = MemoryAgent()
        self.research = ResearchAgent(coordinator=self, knowledge_base=None)
        self.analysis = AnalysisAgent(coordinator=self)
        self.trace = []  # concise trace logs

    def log_trace(self, entry: Dict[str,Any]):
        entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.trace.append(entry)
        print(f"[TRACE] {entry}")

    def complexity_analysis(self, user_query: str) -> Dict[str,Any]:
        """Simple heuristics to decide which agents are required."""
        q = user_query.lower()
        need_research = any(tok in q for tok in ["find", "research", "recent", "papers", "what are", "types", "explain", "info", "information"])
        need_analysis = any(tok in q for tok in ["compare", "analyze", "best", "which", "recommend", "efficiency", "trade-off", "tradeoffs"])
        need_memory = any(tok in q for tok in ["what did we discuss", "what did we learn", "earlier", "remember", "recall"])
        plan = {"research": need_research, "analysis": need_analysis, "memory": need_memory}
        self.log_trace({"event":"complexity_analysis", "query": user_query, "plan": plan})
        return plan

    def handle(self, user_query: str) -> Dict[str,Any]:
        task_id = str(uuid.uuid4())[:8]
        self.memory.log_conversation("user", user_query)
        plan = self.complexity_analysis(user_query)
        final_answer = ""
        agent_outputs = {}
        try:
            # If memory query
            if plan["memory"] and (not plan["research"] and not plan["analysis"]):
                mem = self.memory.retrieve_by_topic(user_query)
                agent_outputs["memory_lookup"] = mem
                final_answer = self._format_memory_response(mem)
                confidence = 0.8
            else:
                # Research phase
                research_out = None
                if plan["research"]:
                    research_out = self.research.search(user_query)
                    agent_outputs["research"] = research_out
                    self.memory.store_knowledge(topic=user_query, content="; ".join([r["content"] for r in research_out["results"]]) or "no results", source="research_sim", agent="ResearchAgent", confidence=research_out.get("confidence",0.6))
                    self.memory.store_agent_state(task_id, "ResearchAgent", f"returned {len(research_out['results'])} results")
                    self.log_trace({"task": task_id, "agent": "ResearchAgent", "payload": research_out})
                # Analysis phase
                if plan["analysis"]:
                    # feed research results if available else ask research agent for similar topics
                    feed = research_out["results"] if research_out else []
                    analysis_out = None
                    # simple routing based on hints in query
                    if "optimizer" in user_query.lower() or "optim" in user_query.lower():
                        analysis_out = self.analysis.compare_optimizers(feed)
                    elif "transformer" in user_query.lower():
                        analysis_out = self.analysis.analyze_transformer_efficiency(feed)
                    else:
                        # generic: ask analysis to compare what it received
                        analysis_out = self.analysis.compare_optimizers(feed if feed else [])
                    agent_outputs["analysis"] = analysis_out
                    self.memory.store_agent_state(task_id, "AnalysisAgent", f"produced analysis with confidence {analysis_out.get('confidence')}")
                    self.log_trace({"task": task_id, "agent": "AnalysisAgent", "payload": analysis_out})
                # Synthesis
                final_answer = self._synthesize(user_query, research_out if plan["research"] else None, agent_outputs.get("analysis"))
                confidence = round((research_out.get("confidence",0.5) if research_out else 0.5) * 0.6 + (agent_outputs.get("analysis",{}).get("confidence",0.5))*0.4, 2)
                # update memory with final synthesis
                self.memory.store_knowledge(topic=user_query, content=final_answer, source="synthesis", agent="Coordinator", confidence=confidence)
                self.memory.store_agent_state(task_id, "Coordinator", f"synthesized final answer with confidence {confidence}")
            # log conversation assistant
            self.memory.log_conversation("assistant", final_answer)
            self.log_trace({"task": task_id, "final_confidence": confidence})
            return {"task_id": task_id, "answer": final_answer, "agents": agent_outputs, "confidence": confidence, "trace": self.trace[-6:]}
        except Exception as e:
            tb = traceback.format_exc()
            self.log_trace({"task": task_id, "error": str(e), "traceback": tb})
            return {"task_id": task_id, "answer": "Sorry, I encountered an error while processing.", "error": str(e)}

    def _synthesize(self, query: str, research_out=None, analysis_out=None) -> str:
        pieces = []
        pieces.append(f"Question: {query}")
        if research_out:
            pieces.append("Research findings:")
            for r in research_out["results"]:
                pieces.append(f"- {r['title']}: {r['content']} (src: {r['source']}, conf: {r['confidence']})")
        if analysis_out:
            pieces.append("Analysis summary:")
            if "ranking" in analysis_out:
                pieces.append("Ranking (best to worst):")
                for title,score in analysis_out["ranking"]:
                    pieces.append(f"  * {title} (score {score})")
            if "impacts" in analysis_out:
                for imp in analysis_out["impacts"]:
                    pieces.append(f"  * {imp['title']}: {imp['reason']} (conf {imp['confidence']})")
        pieces.append("Short recommendation: Use this as a starting point; verify with primary literature.")
        return "\n".join(pieces)

    def _format_memory_response(self, mem):
        parts = []
        keyword_matches = mem.get("keyword_matches", [])
        vector_matches = mem.get("vector_matches", [])
        if not keyword_matches and not vector_matches:
            return "I couldn't find earlier notes about that topic."
        if keyword_matches:
            parts.append("Keyword matches from memory:")
            for r in keyword_matches:
                parts.append(f"- {r['topic']}: {r['content']} (source {r['source']})")
        if vector_matches:
            parts.append("Vector-similarity matches:")
            for r in vector_matches:
                parts.append(f"- score {r['score']}: {r['metadata'].get('topic')} (id {r['id']})")
        return "\n".join(parts)
