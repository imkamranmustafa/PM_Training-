# tests.py
"""
Runs the five sample scenarios and writes outputs to outputs/*.txt
"""
from coordinator import Coordinator
import os
os.makedirs("outputs", exist_ok=True)

scenarios = {
    "simple_query": "What are the main types of neural networks?",
    "complex_query": "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
    "memory_test": "What did we discuss about neural networks earlier?",
    "multi_step": "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
    "collaborative": "Compare SGD and Adam and recommend which is better for our use case."
}

def run_and_save(name, query, coord):
    res = coord.handle(query)
    path = os.path.join("outputs", f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("QUERY:\n")
        f.write(query + "\n\n")
        f.write("ANSWER:\n")
        f.write(res["answer"] + "\n\n")
        f.write("CONFIDENCE:\n")
        f.write(str(res.get("confidence")) + "\n\n")
        f.write("TRACE (last items):\n")
        for t in res.get("trace", []):
            f.write(str(t) + "\n")
    print(f"Wrote {path}")

if __name__ == "__main__":
    c = Coordinator()
    # Run first to populate memory for memory_test
    run_and_save("simple_query", scenarios["simple_query"], c)
    run_and_save("complex_query", scenarios["complex_query"], c)
    run_and_save("multi_step", scenarios["multi_step"], c)
    run_and_save("collaborative", scenarios["collaborative"], c)
    # memory test (queries memory)
    run_and_save("memory_test", scenarios["memory_test"], c)
    print("All tests completed. Check outputs/ directory.")
