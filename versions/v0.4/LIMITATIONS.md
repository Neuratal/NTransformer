# LIMITATIONS — v0.4

Inherits v0.3. Additionally:
- experimental_cuda_graphs is opt-in and CUDA/B=1-only; eager remains default.
- Suboptimal at long prompts (window padding; 0.94x measured at 512 tokens).
- Captures once per generation call; short generations amortize poorly.
- Session caches are private to the call; no cross-call graph reuse yet.
- Graph body is a tested duplication of the eager sublayer flow — drift would
  be caught by tests/test_graph_decode.py, not by architecture.
