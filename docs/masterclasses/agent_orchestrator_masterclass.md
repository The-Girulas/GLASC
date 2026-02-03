# Masterclass: Agent Orchestrator & LangGraph

## 1. The Intent: From Chain to Cycle

Classic pipelines ("RAG Chain") are linear: `Input -> Retrieval -> LLM -> Output`.
In Finance, this is insufficient. An investment decision requires **iterative reflection**:
1.  "I lack info on debt."
2.  "Search for covenants." -> *Tool*
3.  "Covenants found. Ah, it's risky. What about volatility?"
4.  "Calculate VaR." -> *Tool*
5.  "VaR OK. Let's go."

This is why we use **LangGraph**, which allows defining **Cyclical State Graphs**.

## 2. "Under the Hood": Strategist Architecture

The `AgentState` is the shared working memory.
Each node (`QuantAnalyst`, `CorporateSpy`) is a **Specialist** that enriches this state.

The `Strategist` node (our Local LLM Qwen 1.5B) acts as the Frontal Cortex. It does not do the calculations (that's JAX's job), it makes the decisions.
It receives the `risk_report` (structured) and the `brain_score` (numerical) and deduces a semantic strategy ("ATTACK").

## 3. Framework Focus: Local LLM Inference

We made the radical choice to integrate the LLM **into the heart of the runtime** (`glasc/core/llm_client.py`), and not via a distant external API.
Why?
*   **Confidentiality**: Takeover plans never leave the server.
*   **Latency**: No HTTP network calls.
*   **Control**: We use `transformers` to load the precise model (`Qwen2.5-1.5B-Instruct`), ensuring behavior is unchanged from one run to another (Reproducibility).

## 4. Pro-Tip: TypedDict State

Using a `TypedDict` (`AgentState`) rather than a catch-all dictionary is crucial.
It enforces rigor: If the `PredatorOracle` forgets to write the `brain_score`, typing will alert us (or at least the IDE).
In complex systems with 10+ agents, strict data schema is the only thing preventing chaos.
