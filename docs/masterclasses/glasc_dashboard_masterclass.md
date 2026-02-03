# Masterclass: The Dashboard (War Room)

## 1. The Intent: "Iron Man" Interface

Why a "Sci-Fi" interface for finance? It's not (just) for style.
In High-Frequency Trading or M&A, information density is critical.
- **Dark Theme**: Reduces eye strain.
- **Neon / Bright Colors**: Draw immediate attention to anomalies (Risk, Opportunity).
- **Glassmorphism**: Allows layering information without losing global context.

Our Streamlit dashboard is not a passive PDF report. It is a **Control Cockpit**.

## 2. "Under the Hood": Streamlit + CSS Hack

Streamlit is great for rapid Python prototyping, but its default look is... "Web 2.0".
To get the "Palantir" look, we inject raw CSS:

```python
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: rgba(28, 35, 49, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 200, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)
```
This transforms standard widgets into futuristic glass panels.

## 3. Framework Focus: Plotly & JAX Integration

Connecting JAX (Heavy computational backend) to Streamlit (Lightweight frontend) requires a trick.
- **Problem**: If JAX recalculates 100k paths at every click, the interface freezes.
- **Solution**: The `jax.jit` cache is persistent as long as the Python process is alive. Streamlit reruns the script at every interaction, but JAX keeps its compiled XLA kernels in GPU/RAM memory.
- **Plotly**: Used for its interactivity (zoom, hover) on charts, where Matplotlib would be static.

## 4. UX: Explainable AI

The "Strategist Thought Stream" panel is crucial.
It is not enough for the AI to say "ATTACK". The human trader must see **why**.
By displaying the internal LangGraph message flow, we open the "Black Box".
The human sees: "Ah, the AI saw high debt, checked covenants, and THAT triggered the attack."
This is the key to Man-Machine trust.
