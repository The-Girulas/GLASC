"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Dashboard (UI) - Style 'Palantir'

Interface de commande type "Sci-Fi War Room".
"""

import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import random
from glasc.core.agent_orchestrator import create_orchestrator
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIGURATION & CSS ---
st.set_page_config(page_title="GLASC STRATEGIC CONTROLLER", layout="wide", page_icon="🌐")

# Palantir / Sci-Fi Theme CSS
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 50% 50%, #1c2331 0%, #0e1117 100%);
        color: #e0e0e0;
    }
    
    /* Glassmorphism Cards */
    div[data-testid="stMetric"], div.stDataFrame, div.stPlotlyChart {
        background: rgba(28, 35, 49, 0.4);
        border: 1px solid rgba(0, 200, 255, 0.2);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Neon Text */
    h1, h2, h3 {
        color: #00eeff !important;
        text-shadow: 0 0 10px rgba(0, 238, 255, 0.5);
        font-family: 'Segoe UI', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #0055ff, #00aaff);
        color: white;
        border: none;
        border-radius: 4px;
        box-shadow: 0 0 15px rgba(0, 85, 255, 0.6);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 85, 255, 0.9);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0c10;
        border-right: 1px solid rgba(0, 200, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌐 GLASC // STRATEGIC CONTROLLER")
    st.markdown("*Global Leverage & Asset Strategy Controller // v1.0*")
with col2:
    st.metric(label="SYSTEM STATUS", value="ONLINE", delta="SECURE")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("MISSION PARAMETERS")
    ticker = st.text_input("TARGET TICKER", value="EVIL_CORP", help="Target company symbol")
    
    st.subheader("OVERRIDE METRICS")
    volatility = st.slider("Implied Volatility (IV)", 0.1, 1.0, 0.3, 0.05)
    premium = st.slider("Takeover Premium", 0.0, 1.0, 0.2, 0.05)
    
    launch_btn = st.button("LAUNCH ATTACK SIMULATION", use_container_width=True)
    
    st.divider()
    st.info("System connected to Local LLM (Qwen2.5-1.5B). JAX Engine Ready.")

# --- MAIN DASHBOARD LAYOUT ---
# 2 Rows: Top (Viz), Bottom (Logs)

# Placeholder variables
if "history" not in st.session_state:
    st.session_state.history = []

if launch_btn:
    st.session_state.history = [] # Reset
    
    # 1. VISUALIZATION ROW
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📡 MARKET DYNAMICS (JAX)")
        # Simuler ou récupérer les paths JAX
        # Pour l'UI, générons un plot Plotly rapide qui ressemble à JAX output
        # (L'orchestrateur le fait en background, mais on veut un beau visuel ici)
        x = np.linspace(0, 1, 100)
        # Generate 50 paths
        paths = np.cumsum(np.random.randn(50, 100), axis=1) + 100
        
        fig_market = go.Figure()
        for i in range(50):
            fig_market.add_trace(go.Scatter(x=x, y=paths[i], mode='lines', 
                                            line=dict(color='#00eeff', width=1, dash='solid'),
                                            opacity=0.3, showlegend=False))
            
        fig_market.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Time (Years)",
            yaxis_title="Projected Asset Price",
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        st.plotly_chart(fig_market, use_container_width=True)

    with row1_col2:
        st.subheader("🕸️ CORPORATE NEXUS")
        # Fake Graph viz for demo
        G = nx.random_geometric_graph(20, 0.3)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Influence',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'Entity #{node}<br># Connections: {len(adjacencies[1])}')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig_graph = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=300
                     ))
        st.plotly_chart(fig_graph, use_container_width=True)
        
    # 2. ORCHESTRATOR EXECUTION
    st.divider()
    st.subheader("🧠 STRATEGIST THOUGHT STREAM")
    
    log_container = st.container(height=300, border=True)
    
    # Run the real Agent
    with st.spinner("INITIALIZING NEURAL LINK..."):
        app = create_orchestrator()
        initial_state = {
            "ticker": ticker,
            "messages": [],
            "market_data": {"price": 100, "volatility": volatility},
            "risk_report": {},
            "corp_structure": {},
            "brain_score": 0.0,
            "decision": "PENDING"
        }
        
        # Stream the execution
        # Note: LangGraph doesn't stream steps easily in V1 unless using .stream()
        # For V1 demo, we invoke and then iterate messages, 
        # but to make it look cool in UI we simulate streaming delay.
        
        final_state = app.invoke(initial_state)
        
        messages = final_state['messages']
        
        for msg in messages:
            with log_container:
                if isinstance(msg, SystemMessage):
                    st.markdown(f"**🛠️ SYSTEM:** `{msg.content}`")
                elif isinstance(msg, HumanMessage): # Strategist (LLM) often mapped to Human or AI depending on impl
                     st.markdown(f"**🤖 STRATEGIST:** {msg.content}")
                
            time.sleep(0.8) # Cinematic delay

    # 3. FINAL DECISION
    decision = final_state.get("decision", "WAIT")
    
    st.divider()
    col_fin1, col_fin2, col_fin3 = st.columns([1, 2, 1])
    with col_fin2:
        if decision == "ATTACK":
            st.error(f"### 🚨 FINAL DECISION: {decision} 🚨")
            st.markdown("Initiating hostile takeover protocols...")
        elif decision == "WAIT":
            st.warning(f"### ⏸️ FINAL DECISION: {decision}")
            st.markdown("Market conditions suboptimal. Monitoring...")
        else:
            st.info(f"### 🛡️ FINAL DECISION: {decision}")

else:
    st.info("Awaiting Mission Start Command...")
