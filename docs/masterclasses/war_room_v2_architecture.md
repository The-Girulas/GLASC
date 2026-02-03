# Masterclass: War Room V2 (React Architecture)

## 1. The Intent: "Hollywood-Grade" Interface

The Streamlit interface was functional but static.
To convince a High-Level user (Hedge Fund Manager), the interface must be "alive".
- **React Force Graph 3D**: The corporate network is not a frozen drawing, it is a navigable universe.
- **WebSocket Streaming**: The AI doesn't "respond" after 30s. It *thinks* in front of you, line by line. This is psychologically critical for waiting times.

## 2. Fullstack Architecture

### Backend (The Brain) - `api/main.py`
We wrapped the GLASC Core in **FastAPI**.
Why?
- **Async**: FastAPI handles thousands of WebSocket connections without blocking the JAX engine.
- **Microservice Ready**: This API can be deployed independently of the frontend (e.g., on a GPU cluster).

### Frontend (The Face) - `frontend/`
Built with **Vite + React**.
- **TailwindCSS**: For "Rapid Styling". We created a custom config (`tailwind.config.js`) with our Neon colors.
- **Recharts** & **ForceGraph**: D3.js visualization libraries encapsulated for React.

## 3. The "Thought Stream Protocol"
The major challenge is showing the agent's reasoning.
We don't use classic HTTP Request/Response.
We open a **WebSocket (`ws://...`)**.
1. The Frontend subscribes.
2. The LangGraph Agent emits events (`STEP`, `THOUGHT`, `DECISION`).
3. The Frontend displays them like a hacker terminal ("Typewriter effect").

This gives the illusion of "reading the machine's thoughts".
