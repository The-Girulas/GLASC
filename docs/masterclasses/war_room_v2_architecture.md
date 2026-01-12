# Masterclass : War Room V2 (React Architecture)

## 1. L'Intention : "Hollywood-Grade" Interface

L'interface Streamlit était fonctionnelle mais statique.
Pour convaincre un utilisateur High-Level (Hedge Fund Manager), l'interface doit être "vivante".
- **React Force Graph 3D** : Le réseau d'entreprise n'est pas un dessin figé, c'est un univers navigable.
- **WebSocket Streaming** : L'IA ne "répond" pas après 30s. Elle *pense* devant vous, ligne par ligne. C'est psychologiquement crucial pour l'attente.

## 2. Fullstack Architecture

### Backend (The Brain) - `api/main.py`
Nous avons wrappé le GLASC Core dans **FastAPI**.
Pourquoi ?
- **Asynchrone** : FastAPI gère des milliers de connexions WebSocket sans bloquer le moteur JAX.
- **Microservice Ready** : Cette API peut être déployée indépendamment du frontend (ex: sur un cluster GPU).

### Frontend (The Face) - `frontend/`
Construit avec **Vite + React**.
- **TailwindCSS** : Pour le "Rapid Styling". Nous avons créé une config custom (`tailwind.config.js`) avec nos couleurs Néon.
- **Recharts** & **ForceGraph** : Bibliothèques de visualisation D3.js encapsulées pour React.

## 3. Le "Thought Stream Protocol"
Le défi majeur est de montrer le raisonnement de l'agent.
Nous n'utilisons pas HTTP Request/Response classique.
Nous ouvrons un **WebSocket (`ws://...`)**.
1. Le Frontend s'abonne.
2. L'Agent LangGraph émet des événements (`STEP`, `THOUGHT`, `DECISION`).
3. Le Frontend les affiche comme un terminal de hacker ("Typewriter effect").

Cela donne l'illusion de "lire dans les pensées" de la machine.
