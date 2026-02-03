# Masterclass : The Dashboard (War Room)

## 1. L'Intention : "Iron Man" Interface

Pourquoi une interface "Sci-Fi" pour de la finance ? Ce n'est pas (que) pour le style.
Dans le trading haute fréquence ou les M&A, la densité d'information est critique.
- **Thème Sombre** : Réduit la fatigue oculaire.
- **Néons / Couleurs Vives** : Attirent l'attention immédiate sur les anomalies (Risque, Opportunité).
- **Glassmorphism** : Permet de superposer des couches d'information sans perdre le contexte global.

Notre dashboard Streamlit n'est pas un rapport passif PDF. C'est un **Cockpit de Contrôle**.

## 2. "Under the Hood" : Streamlit + CSS Hack

Streamlit est génial pour le prototypage rapide Python, mais son look par défaut est... "Web 2.0".
Pour obtenir le look "Palantir", nous injectons du CSS brut :

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
Cela transforme les widgets standards en panneaux de verre futuristes.

## 3. Focus Framework : Plotly & JAX Integration

Connecter JAX (Backend de calcul lourd) à Streamlit (Frontend léger) demande de l'astuce.
- **Problème** : Si JAX recalcule 100k paths à chaque clic, l'interface fige.
- **Solution** : Le cache `jax.jit` est persistant tant que le processus Python vit. Streamlit relance le script à chaque interaction, mais JAX garde ses noyaux XLA compilés en mémoire GPU/RAM.
- **Plotly** : Utilisé pour son interactivité (zoom, hover) sur les graphes, là où Matplotlib serait statique.

## 4. UX : Explainable AI

Le panneau "Strategist Thought Stream" est crucial.
Il ne suffit pas que l'IA dise "ATTACK". Le trader humain doit voir **pourquoi**.
En affichant le flux de messages interne de LangGraph, nous ouvrons la "Boîte Noire".
L'humain voit : "Ah, l'IA a vu que la dette était haute, a vérifié les covenants, et c'est CA qui a déclenché l'attaque."
C'est la clé de la confiance Homme-Machine.
