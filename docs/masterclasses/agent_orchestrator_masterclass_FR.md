# Masterclass : Agent Orchestrator & LangGraph

## 1. L'Intention : De la Chaîne au Cycle

Les pipelines classiques ("RAG Chain") sont linéaires : `Input -> Retrieval -> LLM -> Output`.
En Finance, c'est insuffisant. Une décision d'investissement nécessite une **réflexion itérative** :
1.  "Je manque d'info sur la dette."
2.  "Cherche les covenants." -> *Outil*
3.  "Covenants trouvés. Ah, c'est risqué. Et la volatilité ?"
4.  "Calcule la VaR." -> *Outil*
5.  "VaR OK. On y va."

C'est pourquoi nous utilisons **LangGraph**, qui permet de définir des **Graphes d'États Cycliques**.

## 2. "Under the Hood" : Architecture du Strategist

Le `AgentState` est la mémoire de travail partagée.
Chaque noeud (`QuantAnalyst`, `CorporateSpy`) est un **Spécialiste** qui enrichit cet état.

Le noeud `Strategist` (notre LLM Local Qwen 1.5B) agit comme le Cortex Frontal. Il ne fait pas les calculs (c'est le job de JAX), il prend les décisions.
Il reçoit le `risk_report` (structuré) et le `brain_score` (numérique) et en déduit une stratégie sémantique ("ATTACK").

## 3. Focus Framework : Local LLM Inference

Nous avons fait le choix radical d'intégrer le LLM **au coeur du runtime** (`glasc/core/llm_client.py`), et non via une API externe distante.
Pourquoi ?
*   **Confidentialité** : Les plans d'OPA ne sortent jamais du serveur.
*   **Latence** : Pas d'appel réseau HTTP.
*   **Contrôle** : Nous utilisons `transformers` pour charger le modèle précis (`Qwen2.5-1.5B-Instruct`), garantissant que le comportement est inchangé d'une exécution à l'autre (Reproductibilité).

## 4. Pro-Tip : TypedDict State

Utiliser un `TypedDict` (`AgentState`) plutôt qu'un dictionnaire fourre-tout est crucial.
Cela force une rigueur : Si le `PredatorOracle` oublie d'écrire le `brain_score`, le typage nous alertera (ou au moins l'IDE).
Dans des systèmes complexes à 10+ agents, le schéma de données strict est la seule chose qui empêche le chaos.
