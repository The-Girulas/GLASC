# 🚀 GLASC War Room - Launch Guide

Voici les commandes pour lancer le système manuellement si nécessaire.

## Pré-requis
Utiliser l'environnement virtuel `kaggle12` qui contient toutes les dépendances (JAX, FastAPI, React/Node).

## 1. Démarrer le Cerveau (Backend JAX/FastAPI)
Ouvrez un terminal :
```bash
cd /home/ubuntu/Projet/Recherche/GLASC
/home/ubuntu/anaconda3/envs/kaggle12/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
✅ **Vérification** : Ouvrez `http://localhost:8000/docs`. Vous devez voir le Swagger API.

## 2. Démarrer la War Room (Frontend React)
Ouvrez un **deuxième** terminal :
```bash
cd /home/ubuntu/Projet/Recherche/GLASC/frontend
npm run dev -- --host
```
✅ **Vérification** : Ouvrez `http://localhost:5173`. L'interface sombre doit apparaître.

## 3. Utilisation
1.  Cliquez sur **INITIATE ATTACK** (haut à droite).
2.  Observez les données de marché en temps réel (JAX Engine).
3.  Utilisez le panneau **CHAOS CONTROL** (en bas à droite) pour injecter des scénarios (ex: "SCANDAL LEAK").
