# Masterclass : Predator Brain & Synthetic Reality

## 1. L'Intention : Un Cerveau pour la Finance

Notre objectif est de créer un agent capable de détecter des opportunités d'OPA hostile que l'humain ne voit pas. Pour cela, nous utilisons un **Réseau de Neurones Profond (Deep Learning)**.

Contrairement à un algorithme classique ("Si Dette > 5, Alors Rejet"), le réseau de neurones apprend des interactions non-linéaires subtiles :
> *"Une dette élevée est mauvaise, SAUF si le Cash Flow est énorme ET que les taux sont bas."*

## 2. "Under the Hood" : Training on Synthetic Reality

Le plus grand défi de l'IA financière est le manque de données labellisées (Historical M&A Data is scarce).
Nous avons résolu ce problème par la **Génération de Données Synthétiques Réalistes**.

Suite à une recherche approfondie sur les marchés US/Europe, nous avons calibré nos générateurs (`RealisticDataGenerator`) :

### A. La Dette (Le Danger)
La distribution `NetDebt/EBITDA` n'est pas normale. Elle est **Log-Normale** avec une "Fat Tail".
- La plupart des boites saines sont autour de **2.5x**.
- Mais il existe des "zombies" à **6x - 8x** (Cibles potentielles de restructuration ou pièges mortels).
Notre générateur reproduit cette asymétrie.

### B. L'Actionnariat (Le Verrou)
Nous utilisons une **Distribution Beta** pour modéliser la part des Institutionnels ("Smart Money").
- Les Large Caps sont détenues à **~70-80%** par des fonds (BlackRock, Vanguard).
- Les Small Caps sont souvent détenues par des Insiders (Familles).
Le Brain apprend ainsi que s'attaquer à une Small Cap familiale (Insider > 40%) est futile, même si la dette est faible.

## 3. Focus Framework : Flax & Optax

Pourquoi **Flax** ?
Flax (par Google Research) est construit sur l'idée de **"Functional Programming"**.
Au lieu d'avoir des objets avec un état caché (`self.weights` en PyTorch), Flax sépare totalement :
1.  **Architecture** (Le code, immuable).
2.  **Paramètres** (Les poids, un simple dictionnaire JAX).

Cela rend le code :
- **Purement fonctionnel** : Pas d'effets de bord.
- **JIT-Compliant** : On peut compiler toute la boucle d'entraînement (`train_step`) avec XLA pour une vitesse extrême.

## 4. Pro-Tip : L'Oracle "Vérité Terrain"

Pour entraîner le cerveau, nous avons codé un "Oracle" (une formule complexe cachée) qui détermine si une OPA *devrait* réussir.
Le réseau de neurones n'a jamais accès à cette formule. Il voit seulement les inputs (Bilans) et le résultat (Succès/Échec).
Il doit **rétro-ingénierier** l'intuition du marché par lui-même.
C'est le principe de la "Distillation de Connaissance".
