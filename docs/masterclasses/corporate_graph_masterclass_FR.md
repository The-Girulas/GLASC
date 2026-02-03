# Masterclass : Corporate Graph & Behavioral Finance

## 1. L'Intention : Modéliser le "Terrain de Bataille"

Une OPA (Offre Publique d'Achat) n'est pas qu'une affaire de prix. C'est une guerre de mouvement sur un graphe social et financier.
Pour réussir un rachat hostile, il faut :
1.  **Identifier les Maillons Faibles** : Les actionnaires prêts à vendre (Mercenaires).
2.  **Contourner les Défenses** : Pillules empoisonnées (Poison Pills), Covenants de dette.
3.  **Convaincre** : Atteindre le seuil de 50% + 1 voix.

Nous modélisons ceci avec un graphe `NetworkX` où :
- **Noeuds** : Actionnaires, Banques, Board Members.
- **Arêtes** : Relations de propriété (OWNS), de dette (LENDS_TO) et d'influence (INFLUENCES).

## 2. "Under the Hood" : Behavioral Finance

Comment simuler la décision de vendre ?
Un modèle purement rationnel ("Prix Offert > Prix Marché => Vente") est faux.
Dans `will_tender()`, nous implémentons une fonction d'utilité comportementale :

$$ Score = \alpha \cdot \text{Premium} + \beta \cdot \text{GainHistorique} $$
$$ \text{Decision} = Score > \text{SeuilLoyauté} + \text{AversionPerte} $$

- **Premium** : L'incitatif immédiat (+20% cash demain).
- **Gain Historique** : L'effet de dotation (Endowment Effect). "J'ai acheté à 10, ça vaut 100, je suis riche". Mais nous avons plafonné cet effet : un actionnaire fidèle ne vend pas son "bébé" juste parce qu'il a fait du profit. Il faut un Premium.
- **Loyauté** : Un facteur multiplicatif sur le premium requis.
- **Aversion à la Perte** : Si `Offre < CostBasis`, le seuil de vente augmente drastiquement. Personne n'aime valider une perte.

## 3. Focus Framework : Pydantic & NetworkX

Pourquoi `Pydantic` ?
Dans un graphe complexe, les données non structurées sont une source de bugs infinis (ex: un covenant manquant).
`Pydantic` garantit le schéma de chaque noeud (`Shareholder`, `DebtTranche`).
Si on essaie d'ajouter une dette sans `max_leverage_ratio`, le code plante *avant* la simulation.

Pourquoi `NetworkX` ?
Pour calculer des propriétés émergentes :
- **Centralité** : Qui est l'actionnaire pivot ?
- **Connected Components** : Y a-t-il un bloc d'actionnaires concertistes (Pacte d'actionnaires) ?

## 4. Pro-Tip : Détection de Covenants en Temps Réel

L'un des plus grands risques d'un LBO (Leveraged Buy-Out) est de déclencher le remboursement immédiat de la dette existante.
Notre méthode `check_poison_pills` surveille le ratio `NetDebt / EBITDA` en temps réel pendant la simulation. C'est un "Système d'Alerte Précoce" pour l'attaquant.
