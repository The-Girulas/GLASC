# Masterclass: Corporate Graph & Behavioral Finance

## 1. The Intent: Modeling the "Battlefield"

A Takeover Bid (TOB) is not just about price. It is a war of movement on a social and financial graph.
To succeed in a hostile takeover, you must:
1.  **Identify Weak Links**: Shareholders ready to sell (Mercenaries).
2.  **Bypass Defenses**: Poison Pills, Debt Covenants.
3.  **Convince**: Reach the threshold of 50% + 1 vote.

We model this with a `NetworkX` graph where:
- **Nodes**: Shareholders, Banks, Board Members.
- **Edges**: Ownership relations (OWNS), debt (LENDS_TO), and influence (INFLUENCES).

## 2. "Under the Hood": Behavioral Finance

How do we simulate the decision to sell?
A purely rational model ("Offer Price > Market Price => Sell") is false.
In `will_tender()`, we implement a behavioral utility function:

$$ Score = \alpha \cdot \text{Premium} + \beta \cdot \text{HistoricalGain} $$
$$ \text{Decision} = Score > \text{LoyaltyThreshold} + \text{LossAversion} $$

- **Premium**: The immediate incentive (+20% cash tomorrow).
- **Historical Gain**: The Endowment Effect. "I bought at 10, it's worth 100, I'm rich." But we capped this effect: a loyal shareholder doesn't sell their "baby" just because they made a profit. A Premium is required.
- **Loyalty**: A multiplicative factor on the required premium.
- **Loss Aversion**: If `Offer < CostBasis`, the selling threshold increases drastically. No one likes to validate a loss.

## 3. Framework Focus: Pydantic & NetworkX

Why `Pydantic`?
In a complex graph, unstructured data is a source of infinite bugs (e.g., a missing covenant).
`Pydantic` guarantees the schema of each node (`Shareholder`, `DebtTranche`).
If we try to add debt without a `max_leverage_ratio`, the code crashes *before* the simulation.

Why `NetworkX`?
To calculate emergent properties:
- **Centrality**: Who is the pivotal shareholder?
- **Connected Components**: Is there a block of concerted shareholders (Shareholder Pact)?

## 4. Pro-Tip: Real-Time Covenant Detection

One of the greatest risks of an LBO (Leveraged Buy-Out) is triggering the immediate repayment of existing debt.
Our `check_poison_pills` method monitors the `NetDebt / EBITDA` ratio in real-time during the simulation. It is an "Early Warning System" for the attacker.
