"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Corporate Graph (Battlefield Mapping)

Ce module modélise la structure de l'entreprise cible sous forme de graphe (NetworkX).
Il intègre les données validées par Pydantic pour simuler des scénarios d'OPA (Tender Offer).

Entities:
- Shareholder: Détenteur de votes.
- DebtTranche: Créancier avec covenants.
- BoardMember: Influenceur.
"""

import networkx as nx
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import math

# --- Pydantic Data Structures ---

class Shareholder(BaseModel):
    """Actionnaire détenant une part du capital et des droits de vote."""
    name: str
    share_count: int = Field(ge=0, description="Nombre d'actions détenues")
    category: Literal["INSTITUTIONAL", "RETAIL", "INSIDER", "ACTIVIST", "PASSIVE"]
    cost_basis: float = Field(..., description="Prix moyen d'achat estimé")
    loyalty_score: float = Field(ge=0.0, le=1.0, description="Probabilité de ne PAS vendre (0=Mercenaire, 1=Fidèle)")
    
    @property
    def value(self, current_price: float) -> float:
        return self.share_count * current_price

    def will_tender(self, offer_price: float, current_price: float) -> bool:
        """
        Modèle probabiliste déterministe simple pour la simulation.
        Décide si l'actionnaire apporte ses titres à l'offre.
        """
        premium = (offer_price - current_price) / current_price
        gain_vs_cost = (offer_price - self.cost_basis) / self.cost_basis
        
        # Score d'attractivité de l'offre
        # Premium est roi. Le gain historique est un bonus plafonné.
        # On ne vend pas sa boite juste parce qu'on a fait x10 depuis 20 ans, on vend pour le premium actuel.
        capped_gain = min(gain_vs_cost, 0.5) # Plafonné à +50% d'impact
        attractiveness = premium + 0.2 * capped_gain
        
        # Seuil de vente ajusté par la loyauté
        # Loyalty 1.0 (Logan) -> Threshold 0.3 (Il faut 30% d'attractivité combinée)
        # Loyalty 0.0 (Stewy) -> Threshold 0.0 (Vend dès que c'est positif)
        tender_threshold = self.loyalty_score * 0.3
        
        # Pour éviter les ventes à perte irrationnelles si premium > 0 mais prix < cost
        if offer_price < self.cost_basis and self.category != "ACTIVIST":
             # Les retail/insiders détestent vendre à perte (Loss Aversion)
             tender_threshold += 0.2
        
        return attractiveness > tender_threshold


class DebtTranche(BaseModel):
    """Dette de l'entreprise avec ses conditions (Covenants)."""
    lender_name: str
    amount: float
    interest_rate: float
    maturity_years: float
    is_convertible: bool = False
    # Covenants: Limite max de Leverage (NetDebt/EBITDA)
    max_leverage_ratio: Optional[float] = None
    
    def check_covenant_breach(self, ebitda: float, total_net_debt: float) -> bool:
        if self.max_leverage_ratio is None:
            return False
        if ebitda <= 0: return True # Breach if EBITDA negative
        return (total_net_debt / ebitda) > self.max_leverage_ratio


class BoardMember(BaseModel):
    """Membre du Conseil d'Administration."""
    name: str
    is_independent: bool
    influence_score: float = Field(ge=0.0, le=10.0)
    

# --- Network Logic ---

class CorporateNetwork:
    """Wrapper autour d'un DiGraph NetworkX représentant l'écosystème."""
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.graph = nx.DiGraph()
        # Le noeud central
        self.graph.add_node("COMPANY", type="TARGET", name=company_name)
        
    def add_shareholder(self, sh: Shareholder):
        """Ajoute un actionnaire au graphe avec un lien 'OWNS'."""
        self.graph.add_node(sh.name, type="SHAREHOLDER", data=sh)
        self.graph.add_edge(sh.name, "COMPANY", relation="OWNS", share_count=sh.share_count)
        
    def add_debt(self, dt: DebtTranche):
        """Ajoute une dette au graphe avec un lien 'LENDS_TO'."""
        self.graph.add_node(dt.lender_name, type="LENDER", data=dt)
        self.graph.add_edge(dt.lender_name, "COMPANY", relation="LENDS_TO", amount=dt.amount)

    def add_board_member(self, bm: BoardMember):
        """Ajoute un membre du board."""
        self.graph.add_node(bm.name, type="BOARD_MEMBER", data=bm)
        self.graph.add_edge(bm.name, "COMPANY", relation="SITS_ON_BOARD", influence=bm.influence_score)

    def get_total_shares(self) -> int:
        return sum(d["share_count"] for u, v, d in self.graph.edges(data=True) if d.get("relation") == "OWNS")

    def get_shareholder_distribution(self) -> Dict[str, float]:
        """Retourne la répartition du capital par catégorie."""
        total = self.get_total_shares()
        if total == 0: return {}
        
        dist = {}
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "SHAREHOLDER":
                sh: Shareholder = d["data"]
                pct = sh.share_count / total
                dist[sh.category] = dist.get(sh.category, 0.0) + pct
        return dist

    def simulate_tender_offer(self, offer_price: float, current_price: float) -> float:
        """
        Simule le pourcentage de réussite d'une OPA à un prix donné.
        Retourne le % du capital apporté.
        """
        total_shares = self.get_total_shares()
        tendered_shares = 0
        
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "SHAREHOLDER":
                sh: Shareholder = d["data"]
                if sh.will_tender(offer_price, current_price):
                    tendered_shares += sh.share_count
                    
        return tendered_shares / total_shares if total_shares > 0 else 0.0

    def check_poison_pills(self, simulated_ebitda: float, additional_debt: float = 0.0) -> List[str]:
        """Vérifie si le rachat (utilisant de la dette) déclenche des covenants."""
        triggers = []
        
        # Calcul dette totale existante
        current_debt = sum(d["amount"] for u, v, d in self.graph.edges(data=True) if d.get("relation") == "LENDS_TO")
        total_debt = current_debt + additional_debt
        
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "LENDER":
                dt: DebtTranche = d["data"]
                if dt.check_covenant_breach(simulated_ebitda, total_debt):
                    triggers.append(f"COVENANT BREACH: {dt.lender_name} (Max Leverage {dt.max_leverage_ratio})")
                    
        return triggers
