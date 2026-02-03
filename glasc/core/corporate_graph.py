"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Corporate Graph (Battlefield Mapping)

This module models the target company structure as a graph (NetworkX).
It integrates Pydantic-validated data to simulate Tender Offer scenarios.

Entities:
- Shareholder: Holder of votes.
- DebtTranche: Creditor with covenants.
- BoardMember: Influencer.
"""

import networkx as nx
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import math

# --- Pydantic Data Structures ---

class Shareholder(BaseModel):
    """Shareholder holding capital and voting rights."""
    name: str
    share_count: int = Field(ge=0, description="Number of shares held")
    category: Literal["INSTITUTIONAL", "RETAIL", "INSIDER", "ACTIVIST", "PASSIVE"]
    cost_basis: float = Field(..., description="Estimated average purchase price")
    loyalty_score: float = Field(ge=0.0, le=1.0, description="Probability of NOT selling (0=Mercenary, 1=Loyal)")
    
    @property
    def value(self, current_price: float) -> float:
        return self.share_count * current_price

    def will_tender(self, offer_price: float, current_price: float) -> bool:
        """
        Simple deterministic probabilistic model for the simulation.
        Decides if the shareholder tenders their shares to the offer.
        """
        premium = (offer_price - current_price) / current_price
        gain_vs_cost = (offer_price - self.cost_basis) / self.cost_basis
        
        # Attractiveness Score of the Offer
        # Premium is king. Historical gain is a capped bonus.
        # You don't sell your company just because you made 10x in 20 years, you sell for the current premium.
        capped_gain = min(gain_vs_cost, 0.5) # Capped at +50% impact
        attractiveness = premium + 0.2 * capped_gain
        
        # Selling threshold adjusted by loyalty
        # Loyalty 1.0 (Logan) -> Threshold 0.3 (Needs 30% combined attractiveness)
        # Loyalty 0.0 (Stewy) -> Threshold 0.0 (Sells as soon as it's positive)
        tender_threshold = self.loyalty_score * 0.3
        
        # To avoid irrational selling at a loss if premium > 0 but price < cost
        if offer_price < self.cost_basis and self.category != "ACTIVIST":
             # Retail/Insiders hate selling at a loss (Loss Aversion)
             tender_threshold += 0.2
        
        return attractiveness > tender_threshold


class DebtTranche(BaseModel):
    """Company debt with its conditions (Covenants)."""
    lender_name: str
    amount: float
    interest_rate: float
    maturity_years: float
    is_convertible: bool = False
    # Covenants: Max Leverage Limit (NetDebt/EBITDA)
    max_leverage_ratio: Optional[float] = None
    
    def check_covenant_breach(self, ebitda: float, total_net_debt: float) -> bool:
        if self.max_leverage_ratio is None:
            return False
        if ebitda <= 0: return True # Breach if EBITDA negative
        return (total_net_debt / ebitda) > self.max_leverage_ratio


class BoardMember(BaseModel):
    """Member of the Board of Directors."""
    name: str
    is_independent: bool
    influence_score: float = Field(ge=0.0, le=10.0)
    

# --- Network Logic ---

class CorporateNetwork:
    """Wrapper around a NetworkX DiGraph representing the ecosystem."""
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.graph = nx.DiGraph()
        # The central node
        self.graph.add_node("COMPANY", type="TARGET", name=company_name)
        
    def add_shareholder(self, sh: Shareholder):
        """Adds a shareholder to the graph with an 'OWNS' link."""
        self.graph.add_node(sh.name, type="SHAREHOLDER", data=sh)
        self.graph.add_edge(sh.name, "COMPANY", relation="OWNS", share_count=sh.share_count)
        
    def add_debt(self, dt: DebtTranche):
        """Adds debt to the graph with a 'LENDS_TO' link."""
        self.graph.add_node(dt.lender_name, type="LENDER", data=dt)
        self.graph.add_edge(dt.lender_name, "COMPANY", relation="LENDS_TO", amount=dt.amount)

    def add_board_member(self, bm: BoardMember):
        """Adds a board member."""
        self.graph.add_node(bm.name, type="BOARD_MEMBER", data=bm)
        self.graph.add_edge(bm.name, "COMPANY", relation="SITS_ON_BOARD", influence=bm.influence_score)

    def get_total_shares(self) -> int:
        return sum(d["share_count"] for u, v, d in self.graph.edges(data=True) if d.get("relation") == "OWNS")

    def get_shareholder_distribution(self) -> Dict[str, float]:
        """Returns the capital distribution by category."""
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
        Simulates the success percentage of a Tender Offer at a given price.
        Returns the % of capital tendered.
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
        """Verifies if the buyout (using debt) triggers covenants."""
        triggers = []
        
        # Calculate total existing debt
        current_debt = sum(d["amount"] for u, v, d in self.graph.edges(data=True) if d.get("relation") == "LENDS_TO")
        total_debt = current_debt + additional_debt
        
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "LENDER":
                dt: DebtTranche = d["data"]
                if dt.check_covenant_breach(simulated_ebitda, total_debt):
                    triggers.append(f"COVENANT BREACH: {dt.lender_name} (Max Leverage {dt.max_leverage_ratio})")
                    
        return triggers
