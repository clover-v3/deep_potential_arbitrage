from abc import ABC, abstractmethod
import numpy as np

class TransactionCostModel(ABC):
    @abstractmethod
    def compute_cost(self, trade_value: float, trade_size: float = None) -> float:
        """
        Returns absolute cost in currency.
        """
        pass

class LinearCostModel(TransactionCostModel):
    """
    Fixed basis points per turnover.
    """
    def __init__(self, bps: float = 10.0):
        self.rate = bps / 10000.0

    def compute_cost(self, trade_value: float, trade_size: float = None) -> float:
        return abs(trade_value) * self.rate

class ImpactCostModel(TransactionCostModel):
    """
    Square-root Impact Model.
    Cost = Linear + const * sigma * sqrt(Size / Volume)
    """
    def __init__(self, bps: float = 5.0, impact_coeff: float = 0.1):
        self.linear_rate = bps / 10000.0
        self.impact_coeff = impact_coeff

    def compute_cost(self, trade_value: float, trade_size: float = None) -> float:
        # Require trade_size (shares) and expected volume?
        # Simplified: Cost proportional to sqrt(Value)
        # This is a placeholder for advanced logic.
        linear = abs(trade_value) * self.linear_rate
        return linear
