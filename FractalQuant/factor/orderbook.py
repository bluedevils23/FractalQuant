"""
订单簿和市场微观结构因子
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base import OrderBookFactor

class OrderBookImbalanceFactor(OrderBookFactor):
    """订单簿失衡因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_imbalance', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿失衡"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        bid_volume = sum([v for p, v in bids])
        ask_volume = sum([v for p, v in asks])
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
        return imbalance

class OrderBookPressureFactor(OrderBookFactor):
    """订单簿压力因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_pressure', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿压力"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        bid_weighted = sum([p * v for p, v in bids])
        ask_weighted = sum([p * v for p, v in asks])
        
        pressure = bid_weighted / (ask_weighted + 1e-8)
        return pressure

class OrderBookSlopeFactor(OrderBookFactor):
    """订单簿斜率因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_slope', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿斜率"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        if len(bids) < 2 or len(asks) < 2:
            return 0
        
        bid_prices = [p for p, v in bids]
        bid_volumes = [v for p, v in bids]
        
        ask_prices = [p for p, v in asks]
        ask_volumes = [v for p, v in asks]
        
        bid_slope = (bid_prices[0] - bid_prices[-1]) / (bid_volumes[0] - bid_volumes[-1] + 1e-8)
        ask_slope = (ask_prices[-1] - ask_prices[0]) / (ask_volumes[-1] - ask_volumes[0] + 1e-8)
        
        slope = bid_slope + ask_slope
        return slope

class OrderBookDecayFactor(OrderBookFactor):
    """订单簿衰减因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_decay', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿衰减"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        if len(bids) < 2 or len(asks) < 2:
            return 0
        
        bid_volumes = [v for p, v in bids]
        ask_volumes = [v for p, v in asks]
        
        bid_decay = bid_volumes[0] / (bid_volumes[-1] + 1e-8)
        ask_decay = ask_volumes[0] / (ask_volumes[-1] + 1e-8)
        
        decay = (bid_decay + ask_decay) / 2
        return decay

class SpreadFactor(OrderBookFactor):
    """买卖价差因子"""
    
    def __init__(self):
        super().__init__('spread', 1)
        
    def calculate(self, orderbook: Dict) -> float:
        """计算买卖价差"""
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return 0
        
        bid_price = orderbook['bids'][0][0]
        ask_price = orderbook['asks'][0][0]
        
        spread = (ask_price - bid_price) / (bid_price + ask_price) * 2
        return spread

class DepthFactor(OrderBookFactor):
    """订单簿深度因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('depth', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿深度"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        bid_depth = sum([v for p, v in bids])
        ask_depth = sum([v for p, v in asks])
        
        total_depth = bid_depth + ask_depth
        return total_depth

class OrderBookVelocityFactor(OrderBookFactor):
    """订单簿变化速度因子"""
    
    def __init__(self, window: int = 5):
        super().__init__('orderbook_velocity', window)
        self.window = window
        self.history = []
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿变化速度"""
        current_imbalance = self.calculate_imbalance(orderbook)
        
        if len(self.history) >= self.window:
            velocity = current_imbalance - self.history[-self.window]
        else:
            velocity = 0
            
        self.history.append(current_imbalance)
        if len(self.history) > self.window * 2:
            self.history = self.history[-self.window * 2:]
            
        return velocity
    
    def calculate_imbalance(self, orderbook: Dict) -> float:
        """计算失衡"""
        bids = orderbook.get('bids', [])[:5]
        asks = orderbook.get('asks', [])[:5]
        
        bid_volume = sum([v for p, v in bids])
        ask_volume = sum([v for p, v in asks])
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
        return imbalance

class OrderBookAsymmetryFactor(OrderBookFactor):
    """订单簿不对称性因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_asymmetry', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿不对称性"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        if len(bids) < 2 or len(asks) < 2:
            return 0
        
        bid_prices = [p for p, v in bids]
        bid_volumes = [v for p, v in bids]
        ask_prices = [p for p, v in asks]
        ask_volumes = [v for p, v in asks]
        
        bid_distribution = np.std(bid_volumes) / (np.mean(bid_volumes) + 1e-8)
        ask_distribution = np.std(ask_volumes) / (np.mean(ask_volumes) + 1e-8)
        
        asymmetry = np.abs(bid_distribution - ask_distribution)
        return asymmetry

class OrderBookConcentrationFactor(OrderBookFactor):
    """订单簿集中度因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_concentration', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿集中度"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        if len(bids) < 1 or len(asks) < 1:
            return 0
        
        bid_total = sum([v for p, v in bids])
        ask_total = sum([v for p, v in asks])
        
        bid_concentration = bids[0][1] / (bid_total + 1e-8)
        ask_concentration = asks[0][1] / (ask_total + 1e-8)
        
        concentration = (bid_concentration + ask_concentration) / 2
        return concentration

class OrderBookLiquidityFactor(OrderBookFactor):
    """订单簿流动性因子"""
    
    def __init__(self, depth: int = 5):
        super().__init__('orderbook_liquidity', depth)
        self.depth = depth
        
    def calculate(self, orderbook: Dict) -> float:
        """计算订单簿流动性"""
        bids = orderbook.get('bids', [])[:self.depth]
        asks = orderbook.get('asks', [])[:self.depth]
        
        if not bids or not asks:
            return 0
        
        bid_volume = sum([v for p, v in bids])
        ask_volume = sum([v for p, v in asks])
        
        bid_price = sum([p * v for p, v in bids]) / (bid_volume + 1e-8)
        ask_price = sum([p * v for p, v in asks]) / (ask_volume + 1e-8)
        
        spread = ask_price - bid_price
        
        liquidity = (bid_volume + ask_volume) / (spread + 1e-8)
        return liquidity