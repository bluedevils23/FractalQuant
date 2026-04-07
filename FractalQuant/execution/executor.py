"""
订单执行模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import asyncio
import ccxt

logger = logging.getLogger(__name__)

class Order:
    """订单类"""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        timestamp: datetime = None,
        status: str = 'pending',
        exchange: str = None
    ):
        """
        初始化订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对
            side: 买卖方向 ('buy' or 'sell')
            order_type: 订单类型 ('market' or 'limit')
            quantity: 数量
            price: 价格（限价单）
            timestamp: 时间戳
            status: 状态
            exchange: 交易所
        """
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.status = status
        self.exchange = exchange
        self.filled_quantity = 0
        self.filled_price = 0
        self.fee = 0
        
    def is_filled(self) -> bool:
        """检查订单是否已成交"""
        return self.status == 'filled'
    
    def is_cancelled(self) -> bool:
        """检查订单是否已取消"""
        return self.status == 'cancelled'
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'exchange': self.exchange,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'fee': self.fee
        }

class OrderExecutor:
    """订单执行器"""
    
    def __init__(
        self,
        exchange: ccxt.Exchange = None,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        初始化订单执行器
        
        Args:
            exchange: 交易所实例
            commission: 手续费
            slippage: 滑点
        """
        self.exchange = exchange
        self.commission = commission
        self.slippage = slippage
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Order]:
        """
        创建订单
        
        Args:
            symbol: 交易对
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格
            
        Returns:
            订单对象
        """
        if not self.exchange:
            logger.error("Exchange not initialized")
            return None
            
        try:
            order_id = f"order_{datetime.now().timestamp()}"
            
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': quantity
            }
            
            if order_type == 'limit' and price:
                order_params['price'] = price
                
            # 调用交易所API
            order_response = await self.exchange.create_order(**order_params)
            
            # 创建订单对象
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                exchange=self.exchange.id
            )
            
            self.orders[order_id] = order
            self.pending_orders[order_id] = order
            
            logger.info(f"Created {side} order for {quantity} {symbol} at {price}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功
        """
        if order_id not in self.pending_orders:
            return False
            
        order = self.pending_orders[order_id]
        
        if not self.exchange:
            return False
            
        try:
            await self.exchange.cancel_order(order_id, order.symbol)
            order.status = 'cancelled'
            del self.pending_orders[order_id]
            
            logger.info(f"Cancelled order {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def update_order_status(self, order_id: str) -> Optional[Order]:
        """
        更新订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            更新后的订单
        """
        if order_id not in self.pending_orders:
            return None
            
        order = self.pending_orders[order_id]
        
        if not self.exchange:
            return order
            
        try:
            order_status = await self.exchange.fetch_order(order_id, order.symbol)
            
            order.status = order_status['status']
            order.filled_quantity = order_status['filled']
            order.filled_price = order_status['price']
            
            if order.status == 'filled':
                # 计算手续费
                order.fee = order.filled_quantity * order.filled_price * self.commission
                del self.pending_orders[order_id]
                
            return order
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return order
    
    async def execute_signal(
        self,
        symbol: str,
        signal: int,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Order]:
        """
        执行信号
        
        Args:
            symbol: 交易对
            signal: 信号 (1: 买入, -1: 卖出, 0: 平仓)
            quantity: 数量
            price: 价格
            
        Returns:
            订单对象
        """
        if signal == 0:
            return None
            
        side = 'buy' if signal > 0 else 'sell'
        order_type = 'market' if price is None else 'limit'
        
        return await self.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
    
    def get_pending_orders(self) -> List[Order]:
        """获取待成交订单"""
        return list(self.pending_orders.values())
    
    def get_order_history(self) -> List[Order]:
        """获取订单历史"""
        return list(self.orders.values())

class MarketOrderExecutor(OrderExecutor):
    """市价单执行器"""
    
    def __init__(
        self,
        exchange: ccxt.Exchange = None,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        super().__init__(exchange, commission, slippage)
        
    async def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Optional[Order]:
        """
        执行市价单
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 数量
            
        Returns:
            订单对象
        """
        return await self.execute_signal(symbol, 1 if side == 'buy' else -1, quantity)

class LimitOrderExecutor(OrderExecutor):
    """限价单执行器"""
    
    def __init__(
        self,
        exchange: ccxt.Exchange = None,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_slippage: float = 0.002
    ):
        super().__init__(exchange, commission, slippage)
        self.max_slippage = max_slippage
        
    async def execute_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Optional[Order]:
        """
        执行限价单
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 数量
            price: 价格
            
        Returns:
            订单对象
        """
        return await self.execute_signal(symbol, 1 if side == 'buy' else -1, quantity, price)

class IcebergOrderExecutor(OrderExecutor):
    """冰山单执行器"""
    
    def __init__(
        self,
        exchange: ccxt.Exchange = None,
        commission: float = 0.001,
        slippage: float = 0.0005,
        chunk_size: float = 0.1
    ):
        super().__init__(exchange, commission, slippage)
        self.chunk_size = chunk_size
        
    async def execute_iceberg_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> List[Order]:
        """
        执行冰山单
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 总数量
            price: 价格
            
        Returns:
            订单列表
        """
        orders = []
        remaining = quantity
        
        while remaining > 0:
            chunk = min(self.chunk_size, remaining)
            
            order = await self.execute_signal(
                symbol,
                1 if side == 'buy' else -1,
                chunk,
                price
            )
            
            if order:
                orders.append(order)
                
            remaining -= chunk
            
        return orders