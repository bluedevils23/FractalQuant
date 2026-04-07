"""
数据模型定义
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class TickData:
    """tick数据"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_price: Optional[float] = None
    ask_volume: Optional[float] = None
    exchange: Optional[str] = None
    trade_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid_price': self.bid_price,
            'bid_volume': self.bid_volume,
            'ask_price': self.ask_price,
            'ask_volume': self.ask_volume,
            'exchange': self.exchange,
            'trade_id': self.trade_id
        }

@dataclass
class BarData:
    """K线数据"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: Optional[float] = None
    exchange: Optional[str] = None
    trades_count: Optional[int] = None
    vwap: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'turnover': self.turnover,
            'exchange': self.exchange,
            'trades_count': self.trades_count,
            'vwap': self.vwap
        }

@dataclass
class OrderBookData:
    """订单簿数据"""
    timestamp: datetime
    symbol: str
    bids: List[tuple]  # [(price, volume), ...]
    asks: List[tuple]
    exchange: Optional[str] = None
    analysis: Optional[Dict] = None
    order_count: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'bids': self.bids,
            'asks': self.asks,
            'exchange': self.exchange,
            'analysis': self.analysis,
            'order_count': self.order_count
        }

@dataclass 
class FactorData:
    """因子数据"""
    timestamp: datetime
    symbol: str
    values: dict  # 因子名称 -> 值
    
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([{
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            **self.values
        }])
        df.set_index('timestamp', inplace=True)
        return df

@dataclass
class TradeData:
    """交易数据"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'side': self.side,
            'trade_id': self.trade_id,
            'exchange': self.exchange
        }

@dataclass
class OnChainData:
    """链上数据"""
    timestamp: datetime
    symbol: str
    metric: str
    value: float
    unit: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'metric': self.metric,
            'value': self.value,
            'unit': self.unit,
            'exchange': self.exchange,
            'metadata': self.metadata
        }

@dataclass
class AlternativeData:
    """另类数据"""
    timestamp: datetime
    symbol: str
    source: str
    content: str
    sentiment: Optional[float] = None
    volume: int = 0
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'source': self.source,
            'content': self.content,
            'sentiment': self.sentiment,
            'volume': self.volume,
            'metadata': self.metadata
        }

@dataclass
class MarketState:
    """市场状态"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    volatility: float
    trend: str  # 'up', 'down', 'sideways'
    regime: str  # 'bull', 'bear', 'normal'
    liquidity: float
    sentiment: float
    metadata: Optional[Dict] = None

@dataclass
class MarketEvent:
    """市场事件"""
    timestamp: datetime
    symbol: str
    event_type: str  # 'price_spike', 'volume_spike', 'order_book_imbalance', etc.
    severity: int  # 1-10
    description: str
    metadata: Optional[Dict] = None

class MarketData:
    """市场数据容器"""
    
    def __init__(self):
        self.bars: Dict[str, List[BarData]] = {}
        self.ticks: Dict[str, List[TickData]] = {}
        self.order_books: Dict[str, List[OrderBookData]] = {}
        self.trades: Dict[str, List[TradeData]] = {}
        self.onchain: Dict[str, List[OnChainData]] = {}
        self.alternative: Dict[str, List[AlternativeData]] = {}
        self.states: Dict[str, MarketState] = {}
        self.events: List[MarketEvent] = []
        
    def add_bar(self, bar: BarData):
        """添加K线数据"""
        if bar.symbol not in self.bars:
            self.bars[bar.symbol] = []
        self.bars[bar.symbol].append(bar)
        
    def add_tick(self, tick: TickData):
        """添加tick数据"""
        if tick.symbol not in self.ticks:
            self.ticks[tick.symbol] = []
        self.ticks[tick.symbol].append(tick)
        
    def add_order_book(self, order_book: OrderBookData):
        """添加订单簿数据"""
        if order_book.symbol not in self.order_books:
            self.order_books[order_book.symbol] = []
        self.order_books[order_book.symbol].append(order_book)
        
    def add_trade(self, trade: TradeData):
        """添加交易数据"""
        if trade.symbol not in self.trades:
            self.trades[trade.symbol] = []
        self.trades[trade.symbol].append(trade)
        
    def add_onchain_data(self, data: OnChainData):
        """添加链上数据"""
        key = f"{data.symbol}_{data.metric}"
        if key not in self.onchain:
            self.onchain[key] = []
        self.onchain[key].append(data)
        
    def add_alternative_data(self, data: AlternativeData):
        """添加另类数据"""
        key = f"{data.symbol}_{data.source}"
        if key not in self.alternative:
            self.alternative[key] = []
        self.alternative[key].append(data)
        
    def set_market_state(self, state: MarketState):
        """设置市场状态"""
        self.states[state.symbol] = state
        
    def add_event(self, event: MarketEvent):
        """添加市场事件"""
        self.events.append(event)
        
    def get_bars(self, symbol: str, count: int = None) -> List[BarData]:
        """获取K线数据"""
        if symbol not in self.bars:
            return []
        bars = self.bars[symbol]
        if count:
            return bars[-count:]
        return bars
    
    def get_ticks(self, symbol: str, count: int = None) -> List[TickData]:
        """获取tick数据"""
        if symbol not in self.ticks:
            return []
        ticks = self.ticks[symbol]
        if count:
            return ticks[-count:]
        return ticks
    
    def get_order_books(self, symbol: str, count: int = None) -> List[OrderBookData]:
        """获取订单簿数据"""
        if symbol not in self.order_books:
            return []
        order_books = self.order_books[symbol]
        if count:
            return order_books[-count:]
        return order_books
    
    def get_trades(self, symbol: str, count: int = None) -> List[TradeData]:
        """获取交易数据"""
        if symbol not in self.trades:
            return []
        trades = self.trades[symbol]
        if count:
            return trades[-count:]
        return trades
    
    def get_onchain_data(self, symbol: str, metric: str) -> List[OnChainData]:
        """获取链上数据"""
        key = f"{symbol}_{metric}"
        return self.onchain.get(key, [])
    
    def get_alternative_data(self, symbol: str, source: str = None) -> List[AlternativeData]:
        """获取另类数据"""
        if source:
            key = f"{symbol}_{source}"
            return self.alternative.get(key, [])
        else:
            all_data = []
            for key, data in self.alternative.items():
                if symbol in key:
                    all_data.extend(data)
            return all_data
    
    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """获取市场状态"""
        return self.states.get(symbol)
    
    def get_events(self, symbol: str = None, event_type: str = None) -> List[MarketEvent]:
        """获取市场事件"""
        events = self.events
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events
    
    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        """获取最新K线"""
        if symbol not in self.bars or not self.bars[symbol]:
            return None
        return self.bars[symbol][-1]
    
    def to_dataframe(self, symbol: str, data_type: str = 'bars') -> pd.DataFrame:
        """转换为DataFrame"""
        if data_type == 'bars':
            if symbol not in self.bars:
                return pd.DataFrame()
            data = [bar.to_dict() for bar in self.bars[symbol]]
        elif data_type == 'ticks':
            if symbol not in self.ticks:
                return pd.DataFrame()
            data = [tick.to_dict() for tick in self.ticks[symbol]]
        elif data_type == 'trades':
            if symbol not in self.trades:
                return pd.DataFrame()
            data = [trade.to_dict() for trade in self.trades[symbol]]
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
    
    def get_summary(self) -> Dict:
        """获取数据摘要"""
        summary = {
            'total_symbols': len(set(
                list(self.bars.keys()) + 
                list(self.ticks.keys()) + 
                list(self.order_books.keys()) +
                list(self.trades.keys())
            )),
            'total_bars': sum(len(bars) for bars in self.bars.values()),
            'total_ticks': sum(len(ticks) for ticks in self.ticks.values()),
            'total_order_books': sum(len(books) for books in self.order_books.values()),
            'total_trades': sum(len(trades) for trades in self.trades.values()),
            'total_onchain': sum(len(data) for data in self.onchain.values()),
            'total_alternative': sum(len(data) for data in self.alternative.values()),
            'total_events': len(self.events)
        }
        return summary
