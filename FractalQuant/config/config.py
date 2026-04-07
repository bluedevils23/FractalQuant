"""
配置管理模块
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

load_dotenv()

@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str
    api_key: str
    api_secret: str
    api_passphrase: Optional[str] = None
    test_mode: bool = True

@dataclass
class FactorConfig:
    """因子配置"""
    # 窗口参数
    window_short: int = 5
    window_medium: int = 20
    window_long: int = 60
    volatility_window: int = 20
    momentum_window: int = 30
    volume_window: int = 10
    correlation_window: int = 50
    
    # 高级因子参数
    lyapunov_window: int = 50
    hurst_window: int = 50
    ml_forecast_steps: int = 5
    ml_hidden_size: int = 10
    
    # 微观结构因子参数
    orderflow_window: int = 50
    liquidity_window: int = 50
    
    # 跨市场因子参数
    crossmarket_window: int = 50
    arbitrage_threshold: float = 0.001

@dataclass
class RiskConfig:
    """风险管理配置"""
    max_position_size: float = 0.1
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.10
    leverage: int = 10
    max_drawdown: float = 0.15
    max_positions: int = 10

@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 100000
    commission: float = 0.001
    slippage: float = 0.0005

class Config:
    """配置管理器"""
    
    def __init__(self):
        self.exchanges = self._load_exchanges()
        self.factors = FactorConfig(
            window_short=int(os.getenv('FACTOR_WINDOW_SHORT', 5)),
            window_medium=int(os.getenv('FACTOR_WINDOW_MEDIUM', 20)),
            window_long=int(os.getenv('FACTOR_WINDOW_LONG', 60)),
            volatility_window=int(os.getenv('VOLATILITY_WINDOW', 20)),
            lyapunov_window=int(os.getenv('LYAPUNOV_WINDOW', 50)),
            hurst_window=int(os.getenv('HURST_WINDOW', 50)),
            ml_forecast_steps=int(os.getenv('ML_FORECAST_STEPS', 5)),
            ml_hidden_size=int(os.getenv('ML_HIDDEN_SIZE', 10)),
            orderflow_window=int(os.getenv('ORDERFLOW_WINDOW', 50)),
            liquidity_window=int(os.getenv('LIQUIDITY_WINDOW', 50)),
            crossmarket_window=int(os.getenv('CROSSMARKET_WINDOW', 50)),
            arbitrage_threshold=float(os.getenv('ARBITRAGE_THRESHOLD', 0.001)),
        )
        self.risk = RiskConfig(
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', 0.1)),
            stop_loss_threshold=float(os.getenv('STOP_LOSS_THRESHOLD', 0.05)),
            take_profit_threshold=float(os.getenv('TAKE_PROFIT_THRESHOLD', 0.10)),
            leverage=int(os.getenv('LEVERAGE', 10)),
        )
        self.backtest = BacktestConfig(
            start_date=os.getenv('BACKTEST_START_DATE', '2023-01-01'),
            end_date=os.getenv('BACKTEST_END_DATE', '2023-12-31'),
            initial_capital=float(os.getenv('BACKTEST_INITIAL_CAPITAL', 100000)),
            commission=float(os.getenv('BACKTEST_COMMISSION', 0.001)),
        )
        
        self.trading_pairs = os.getenv('TRADING_PAIRS', 'BTC/USDT,ETH/USDT,SOL/USDT').split(',')
        self.target_returns = float(os.getenv('TARGET_RETURNS', 0.001))
        self.rebalance_frequency = os.getenv('REBALANCE_FREQUENCY', '1h')
        
    def _load_exchanges(self) -> Dict[str, ExchangeConfig]:
        """加载交易所配置"""
        exchanges = {}
        
        binance_key = os.getenv('BINANCE_API_KEY')
        binance_secret = os.getenv('BINANCE_API_SECRET')
        if binance_key and binance_secret:
            exchanges['binance'] = ExchangeConfig(
                name='binance',
                api_key=binance_key,
                api_secret=binance_secret,
                test_mode=True
            )
            
        okx_key = os.getenv('OKX_API_KEY')
        okx_secret = os.getenv('OKX_API_SECRET')
        okx_passphrase = os.getenv('OKX_PASSPHRASE')
        if okx_key and okx_secret:
            exchanges['okx'] = ExchangeConfig(
                name='okx',
                api_key=okx_key,
                api_secret=okx_secret,
                api_passphrase=okx_passphrase,
                test_mode=True
            )
            
        return exchanges
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """获取交易所配置"""
        return self.exchanges.get(exchange_name.lower())
    
    def get_all_pairs(self) -> List[str]:
        """获取所有交易对"""
        return self.trading_pairs

# 全局配置实例
config = Config()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)