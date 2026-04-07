"""
因子组合和评分模块
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

from .base import BaseFactor
from .price import ReturnsFactor, PriceMomentumFactor, VolumePriceTrendFactor
from .volatility import HistoricalVolatilityFactor, ParkinsonVolatilityFactor
from .advanced import FutureReturnsFactor, CorrelationFactor, HurstExponentFactor
from .trend import MACDFactor, RSIFactor, EMAFactor
from .orderbook import OrderBookImbalanceFactor
from .ml import MLForecastFactor, MLAnomalyDetectionFactor, ClusteringRegimeFactor
from .microstructure import OrderFlowImbalanceFactor, LiquidityRatioFactor, VolumeWeightedPriceFactor
from .crossmarket import CrossMarketCorrelationFactor, ArbitrageOpportunityFactor, RelativeStrengthFactor

class FactorCombiner:
    """因子组合器"""
    
    def __init__(self, factors: List[BaseFactor], method: str = 'equal', 
                 adaptive_weights: bool = False, decay: float = 0.95):
        """
        初始化因子组合器
        
        Args:
            factors: 因子列表
            method: 组合方法 ('equal', 'rank', 'zscore', 'pca', 'optimal', 'adaptive')
            adaptive_weights: 是否使用自适应权重
            decay: 自适应权重衰减因子
        """
        self.factors = factors
        self.method = method
        self.adaptive_weights = adaptive_weights
        self.decay = decay
        self.scaler = StandardScaler()
        self.weights_history = []
        
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        factor_values = {}
        
        for factor in self.factors:
            try:
                result = factor.calculate(df)
                factor_values[factor.name] = result
            except Exception as e:
                print(f"Error calculating factor {factor.name}: {e}")
                factor_values[factor.name] = pd.Series(np.nan, index=df.index)
        
        return pd.DataFrame(factor_values)
    
    def calculate_optimal_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算最优权重（基于风险平价）"""
        factor_df = self.calculate_all(df)
        factor_df = factor_df.dropna()
        
        if len(factor_df) < 20 or len(factor_df.columns) < 2:
            return {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
        
        try:
            returns = factor_df.pct_change().dropna()
            
            cov_matrix = returns.cov()
            
            if cov_matrix is None or len(cov_matrix) < 2:
                return {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
            
            inv_diag = 1.0 / np.diag(cov_matrix)
            inv_diag_matrix = np.diag(inv_diag)
            
            raw_weights = inv_diag_matrix @ np.ones(len(cov_matrix))
            raw_weights = raw_weights / raw_weights.sum()
            
            vol_target = 1.0 / np.sqrt(np.diag(cov_matrix))
            risk_weights = vol_target * raw_weights
            risk_weights = risk_weights / risk_weights.sum()
            
            return {col: risk_weights[i] for i, col in enumerate(cov_matrix.columns)}
        except:
            return {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
    
    def calculate_adaptive_weights(self, df: pd.DataFrame, performance: pd.Series) -> Dict[str, float]:
        """计算自适应权重（基于因子表现）"""
        factor_df = self.calculate_all(df)
        
        if len(performance) < 10:
            return {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
        
        try:
            factor_returns = {}
            for col in factor_df.columns:
                factor_returns[col] = performance * factor_df[col].shift(1)
            
            weights = {}
            total_weight = 0
            
            for col in factor_returns:
                if col in performance.index and len(factor_returns[col].dropna()) > 5:
                    mean_return = factor_returns[col].dropna().mean()
                    weights[col] = max(0, mean_return)
                    total_weight += weights[col]
            
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                weights = {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
            
            return weights
        except:
            return {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
    
    def combine_factors(self, df: pd.DataFrame, performance: pd.Series = None) -> pd.Series:
        """组合因子"""
        factor_df = self.calculate_all(df)
        
        if self.method == 'equal':
            combined = factor_df.mean(axis=1)
            
        elif self.method == 'rank':
            factor_df = factor_df.rank(pct=True)
            combined = factor_df.mean(axis=1)
            
        elif self.method == 'zscore':
            factor_df = factor_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            combined = factor_df.mean(axis=1)
            
        elif self.method == 'pca':
            factor_df = factor_df.fillna(0)
            factor_df = factor_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            
            factor_df = factor_df.loc[:, factor_df.std() > 0]
            
            if len(factor_df.columns) > 0:
                pca = PCA(n_components=min(3, len(factor_df.columns)))
                principal_components = pca.fit_transform(factor_df)
                weights = np.abs(pca.components_[0])
                weights = weights / weights.sum()
                combined = pd.Series(
                    principal_components @ weights,
                    index=factor_df.index
                )
            else:
                combined = pd.Series(0, index=df.index)
                
        elif self.method == 'optimal':
            weights = self.calculate_optimal_weights(df)
            factor_df = factor_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            combined = sum(weights[col] * factor_df[col] for col in weights)
            
        elif self.method == 'adaptive':
            if performance is not None and len(performance) > 10:
                weights = self.calculate_adaptive_weights(df, performance)
            else:
                weights = {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}
            
            factor_df = factor_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
            combined = sum(weights[col] * factor_df[col] for col in weights)
            
        else:
            combined = factor_df.mean(axis=1)
            
        return combined

class FactorScore:
    """因子评分"""
    
    def __init__(self, window: int = 20):
        self.window = window
        
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0
        mean = returns.mean()
        std = returns.std()
        if std == 0:
            return 0
        return mean / std * np.sqrt(252 * 24 * 60)
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        if len(returns) < 2:
            return 0
        mean = returns.mean()
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0
        return mean / downside.std() * np.sqrt(252 * 24 * 60)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark: float = 0) -> float:
        """计算信息比率"""
        if len(returns) < 2:
            return 0
        active_return = returns - benchmark
        active_volatility = active_return.std()
        if active_volatility == 0:
            return 0
        return active_return.mean() / active_volatility * np.sqrt(252 * 24 * 60)
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """计算胜率"""
        if len(returns) == 0:
            return 0
        return (returns > 0).sum() / len(returns)
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """计算盈亏比"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 0
        return gains / losses
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡玛比率（收益/最大回撤）"""
        if len(returns) < 2:
            return 0
        mean = returns.mean()
        max_dd = self.calculate_max_drawdown(returns)
        if max_dd == 0:
            return 0
        return mean / abs(max_dd) * np.sqrt(252 * 24 * 60)
    
    def calculate_stability(self, returns: pd.Series) -> float:
        """计算收益稳定性（R-squared）"""
        if len(returns) < 10:
            return 0
        cumulative = (1 + returns).cumprod()
        x = np.arange(len(cumulative))
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, np.log(cumulative + 1e-8))
            return r_value ** 2
        except:
            return 0
    
    def calculate_turnover_ratio(self, signals: pd.Series) -> float:
        """计算换手率（信号变化频率）"""
        if len(signals) < 2:
            return 0
        changes = (signals.diff() != 0).sum()
        return changes / (len(signals) - 1)
    
    def calculate_factor_rank_correlation(self, factor_df: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """计算因子排名与收益的相关性"""
        correlations = {}
        for col in factor_df.columns:
            factor_rank = factor_df[col].rank(pct=True)
            try:
                corr, _ = spearmanr(factor_rank, returns)
                correlations[col] = corr if not np.isnan(corr) else 0
            except:
                correlations[col] = 0
        return correlations

class MultiFactorSignal:
    """多因子信号生成"""
    
    def __init__(
        self, 
        price_factors: List[BaseFactor] = None,
        volatility_factors: List[BaseFactor] = None,
        trend_factors: List[BaseFactor] = None,
        orderbook_factors: List[BaseFactor] = None,
        weights: Dict[str, float] = None
    ):
        """
        初始化多因子信号生成器
        
        Args:
            price_factors: 价格因子列表
            volatility_factors: 波动率因子列表
            trend_factors: 趋势因子列表
            orderbook_factors: 订单簿因子列表
            weights: 因子权重字典
        """
        self.price_factors = price_factors or [
            ReturnsFactor(window=5),
            PriceMomentumFactor(window=20),
        ]
        
        self.volatility_factors = volatility_factors or [
            HistoricalVolatilityFactor(window=20),
            ParkinsonVolatilityFactor(window=20),
        ]
        
        self.trend_factors = trend_factors or [
            MACDFactor(),
            RSIFactor(),
            EMAFactor(window=20),
        ]
        
        self.orderbook_factors = orderbook_factors or [
            OrderBookImbalanceFactor(),
        ]
        
        self.ml_factors = ml_factors or [
            MLForecastFactor(model_type='linear'),
            MLAnomalyDetectionFactor(),
            ClusteringRegimeFactor(),
        ]
        
        self.microstructure_factors = microstructure_factors or [
            OrderFlowImbalanceFactor(),
            LiquidityRatioFactor(),
            VolumeWeightedPriceFactor(),
        ]
        
        self.crossmarket_factors = crossmarket_factors or [
            CrossMarketCorrelationFactor(),
            ArbitrageOpportunityFactor(),
            RelativeStrengthFactor(),
        ]
        
        self.weights = weights or self._initialize_weights()
        
        self.combiner = FactorCombiner(
            self.price_factors + self.volatility_factors + 
            self.trend_factors + self.orderbook_factors +
            self.ml_factors + self.microstructure_factors + self.crossmarket_factors,
            method='pca'
        )
        
        self.score_calculator = FactorScore()
        
    def _initialize_weights(self) -> Dict[str, float]:
        """初始化因子权重"""
        all_factors = (
            self.price_factors + self.volatility_factors + 
            self.trend_factors + self.orderbook_factors +
            self.ml_factors + self.microstructure_factors + self.crossmarket_factors
        )
        return {f.name: 1.0 / len(all_factors) for f in all_factors}
    
    def generate_signal(self, df: pd.DataFrame, orderbook: Dict = None) -> Dict:
        """
        生成交易信号
        
        Args:
            df: K线数据
            orderbook: 订单簿数据
            
        Returns:
            信号字典
        """
        # 计算基础因子
        base_factors = self.price_factors + self.volatility_factors + self.trend_factors
        base_df = pd.DataFrame({f.name: f.calculate(df) for f in base_factors})
        
        # 计算订单簿因子
        if orderbook:
            ob_factors = {}
            for factor in self.orderbook_factors:
                try:
                    ob_factors[factor.name] = factor.calculate(orderbook)
                except:
                    ob_factors[factor.name] = 0
            ob_df = pd.DataFrame([ob_factors])
        else:
            ob_df = pd.DataFrame([{f.name: 0 for f in self.orderbook_factors}])
        
        # 组合因子
        all_factors = pd.concat([base_df.tail(1), ob_df], axis=1)
        
        # 计算综合得分
        score = 0
        for factor_name, weight in self.weights.items():
            if factor_name in all_factors.columns:
                value = all_factors[factor_name].iloc[-1]
                if pd.notna(value):
                    score += weight * value
        
        # 归一化得分
        if score > 0:
            signal = 1  # 买入
        elif score < 0:
            signal = -1  # 卖出
        else:
            signal = 0  # 持平
            
        # 计算信号强度
        signal_strength = abs(score) / (len(self.weights) + 1e-8)
        
        return {
            'signal': signal,
            'strength': signal_strength,
            'score': score,
            'factors': all_factors.iloc[-1].to_dict() if not all_factors.empty else {}
        }
    
    def backtest_signal(self, df: pd.DataFrame) -> Dict:
        """
        回测信号效果
        
        Args:
            df: K线数据
            
        Returns:
            回测结果字典
        """
        # 计算信号
        signals = []
        for i in range(20, len(df)):
            window_df = df.iloc[:i+1]
            signal = self.generate_signal(window_df)
            signals.append(signal['signal'])
        
        signals = pd.Series(signals, index=df.index[20:])
        
        # 计算策略收益
        returns = df['close'].pct_change()[20:]
        strategy_returns = signals.shift(1) * returns
        
        # 计算统计指标
        metrics = {
            'sharpe_ratio': self.score_calculator.calculate_sharpe_ratio(strategy_returns),
            'sortino_ratio': self.score_calculator.calculate_sortino_ratio(strategy_returns),
            'max_drawdown': self.score_calculator.calculate_max_drawdown(strategy_returns),
            'calmar_ratio': self.score_calculator.calculate_calmar_ratio(strategy_returns),
            'stability': self.score_calculator.calculate_stability(strategy_returns),
            'win_rate': self.score_calculator.calculate_win_rate(strategy_returns),
            'profit_factor': self.score_calculator.calculate_profit_factor(strategy_returns),
            'total_return': (1 + strategy_returns).prod() - 1,
            'turnover_ratio': self.score_calculator.calculate_turnover_ratio(signals),
        }
        
        return metrics
    
    def get_factor_analysis(self, df: pd.DataFrame) -> Dict:
        """
        获取因子分析报告
        
        Args:
            df: K线数据
            
        Returns:
            因子分析字典
        """
        factor_df = self.combiner.calculate_all(df)
        
        returns = df['close'].pct_change().shift(-1).dropna()
        factor_df = factor_df.shift(1).dropna()
        
        common_index = factor_df.index.intersection(returns.index)
        factor_df = factor_df.loc[common_index]
        returns = returns.loc[common_index]
        
        correlations = self.score_calculator.calculate_factor_rank_correlation(factor_df, returns)
        
        factor_stats = {}
        for col in factor_df.columns:
            factor_stats[col] = {
                'mean': factor_df[col].mean(),
                'std': factor_df[col].std(),
                'min': factor_df[col].min(),
                'max': factor_df[col].max(),
                'skewness': stats.skew(factor_df[col].dropna()),
                'kurtosis': stats.kurtosis(factor_df[col].dropna()),
                'correlation_with_returns': correlations.get(col, 0),
            }
        
        return {
            'factor_correlations': correlations,
            'factor_statistics': factor_stats,
            'total_factors': len(factor_df.columns),
        }