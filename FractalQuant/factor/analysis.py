"""
因子回测和分析模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from factor.combiner import MultiFactorSignal
from factor.price import ReturnsFactor, PriceMomentumFactor
from factor.volatility import HistoricalVolatilityFactor
from factor.trend import MACDFactor, RSIFactor

logger = logging.getLogger(__name__)

class FactorBacktest:
    """因子回测"""
    
    def __init__(
        self,
        factors: List = None,
        window: int = 20
    ):
        """
        初始化因子回测
        
        Args:
            factors: 因子列表
            window: 回看窗口
        """
        self.factors = factors or [
            ReturnsFactor(window=5),
            PriceMomentumFactor(window=20),
            HistoricalVolatilityFactor(window=20),
            MACDFactor(),
            RSIFactor(),
        ]
        
        self.window = window
        self.factor_scores: Dict[str, List[float]] = {}
        
    def backtest_single_factor(
        self,
        df: pd.DataFrame,
        factor_name: str,
        forward_window: int = 5
    ) -> Dict:
        """
        回测单个因子
        
        Args:
            df: 数据
            factor_name: 因子名称
            forward_window: 前向窗口
            
        Returns:
            回测结果
        """
        # 计算因子值
        if factor_name == 'returns':
            factor_values = df['close'].pct_change(periods=5)
        elif factor_name == 'momentum':
            factor_values = df['close'] / df['close'].shift(20) - 1
        elif factor_name == 'volatility':
            log_returns = np.log(df['close'] / df['close'].shift(1))
            factor_values = log_returns.rolling(window=20).std()
        elif factor_name == 'macd':
            ema_fast = df['close'].ewm(span=12).mean()
            ema_slow = df['close'].ewm(span=26).mean()
            factor_values = ema_fast - ema_slow
        elif factor_name == 'rsi':
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            factor_values = 100 - (100 / (1 + rs))
        else:
            factor_values = pd.Series(0, index=df.index)
            
        # 计算未来收益
        future_returns = df['close'].shift(-forward_window) / df['close'] - 1
        
        # 计算相关性
        correlation = factor_values.corr(future_returns)
        
        # 分组测试
        factor_quantiles = pd.qcut(factor_values, 5, labels=False, duplicates='drop')
        group_returns = future_returns.groupby(factor_quantiles).mean()
        
        return {
            'correlation': correlation,
            'group_returns': group_returns,
            't_stat': self._calculate_t_stat(factor_values, future_returns),
            'p_value': self._calculate_p_value(correlation, len(df))
        }
    
    def _calculate_t_stat(self, factor_values: pd.Series, future_returns: pd.Series) -> float:
        """计算t统计量"""
        correlation = factor_values.corr(future_returns)
        n = len(factor_values.dropna())
        if n <= 2:
            return 0
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation ** 2)
        return t_stat
    
    def _calculate_p_value(self, correlation: float, n: int) -> float:
        """计算p值（简化版本）"""
        from scipy import stats
        if n <= 2:
            return 1
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation ** 2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        return p_value
    
    def backtest_all_factors(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """回测所有因子"""
        results = {}
        
        for factor in self.factors:
            try:
                result = self.backtest_single_factor(df, factor.name)
                results[factor.name] = result
                logger.info(f"Backtested {factor.name}: correlation={result['correlation']:.4f}")
            except Exception as e:
                logger.error(f"Error backtesting {factor.name}: {e}")
                
        return results
    
    def rank_factors(self, results: Dict[str, Dict]) -> List[str]:
        """对因子进行排名"""
        factor_scores = []
        
        for factor_name, result in results.items():
            score = result['correlation'] * 100 - abs(result['p_value'] - 0.5) * 10
            factor_scores.append((factor_name, score))
            
        factor_scores.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in factor_scores]

class FactorAnalysis:
    """因子分析"""
    
    def __init__(self):
        """初始化因子分析"""
        self.factor_correlations: Dict[str, Dict[str, float]] = {}
        
    def calculate_factor_correlations(
        self,
        df: pd.DataFrame,
        factors: List
    ) -> pd.DataFrame:
        """
        计算因子相关性
        
        Args:
            df: 数据
            factors: 因子列表
            
        Returns:
            相关性矩阵
        """
        factor_values = {}
        
        for factor in factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        factor_df = pd.DataFrame(factor_values)
        correlation_matrix = factor_df.corr()
        
        return correlation_matrix
    
    def analyze_factor_IC(
        self,
        df: pd.DataFrame,
        factors: List,
        forward_window: int = 5
    ) -> Dict:
        """
        分析因子IC值
        
        Args:
            df: 数据
            factors: 因子列表
            forward_window: 前向窗口
            
        Returns:
            IC值分析
        """
        future_returns = df['close'].shift(-forward_window) / df['close'] - 1
        
        ic_values = {}
        
        for factor in factors:
            try:
                factor_values = factor.calculate(df)
                ic = factor_values.corr(future_returns)
                ic_values[factor.name] = ic
            except Exception as e:
                logger.error(f"Error calculating IC for {factor.name}: {e}")
                
        return ic_values
    
    def visualize_factor_IC(self, ic_values: Dict[str, float]):
        """可视化IC值"""
        if not ic_values:
            return
            
        plt.figure(figsize=(12, 6))
        
        # IC值条形图
        plt.subplot(1, 2, 1)
        factors = list(ic_values.keys())
        values = list(ic_values.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        plt.bar(factors, values, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Factor IC Values')
        plt.xticks(rotation=45)
        plt.ylabel('IC Value')
        
        # 累积IC值
        plt.subplot(1, 2, 2)
        sorted_ic = sorted(ic_values.values())
        plt.plot(range(len(sorted_ic)), sorted_ic, marker='o')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Cumulative IC Values')
        plt.xlabel('Factor Rank')
        plt.ylabel('IC Value')
        
        plt.tight_layout()
        plt.savefig('factor_ic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Factor IC analysis saved to factor_ic_analysis.png")

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        """初始化性能分析器"""
        self.metrics = {}
        
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        计算性能指标
        
        Args:
            equity_curve: 权益曲线
            returns: 收益率序列
            
        Returns:
            性能指标
        """
        # 基础指标
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        # 风险指标
        volatility = returns.std() * np.sqrt(252 * 24 * 60)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # 风险调整收益
        if volatility > 0:
            sharpe_ratio = returns.mean() / volatility * np.sqrt(252 * 24 * 60)
        else:
            sharpe_ratio = 0
            
        # 索提诺比率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252 * 24 * 60)
        else:
            sortino_ratio = 0
            
        # 胜率
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # 盈亏比
        gains = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
        losses = abs(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        # 最大连续亏损
        max_consecutive_losses = self._calculate_max_consecutive_losses(returns)
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses
        }
        
        return self.metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """计算最大回撤"""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """计算最大连续亏损"""
        consecutive_losses = 0
        max_losses = 0
        
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_losses = max(max_losses, consecutive_losses)
            else:
                consecutive_losses = 0
                
        return max_losses
    
    def visualize_performance(self, equity_curve: pd.Series, returns: pd.Series):
        """可视化性能"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 权益曲线
        axes[0, 0].plot(equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Equity')
        
        # 收益率分布
        axes[0, 1].hist(returns, bins=50, alpha=0.7)
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        
        # 累积收益
        cumulative_returns = (1 + returns).cumprod() - 1
        axes[1, 0].plot(cumulative_returns)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Cumulative Return')
        
        # 月度收益热力图
        monthly_returns = returns.resample('M').sum()
        monthly_matrix = monthly_returns.values.reshape(-1, 1)
        sns.heatmap(monthly_matrix, annot=True, cmap='RdYlGn', center=0, 
                   ax=axes[1, 1], cbar_kws={'label': 'Monthly Return'})
        axes[1, 1].set_title('Monthly Returns')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Year')
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance analysis saved to performance_analysis.png")

class FactorSelection:
    """因子选择"""
    
    def __init__(
        self,
        min_ic: float = 0.05,
        max_correlation: float = 0.7,
        min_sharpe: float = 0.5
    ):
        """
        初始化因子选择
        
        Args:
            min_ic: 最小IC值
            max_correlation: 最大相关性
            min_sharpe: 最小夏普比率
        """
        self.min_ic = min_ic
        self.max_correlation = max_correlation
        self.min_sharpe = min_sharpe
        
    def select_factors(
        self,
        ic_values: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """
        选择因子
        
        Args:
            ic_values: IC值字典
            correlation_matrix: 相关性矩阵
            
        Returns:
            选择的因子列表
        """
        selected_factors = []
        
        # 筛选IC值大于阈值的因子
        for factor, ic in ic_values.items():
            if abs(ic) >= self.min_ic:
                selected_factors.append(factor)
                
        # 确保因子间相关性较低
        final_factors = []
        for factor in selected_factors:
            is_correlated = False
            for final_factor in final_factors:
                if factor in correlation_matrix.index and final_factor in correlation_matrix.columns:
                    corr = correlation_matrix.loc[factor, final_factor]
                    if abs(corr) > self.max_correlation:
                        is_correlated = True
                        break
                        
            if not is_correlated:
                final_factors.append(factor)
                
        return final_factors

def run_factor_analysis():
    """运行因子分析"""
    logger.info("Running factor analysis...")
    
    # 生成示例数据
    from main import generate_sample_data
    symbols = ['BTC/USDT', 'ETH/USDT']
    data = generate_sample_data(symbols, days=30)
    
    # 创建因子
    from factor.price import ReturnsFactor, PriceMomentumFactor
    from factor.volatility import HistoricalVolatilityFactor
    from factor.trend import MACDFactor, RSIFactor
    
    factors = [
        ReturnsFactor(window=5),
        PriceMomentumFactor(window=20),
        HistoricalVolatilityFactor(window=20),
        MACDFactor(),
        RSIFactor(),
    ]
    
    # 分析IC值
    analysis = FactorAnalysis()
    ic_values = analysis.analyze_factor_IC(data, factors)
    
    logger.info("IC Values:")
    for factor, ic in ic_values.items():
        logger.info(f"  {factor}: {ic:.4f}")
    
    # 可视化
    analysis.visualize_factor_IC(ic_values)
    
    # 回测单个因子
    backtest = FactorBacktest(factors)
    results = backtest.backtest_all_factors(data)
    
    # 排名因子
    ranked_factors = backtest.rank_factors(results)
    logger.info("Ranked Factors:")
    for i, factor in enumerate(ranked_factors, 1):
        logger.info(f"  {i}. {factor}")
    
    return results

if __name__ == '__main__':
    run_factor_analysis()