"""
策略回测和优化模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression

from factor.combiner import MultiFactorSignal
from factor.price import ReturnsFactor, PriceMomentumFactor
from factor.volatility import HistoricalVolatilityFactor
from factor.trend import MACDFactor, RSIFactor

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(
        self,
        factor_weights: Dict[str, float] = None,
        parameter_ranges: Dict[str, Tuple[float, float]] = None
    ):
        """
        初始化策略优化器
        
        Args:
            factor_weights: 因子权重
            parameter_ranges: 参数范围
        """
        self.factor_weights = factor_weights or {}
        self.parameter_ranges = parameter_ranges or {}
        self.best_parameters = {}
        self.optimization_history = []
        
    def optimize_factor_weights(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        优化因子权重
        
        Args:
            returns: 收益率数据
            signals: 信号数据
            
        Returns:
            最优权重
        """
        def negative_sharpe(weights):
            # 计算组合收益
            portfolio_returns = (signals * weights).sum(axis=1)
            if portfolio_returns.std() == 0:
                return 100  # 返回一个大的值表示差
            
            sharpe = portfolio_returns.mean() / portfolio_returns.std()
            return -sharpe
        
        # 初始权重
        initial_weights = np.array([self.factor_weights.get(col, 1.0) for col in signals.columns])
        initial_weights = initial_weights / initial_weights.sum()
        
        # 约束条件
        bounds = Bounds([0] * len(signals.columns), [1] * len(signals.columns))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # 优化
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = dict(zip(signals.columns, result.x))
            self.best_parameters['factor_weights'] = optimal_weights
            return optimal_weights
        else:
            return self.factor_weights
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        parameter_names: List[str],
        objective_function
    ) -> Dict[str, float]:
        """
        优化策略参数
        
        Args:
            data: 数据
            parameter_names: 参数名称
            objective_function: 目标函数
            
        Returns:
            最优参数
        """
        def objective(params):
            # 将参数设置到策略中
            params_dict = dict(zip(parameter_names, params))
            result = objective_function(params_dict)
            
            # 记录优化历史
            self.optimization_history.append({
                'parameters': params_dict,
                'objective': result
            })
            
            return -result  # 最大化目标函数
        
        # 初始参数
        initial_params = [self.parameter_ranges[name][0] for name in parameter_names]
        
        # 边界
        bounds = [self.parameter_ranges[name] for name in parameter_names]
        
        # 优化
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            optimal_params = dict(zip(parameter_names, result.x))
            self.best_parameters['optimal_parameters'] = optimal_params
            return optimal_params
        else:
            return {name: self.parameter_ranges[name][0] for name in parameter_names}
    
    def get_optimization_history(self) -> List[Dict]:
        """获取优化历史"""
        return self.optimization_history

class WalkForwardOptimizer:
    """滚动窗口优化器"""
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        rolling_window: int = 30,
        min_window: int = 10
    ):
        """
        初始化滚动窗口优化器
        
        Args:
            train_ratio: 训练集比例
            rolling_window: 滚动窗口大小
            min_window: 最小窗口大小
        """
        self.train_ratio = train_ratio
        self.rolling_window = rolling_window
        self.min_window = min_window
        self.optimized_parameters = []
        
    def optimize(
        self,
        data: pd.DataFrame,
        parameter_names: List[str],
        objective_function
    ) -> List[Dict]:
        """
        滚动窗口优化
        
        Args:
            data: 数据
            parameter_names: 参数名称
            objective_function: 目标函数
            
        Returns:
            优化结果列表
        """
        results = []
        
        # 滚动窗口
        for i in range(self.min_window, len(data) - self.rolling_window, self.rolling_window):
            # 训练集
            train_data = data.iloc[i:i+int(len(data) * self.train_ratio)]
            
            # 测试集
            test_data = data.iloc[i+int(len(data) * self.train_ratio):i+self.rolling_window]
            
            if len(test_data) == 0:
                break
            
            # 在训练集上优化
            optimizer = StrategyOptimizer()
            optimal_params = optimizer.optimize_parameters(
                train_data,
                parameter_names,
                objective_function
            )
            
            # 在测试集上验证
            test_result = objective_function(optimal_params)
            
            results.append({
                'window_start': i,
                'window_end': i + self.rolling_window,
                'parameters': optimal_params,
                'train_result': optimizer.optimization_history[-1]['objective'] if optimizer.optimization_history else 0,
                'test_result': test_result
            })
            
            self.optimized_parameters.append(optimal_params)
        
        return results
    
    def get_average_parameters(self) -> Dict[str, float]:
        """获取平均参数"""
        if not self.optimized_parameters:
            return {}
        
        avg_params = {}
        for param_name in self.optimized_parameters[0].keys():
            values = [params[param_name] for params in self.optimized_parameters]
            avg_params[param_name] = np.mean(values)
        
        return avg_params

class ParameterSensitivityAnalyzer:
    """参数敏感性分析器"""
    
    def __init__(self):
        """初始化参数敏感性分析器"""
        self.sensitivity_results = {}
        
    def analyze(
        self,
        data: pd.DataFrame,
        parameter_name: str,
        parameter_range: List[float],
        objective_function
    ) -> Dict[str, List[float]]:
        """
        分析参数敏感性
        
        Args:
            data: 数据
            parameter_name: 参数名称
            parameter_range: 参数范围
            objective_function: 目标函数
            
        Returns:
            敏感性结果
        """
        results = []
        
        for param_value in parameter_range:
            params = {parameter_name: param_value}
            result = objective_function(params)
            results.append((param_value, result))
        
        self.sensitivity_results[parameter_name] = results
        
        return {
            parameter_name: results
        }
    
    def plot_sensitivity(self, parameter_name: str):
        """绘制敏感性曲线（简化版本）"""
        if parameter_name not in self.sensitivity_results:
            return None
        
        results = self.sensitivity_results[parameter_name]
        params = [r[0] for r in results]
        values = [r[1] for r in results]
        
        # 找到最优参数
        optimal_idx = np.argmax(values)
        optimal_param = params[optimal_idx]
        optimal_value = values[optimal_idx]
        
        return {
            'parameter_name': parameter_name,
            'optimal_param': optimal_param,
            'optimal_value': optimal_value,
            'params': params,
            'values': values
        }

class PerformanceAttribution:
    """业绩归因分析"""
    
    def __init__(self):
        """初始化业绩归因分析器"""
        self.attributions = {}
        
    def analyze(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> Dict[str, float]:
        """
        业绩归因
        
        Args:
            returns: 收益率
            factors: 因子数据
            
        Returns:
            归因结果
        """
        # 线性回归
        X = factors.fillna(0)
        y = returns
        
        if len(X) > 10:
            model = LinearRegression()
            model.fit(X, y)
            
            # 计算各因子的贡献
            contributions = {}
            for i, column in enumerate(X.columns):
                contribution = model.coef_[i] * X[column].mean()
                contributions[column] = contribution
            
            # 计算R-squared
            r_squared = model.score(X, y)
            
            self.attributions = {
                'factor_contributions': contributions,
                'r_squared': r_squared,
                'intercept': model.intercept_
            }
            
            return self.attributions
        else:
            return {
                'factor_contributions': {col: 0 for col in factors.columns},
                'r_squared': 0,
                'intercept': 0
            }
    
    def get_factor_importance(self) -> List[Tuple[str, float]]:
        """获取因子重要性"""
        if not self.attributions:
            return []
        
        contributions = self.attributions.get('factor_contributions', {})
        importance = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        return importance

class RiskAdjustmentAnalyzer:
    """风险调整分析器"""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02
    ):
        """
        初始化风险调整分析器
        
        Args:
            risk_free_rate: 无风险利率
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        计算风险调整收益指标
        
        Args:
            returns: 收益率
            benchmark_returns: 基准收益率
            
        Returns:
            风险调整指标
        """
        # 基础指标
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        # 风险指标
        volatility = returns.std() * np.sqrt(252 * 24 * 60)
        downside_risk = returns[returns < 0].std() * np.sqrt(252 * 24 * 60) if len(returns[returns < 0]) > 0 else 0
        
        # 夏普比率
        excess_return = annual_return - self.risk_free_rate
        if volatility > 0:
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0
        
        # 索提诺比率
        if downside_risk > 0:
            sortino_ratio = excess_return / downside_risk
        else:
            sortino_ratio = 0
        
        # 信息比率
        if benchmark_returns is not None:
            active_return = returns - benchmark_returns
            active_volatility = active_return.std() * np.sqrt(252 * 24 * 60)
            if active_volatility > 0:
                information_ratio = active_return.mean() / active_volatility * np.sqrt(252 * 24 * 60)
            else:
                information_ratio = 0
        else:
            information_ratio = 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 其他指标
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).sum() != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'downside_risk': downside_risk,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

class StrategyEvaluator:
    """策略评估器"""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        benchmark_symbol: str = 'BTC/USDT'
    ):
        """
        初始化策略评估器
        
        Args:
            risk_free_rate: 无风险利率
            benchmark_symbol: 基准交易对
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_symbol = benchmark_symbol
        self.evaluation_results = {}
        
    def evaluate(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factors: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        评估策略
        
        Args:
            returns: 收益率
            benchmark_returns: 基准收益率
            factors: 因子数据
            
        Returns:
            评估结果
        """
        # 风险调整收益
        risk_adjusted = RiskAdjustmentAnalyzer(self.risk_free_rate)
        risk_metrics = risk_adjusted.calculate_metrics(returns, benchmark_returns)
        
        # 业绩归因
        attribution = {}
        if factors is not None:
            pa = PerformanceAttribution()
            attribution = pa.analyze(returns, factors)
        
        self.evaluation_results = {
            'risk_metrics': risk_metrics,
            'attribution': attribution
        }
        
        return self.evaluation_results
    
    def generate_report(self) -> str:
        """生成评估报告"""
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report = []
        report.append("=" * 60)
        report.append("STRATEGY EVALUATION REPORT")
        report.append("=" * 60)
        
        # 风险指标
        report.append("\nRISK METRICS:")
        for key, value in self.evaluation_results['risk_metrics'].items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        
        # 业绩归因
        if self.evaluation_results['attribution']:
            report.append("\nFACTOR ATTRIBUTION:")
            factor_contributions = self.evaluation_results['attribution'].get('factor_contributions', {})
            for factor, contribution in factor_contributions.items():
                report.append(f"  {factor}: {contribution:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def run_strategy_optimization():
    """运行策略优化示例"""
    logger.info("Running strategy optimization...")
    
    from main import generate_sample_data
    
    # 生成示例数据
    symbols = ['BTC/USDT', 'ETH/USDT']
    data = generate_sample_data(symbols, days=30)
    
    # 创建策略
    from strategy.strategy import MeanReversionStrategy
    
    def objective_function(params):
        """目标函数"""
        strategy = MeanReversionStrategy(
            zscore_threshold=params.get('zscore_threshold', 2.0),
            window=params.get('window', 50)
        )
        
        result = strategy.run_backtest(data)
        r = result['result']
        
        # 返回夏普比率作为目标函数
        return r.sharpe_ratio
    
    # 优化参数
    parameter_ranges = {
        'zscore_threshold': (1.5, 3.0),
        'window': (30, 100)
    }
    
    optimizer = StrategyOptimizer(parameter_ranges=parameter_ranges)
    optimal_params = optimizer.optimize_parameters(data, ['zscore_threshold', 'window'], objective_function)
    
    logger.info(f"Optimal parameters: {optimal_params}")
    
    # 滚动窗口优化
    wfo = WalkForwardOptimizer(train_ratio=0.7, rolling_window=10, min_window=5)
    wfo_results = wfo.optimize(data, ['zscore_threshold', 'window'], objective_function)
    
    logger.info(f"Walk-forward optimization completed. {len(wfo_results)} windows.")
    
    # 参数敏感性分析
    sensitivity_analyzer = ParameterSensitivityAnalyzer()
    sensitivity_results = sensitivity_analyzer.analyze(
        data,
        'zscore_threshold',
        [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0],
        objective_function
    )
    
    logger.info(f"Sensitivity analysis completed.")
    
    return {
        'optimal_params': optimal_params,
        'wfo_results': wfo_results,
        'sensitivity': sensitivity_results
    }

if __name__ == '__main__':
    run_strategy_optimization()