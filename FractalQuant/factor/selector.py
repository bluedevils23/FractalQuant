"""
因子选择和权重优化模块
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from .base import BaseFactor

class FactorSelector:
    """因子选择器"""
    
    def __init__(self, min_samples: int = 30, min_correlation: float = 0.1):
        """
        初始化因子选择器
        
        Args:
            min_samples: 最小样本数
            min_correlation: 最小相关性阈值
        """
        self.min_samples = min_samples
        self.min_correlation = min_correlation
        self.selected_factors = []
        
    def select_by_correlation(self, factor_df: pd.DataFrame, returns: pd.Series) -> List[str]:
        """基于与收益的相关性选择因子"""
        selected = []
        
        for col in factor_df.columns:
            factor_values = factor_df[col].dropna()
            common_index = factor_values.index.intersection(returns.index)
            
            if len(common_index) < self.min_samples:
                continue
            
            factor_vals = factor_values.loc[common_index]
            returns_vals = returns.loc[common_index]
            
            try:
                correlation, _ = stats.spearmanr(factor_vals, returns_vals)
                
                if abs(correlation) >= self.min_correlation:
                    selected.append(col)
            except:
                continue
        
        return selected
    
    def select_by_information_ratio(self, factor_df: pd.DataFrame, returns: pd.Series) -> List[str]:
        """基于信息比率选择因子"""
        selected = []
        
        for col in factor_df.columns:
            factor_values = factor_df[col].dropna()
            common_index = factor_values.index.intersection(returns.index)
            
            if len(common_index) < self.min_samples:
                continue
            
            factor_vals = factor_values.loc[common_index]
            returns_vals = returns.loc[common_index]
            
            try:
                factor_returns = factor_vals * returns_vals.shift(1).fillna(0)
                mean_return = factor_returns.mean()
                std_return = factor_returns.std()
                
                if std_return > 0:
                    info_ratio = mean_return / std_return
                    if info_ratio > 0:
                        selected.append(col)
            except:
                continue
        
        return selected
    
    def select_by_stability(self, factor_df: pd.DataFrame, window: int = 20) -> List[str]:
        """基于稳定性选择因子"""
        selected = []
        
        for col in factor_df.columns:
            factor_values = factor_df[col].dropna()
            
            if len(factor_values) < window * 2:
                continue
            
            rolling_corr = []
            for i in range(window, len(factor_values)):
                window_data = factor_values.iloc[i-window:i]
                x = np.arange(len(window_data))
                try:
                    slope, _, r_value, _, _ = stats.linregress(x, window_data)
                    rolling_corr.append(abs(r_value))
                except:
                    continue
            
            if rolling_corr and np.mean(rolling_corr) > 0.3:
                selected.append(col)
        
        return selected
    
    def select_by_diversification(self, factor_df: pd.DataFrame, target_count: int = 10) -> List[str]:
        """基于多样性选择因子（降低相关性）"""
        if len(factor_df.columns) <= target_count:
            return list(factor_df.columns)
        
        factor_df = factor_df.dropna()
        
        if len(factor_df) < 20:
            return list(factor_df.columns)[:target_count]
        
        try:
            corr_matrix = factor_df.corr()
            
            selected = []
            remaining = list(corr_matrix.columns)
            
            while len(selected) < target_count and remaining:
                if not selected:
                    # 选择方差最大的因子
                    variances = factor_df.std()
                    best_factor = variances.idxmax()
                    selected.append(best_factor)
                    remaining.remove(best_factor)
                else:
                    # 选择与已选因子相关性最低的因子
                    min_avg_corr = float('inf')
                    best_factor = None
                    
                    for factor in remaining:
                        avg_corr = 0
                        for selected_factor in selected:
                            avg_corr += abs(corr_matrix.loc[factor, selected_factor])
                        avg_corr /= len(selected)
                        
                        if avg_corr < min_avg_corr:
                            min_avg_corr = avg_corr
                            best_factor = factor
                    
                    if best_factor and min_avg_corr < 0.7:
                        selected.append(best_factor)
                        remaining.remove(best_factor)
                    else:
                        break
            
            return selected
        except:
            return list(factor_df.columns)[:target_count]
    
    def select_factors(self, factor_df: pd.DataFrame, returns: pd.Series, 
                      method: str = 'all', target_count: int = 10) -> List[str]:
        """
        综合因子选择
        
        Args:
            factor_df: 因子数据框
            returns: 收益序列
            method: 选择方法 ('correlation', 'information', 'stability', 'diversification', 'all')
            target_count: 目标因子数量
            
        Returns:
            选中的因子列表
        """
        if method == 'correlation':
            return self.select_by_correlation(factor_df, returns)
        elif method == 'information':
            return self.select_by_information_ratio(factor_df, returns)
        elif method == 'stability':
            return self.select_by_stability(factor_df)
        elif method == 'diversification':
            return self.select_by_diversification(factor_df, target_count)
        elif method == 'all':
            # 综合多种方法
            corr_selected = self.select_by_correlation(factor_df, returns)
            info_selected = self.select_by_information_ratio(factor_df, returns)
            stability_selected = self.select_by_stability(factor_df)
            
            # 取交集
            all_selected = set(corr_selected) & set(info_selected) & set(stability_selected)
            
            if len(all_selected) < target_count:
                # 补充多样性选择
                diversification_selected = self.select_by_diversification(factor_df, target_count)
                all_selected = all_selected | set(diversification_selected)
            
            return list(all_selected)[:target_count]
        else:
            return list(factor_df.columns)[:target_count]


class WeightOptimizer:
    """权重优化器"""
    
    def __init__(self, method: str = 'risk_parity', min_weight: float = 0.01, max_weight: float = 0.3):
        """
        初始化权重优化器
        
        Args:
            method: 优化方法 ('equal', 'risk_parity', 'mean_var', 'bayesian', 'adaptive')
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights_history = []
        
    def optimize_equal(self, factor_count: int) -> Dict[str, float]:
        """等权优化"""
        return {f'factor_{i}': 1.0 / factor_count for i in range(factor_count)}
    
    def optimize_risk_parity(self, factor_df: pd.DataFrame) -> Dict[str, float]:
        """风险平价优化"""
        factor_df = factor_df.dropna()
        
        if len(factor_df) < 20 or len(factor_df.columns) < 2:
            return self.optimize_equal(len(factor_df.columns))
        
        try:
            returns = factor_df.pct_change().dropna()
            cov_matrix = returns.cov()
            
            if cov_matrix is None or len(cov_matrix) < 2:
                return self.optimize_equal(len(factor_df.columns))
            
            # 计算风险贡献
            vol = np.sqrt(np.diag(cov_matrix))
            risk_budget = 1.0 / len(vol)
            
            # 迭代优化
            weights = np.ones(len(vol)) / len(vol)
            
            for _ in range(100):
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
                marginal_risk = cov_matrix @ weights / portfolio_vol
                risk_contrib = weights * marginal_risk
                
                # 调整权重
                target_risk = risk_budget * portfolio_vol
                adjustment = target_risk / risk_contrib
                weights = weights * adjustment
                
                # 归一化
                weights = weights / weights.sum()
                
                # 约束
                weights = np.clip(weights, self.min_weight, self.max_weight)
                weights = weights / weights.sum()
            
            return {col: weights[i] for i, col in enumerate(cov_matrix.columns)}
        except:
            return self.optimize_equal(len(factor_df.columns))
    
    def optimize_mean_var(self, factor_df: pd.DataFrame, risk_aversion: float = 1.0) -> Dict[str, float]:
        """均值-方差优化"""
        factor_df = factor_df.dropna()
        
        if len(factor_df) < 20 or len(factor_df.columns) < 2:
            return self.optimize_equal(len(factor_df.columns))
        
        try:
            returns = factor_df.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            if cov_matrix is None or len(cov_matrix) < 2:
                return self.optimize_equal(len(factor_df.columns))
            
            # 简化的均值-方差优化
            weights = np.ones(len(mean_returns))
            
            for _ in range(100):
                portfolio_return = weights @ mean_returns
                portfolio_variance = weights @ cov_matrix @ weights
                
                # 梯度下降
                gradient = mean_returns - risk_aversion * cov_matrix @ weights
                weights = weights + 0.01 * gradient
                
                # 约束
                weights = np.clip(weights, self.min_weight, self.max_weight)
                weights = weights / weights.sum()
            
            return {col: weights[i] for i, col in enumerate(mean_returns.index)}
        except:
            return self.optimize_equal(len(factor_df.columns))
    
    def optimize_bayesian(self, factor_df: pd.DataFrame) -> Dict[str, float]:
        """贝叶斯优化（考虑估计误差）"""
        factor_df = factor_df.dropna()
        
        if len(factor_df) < 30 or len(factor_df.columns) < 2:
            return self.optimize_equal(len(factor_df.columns))
        
        try:
            returns = factor_df.pct_change().dropna()
            
            # 使用Ledoit-Wolf收缩估计
            lw = LedoitWolf()
            lw.fit(returns)
            shrunk_cov = lw.covariance_
            shrunk_mean = returns.mean() * 0.9
            
            # 优化
            weights = np.ones(len(shrunk_mean))
            
            for _ in range(100):
                portfolio_return = weights @ shrunk_mean
                portfolio_variance = weights @ shrunk_cov @ weights
                
                gradient = shrunk_mean - 1.0 * shrunk_cov @ weights
                weights = weights + 0.01 * gradient
                
                weights = np.clip(weights, self.min_weight, self.max_weight)
                weights = weights / weights.sum()
            
            return {col: weights[i] for i, col in enumerate(returns.columns)}
        except:
            return self.optimize_equal(len(factor_df.columns))
    
    def optimize_adaptive(self, factor_df: pd.DataFrame, returns: pd.Series, 
                         window: int = 20) -> Dict[str, float]:
        """自适应优化（基于因子表现）"""
        factor_df = factor_df.dropna()
        returns = returns.loc[factor_df.index]
        
        if len(returns) < window * 2:
            return self.optimize_equal(len(factor_df.columns))
        
        try:
            factor_returns = {}
            for col in factor_df.columns:
                factor_returns[col] = returns * factor_df[col].shift(1).fillna(0)
            
            recent_performance = {}
            for col in factor_returns:
                recent_factor_returns = factor_returns[col].iloc[-window:]
                if len(recent_factor_returns.dropna()) > window // 2:
                    recent_performance[col] = recent_factor_returns.dropna().mean()
                else:
                    recent_performance[col] = 0
            
            weights = {k: max(0, v) for k, v in recent_performance.items()}
            total = sum(weights.values())
            
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = self.optimize_equal(len(factor_df.columns))
            
            # 约束
            weights = {k: max(self.min_weight, min(self.max_weight, v)) for k, v in weights.items()}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            
            return weights
        except:
            return self.optimize_equal(len(factor_df.columns))
    
    def optimize(self, factor_df: pd.DataFrame, returns: pd.Series = None) -> Dict[str, float]:
        """
        优化因子权重
        
        Args:
            factor_df: 因子数据框
            returns: 收益序列（用于自适应优化）
            
        Returns:
            权重字典
        """
        if self.method == 'equal':
            weights = self.optimize_equal(len(factor_df.columns))
        elif self.method == 'risk_parity':
            weights = self.optimize_risk_parity(factor_df)
        elif self.method == 'mean_var':
            weights = self.optimize_mean_var(factor_df)
        elif self.method == 'bayesian':
            weights = self.optimize_bayesian(factor_df)
        elif self.method == 'adaptive':
            weights = self.optimize_adaptive(factor_df, returns) if returns is not None else self.optimize_equal(len(factor_df.columns))
        else:
            weights = self.optimize_equal(len(factor_df.columns))
        
        self.weights_history.append(weights)
        return weights


class FactorEnsemble:
    """因子集成"""
    
    def __init__(self, methods: List[str] = None):
        """
        初始化因子集成
        
        Args:
            methods: 优化方法列表
        """
        self.methods = methods or ['equal', 'risk_parity', 'adaptive']
        self.ensembled_weights = {}
        
    def ensemble_equal(self, all_weights: List[Dict[str, float]]) -> Dict[str, float]:
        """等权集成"""
        if not all_weights:
            return {}
        
        factors = set()
        for weights in all_weights:
            factors.update(weights.keys())
        
        ensemble = {f: 0 for f in factors}
        
        for weights in all_weights:
            for f in factors:
                ensemble[f] += weights.get(f, 0)
        
        for f in ensemble:
            ensemble[f] /= len(all_weights)
        
        return ensemble
    
    def ensemble_rank(self, all_weights: List[Dict[str, float]]) -> Dict[str, float]:
        """排名集成"""
        if not all_weights:
            return {}
        
        factors = set()
        for weights in all_weights:
            factors.update(weights.keys())
        
        ensemble = {f: 0 for f in factors}
        
        for weights in all_weights:
            sorted_factors = sorted(weights.keys(), key=lambda x: weights[x], reverse=True)
            for i, f in enumerate(sorted_factors):
                ensemble[f] += len(sorted_factors) - i
        
        for f in ensemble:
            ensemble[f] /= sum(ensemble.values())
        
        return ensemble
    
    def ensemble_vote(self, all_weights: List[Dict[str, float]], threshold: float = 0.1) -> Dict[str, float]:
        """投票集成（只保留权重超过阈值的因子）"""
        if not all_weights:
            return {}
        
        factors = set()
        for weights in all_weights:
            factors.update(weights.keys())
        
        ensemble = {f: 0 for f in factors}
        
        for weights in all_weights:
            for f in factors:
                if weights.get(f, 0) > threshold:
                    ensemble[f] += 1
        
        for f in ensemble:
            ensemble[f] /= len(all_weights)
        
        return ensemble
    
    def ensemble(self, factor_df: pd.DataFrame, returns: pd.Series = None) -> Dict[str, float]:
        """
        集成多个优化方法的结果
        
        Args:
            factor_df: 因子数据框
            returns: 收益序列
            
        Returns:
            集成权重
        """
        all_weights = []
        
        for method in self.methods:
            optimizer = WeightOptimizer(method=method)
            weights = optimizer.optimize(factor_df, returns)
            all_weights.append(weights)
        
        self.ensembled_weights = self.ensemble_equal(all_weights)
        return self.ensembled_weights


class FactorBacktest:
    """因子回测"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        
    def backtest_single_factor(self, factor: pd.Series, returns: pd.Series) -> Dict:
        """
        回测单个因子
        
        Args:
            factor: 因子值
            returns: 收益序列
            
        Returns:
            回测结果
        """
        factor = factor.dropna()
        common_index = factor.index.intersection(returns.index)
        
        if len(common_index) < 20:
            return {'error': 'Insufficient data'}
        
        factor_vals = factor.loc[common_index]
        returns_vals = returns.loc[common_index]
        
        # 生成信号
        signals = np.sign(factor_vals)
        
        # 计算策略收益
        strategy_returns = signals.shift(1) * returns_vals
        
        # 计算指标
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (365 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252 * 24 * 60)
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24 * 60) if strategy_returns.std() > 0 else 0
        
        negative_returns = strategy_returns[strategy_returns < 0]
        sortino_ratio = strategy_returns.mean() / negative_returns.std() * np.sqrt(252 * 24 * 60) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
        
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
        
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'factor_correlation': factor_vals.corr(returns_vals),
        }
    
    def backtest_factor_pairs(self, factor_df: pd.DataFrame, returns: pd.Series, 
                             top_n: int = 5) -> pd.DataFrame:
        """
        回测因子组合
        
        Args:
            factor_df: 因子数据框
            returns: 收益序列
            top_n: 选择前N个因子
            
        Returns:
            因子组合回测结果
        """
        factor_df = factor_df.dropna()
        common_index = factor_df.index.intersection(returns.index)
        
        if len(common_index) < 30:
            return pd.DataFrame()
        
        factor_df = factor_df.loc[common_index]
        returns_vals = returns.loc[common_index]
        
        results = []
        
        # 单因子回测
        for col in factor_df.columns:
            result = self.backtest_single_factor(factor_df[col], returns_vals)
            result['factors'] = col
            results.append(result)
        
        # 双因子组合回测
        if len(factor_df.columns) >= 2:
            for i in range(min(top_n, len(factor_df.columns))):
                for j in range(i + 1, min(top_n, len(factor_df.columns))):
                    factor1 = factor_df.iloc[:, i]
                    factor2 = factor_df.iloc[:, j]
                    
                    combined = (factor1 + factor2) / 2
                    result = self.backtest_single_factor(combined, returns_vals)
                    result['factors'] = f'{factor_df.columns[i]}+{factor_df.columns[j]}'
                    results.append(result)
        
        return pd.DataFrame(results)
