"""
跨市场因子（相关性、套利机会、市场联动等）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.signal import coherence
from .base import BaseFactor

class CrossMarketCorrelationFactor(BaseFactor):
    """跨市场相关性因子"""
    
    def __init__(self, window: int = 50, correlation_window: int = 20):
        super().__init__('cross_market_correlation', window)
        self.correlation_window = correlation_window
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算与参考市场的相关性"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_correlation(x):
            if len(x) < self.correlation_window:
                return 0
            
            current_window = x[-self.correlation_window:]
            ref_window = ref_close[-self.correlation_window:]
            
            if len(current_window) != len(ref_window):
                min_len = min(len(current_window), len(ref_window))
                current_window = current_window[-min_len:]
                ref_window = ref_window[-min_len:]
            
            if len(current_window) < 10:
                return 0
            
            try:
                correlation, _ = pearsonr(current_window, ref_window)
                return correlation
            except:
                return 0
        
        correlation = close.rolling(window=self.window).apply(calc_correlation)
        return correlation

class ArbitrageOpportunityFactor(BaseFactor):
    """套利机会因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.01):
        super().__init__('arbitrage_opportunity', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算套利机会（价差标准化）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_arbitrage(x):
            if len(x) < 20:
                return 0
            
            current_window = x[-20:]
            ref_window = ref_close[-20:]
            
            if len(current_window) != len(ref_window):
                min_len = min(len(current_window), len(ref_window))
                current_window = current_window[-min_len:]
                ref_window = ref_window[-min_len:]
            
            price_spread = current_window - ref_window
            
            mean_spread = np.mean(price_spread)
            std_spread = np.std(price_spread)
            
            current_spread = price_spread[-1]
            
            if std_spread > 0:
                z_score = (current_spread - mean_spread) / std_spread
                return z_score
            return 0
        
        arbitrage = close.rolling(window=self.window).apply(calc_arbitrage)
        return arbitrage

class MarketLinkageFactor(BaseFactor):
    """市场联动因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('market_linkage', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算市场联动性（协整关系）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_linkage(x):
            if len(x) < 50:
                return 0
            
            current_series = x[-50:]
            ref_series = ref_close[-50:]
            
            if len(current_series) != len(ref_series):
                min_len = min(len(current_series), len(ref_series))
                current_series = current_series[-min_len:]
                ref_series = ref_series[-min_len:]
            
            try:
                returns_current = np.diff(np.log(current_series))
                returns_ref = np.diff(np.log(ref_series))
                
                correlation, _ = pearsonr(returns_current, returns_ref)
                
                return abs(correlation)
            except:
                return 0
        
        linkage = close.rolling(window=self.window).apply(calc_linkage)
        return linkage

class RelativeStrengthFactor(BaseFactor):
    """相对强度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('relative_strength', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算相对强度（相对于参考市场的表现）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_strength(x):
            if len(x) < 20:
                return 0
            
            current_window = x[-20:]
            ref_window = ref_close[-20:]
            
            current_return = (current_window[-1] - current_window[0]) / (current_window[0] + 1e-8)
            ref_return = (ref_window[-1] - ref_window[0]) / (ref_window[0] + 1e-8)
            
            relative_strength = current_return - ref_return
            
            return relative_strength * 100
        
        strength = close.rolling(window=self.window).apply(calc_strength)
        return strength

class CointegrationFactor(BaseFactor):
    """协整因子"""
    
    def __init__(self, window: int = 50, min_window: int = 30):
        super().__init__('cointegration', window)
        self.min_window = min_window
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算协整关系（Engle-Granger两步法）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_cointegration(x):
            if len(x) < self.min_window:
                return 0
            
            current_series = x[-self.min_window:]
            ref_series = ref_close[-self.min_window:]
            
            if len(current_series) != len(ref_series):
                min_len = min(len(current_series), len(ref_series))
                current_series = current_series[-min_len:]
                ref_series = ref_series[-min_len:]
            
            try:
                y = np.log(current_series)
                x_log = np.log(ref_series)
                
                X = np.column_stack([np.ones(len(x_log)), x_log])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                residuals = y - X @ beta
                
                adf_stat = self._adf_test(residuals)
                
                return -adf_stat
            except:
                return 0
        
        def _adf_test(self, series):
            if len(series) < 20:
                return 0
            
            diff_series = np.diff(series)
            lagged_series = series[:-1]
            
            X = np.column_stack([np.ones(len(lagged_series)), lagged_series])
            y = diff_series[1:]
            
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                
                if ss_tot > 0:
                    r_squared = 1 - ss_res / ss_tot
                else:
                    r_squared = 0
                
                n = len(y)
                k = 2
                df = n - k - 1
                
                if df > 0 and ss_res > 0:
                    t_stat = np.sqrt(df * r_squared / (1 - r_squared))
                else:
                    t_stat = 0
                
                return t_stat
            except:
                return 0
        
        cointegration = close.rolling(window=self.window).apply(calc_cointegration)
        return cointegration

class CrossMarketVolatilityFactor(BaseFactor):
    """跨市场波动率因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_volatility', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算相对于参考市场的波动率比率"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_vol_ratio(x):
            if len(x) < 30:
                return 0
            
            current_returns = np.diff(np.log(x[-30:]))
            ref_returns = np.diff(np.log(ref_close[-30:]))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            current_vol = np.std(current_returns)
            ref_vol = np.std(ref_returns)
            
            if ref_vol > 0:
                vol_ratio = current_vol / ref_vol
                return vol_ratio
            return 1.0
        
        vol_ratio = close.rolling(window=self.window).apply(calc_vol_ratio)
        return vol_ratio

class MarketRegimeSwitchFactor(BaseFactor):
    """市场 regime 切换因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.8):
        super().__init__('market_regime_switch', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """检测市场 regime 切换"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_regime_switch(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            current_vol = np.std(current_returns[-20:])
            ref_vol = np.std(ref_returns[-20:])
            
            current_corr, _ = pearsonr(current_returns[-20:], ref_returns[-20:])
            
            regime_strength = abs(current_corr) * (current_vol / (ref_vol + 1e-8))
            
            return regime_strength
        
        regime_switch = close.rolling(window=self.window).apply(calc_regime_switch)
        return regime_switch

class CrossMarketEntropyFactor(BaseFactor):
    """跨市场熵因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_entropy', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算跨市场联合熵"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_entropy(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            joint_returns = np.column_stack([current_returns, ref_returns])
            
            hist, _ = np.histogramdd(joint_returns, bins=10)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            
            if len(prob) > 0:
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                return entropy / 10
            return 0
        
        entropy = close.rolling(window=self.window).apply(calc_entropy)
        return entropy

class CrossMarketCoherenceFactor(BaseFactor):
    """跨市场相干性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_coherence', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算跨市场相干性（频域相关性）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_coherence(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                f, coh = coherence(current_returns, ref_returns, nperseg=10)
                
                if len(coh) > 0:
                    mean_coh = np.mean(coh)
                    return mean_coh
            except:
                pass
            
            return 0
        
        coherence = close.rolling(window=self.window).apply(calc_coherence)
        return coherence

class CrossMarketGrangerFactor(BaseFactor):
    """跨市场格兰杰因果因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('cross_market_granger', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算格兰杰因果性"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_granger(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                n = len(current_returns)
                
                y = current_returns[self.lag:]
                
                X_restricted = np.column_stack([np.ones(n - self.lag), 
                                               current_returns[self.lag-1:-1]])
                X_unrestricted = np.column_stack([X_restricted, ref_returns[self.lag-1:-1]])
                
                beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                beta_unrestricted = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
                
                residuals_restricted = y - X_restricted @ beta_restricted
                residuals_unrestricted = y - X_unrestricted @ beta_unrestricted
                
                ss_res_restricted = np.sum(residuals_restricted ** 2)
                ss_res_unrestricted = np.sum(residuals_unrestricted ** 2)
                
                if ss_res_unrestricted > 0:
                    f_stat = ((ss_res_restricted - ss_res_unrestricted) / self.lag) / \
                            (ss_res_unrestricted / (len(y) - 2 - self.lag))
                    return max(0, f_stat)
            except:
                pass
            
            return 0
        
        granger = close.rolling(window=self.window).apply(calc_granger)
        return granger

class CrossMarketJointDistributionFactor(BaseFactor):
    """跨市场联合分布因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_joint_dist', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算跨市场联合分布特征"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_joint_dist(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                current_skew = stats.skew(current_returns)
                ref_skew = stats.skew(ref_returns)
                
                current_kurt = stats.kurtosis(current_returns)
                ref_kurt = stats.kurtosis(ref_returns)
                
                joint_score = abs(current_skew - ref_skew) + abs(current_kurt - ref_kurt) / 3
                return joint_score
            except:
                return 0
        
        joint_dist = close.rolling(window=self.window).apply(calc_joint_dist)
        return joint_dist

class CrossMarketCopulaFactor(BaseFactor):
    """跨市场 copula 因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_copula', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算 copula 相关性（捕捉非线性依赖）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_copula(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                current_cdf = stats.rankdata(current_returns) / (len(current_returns) + 1)
                ref_cdf = stats.rankdata(ref_returns) / (len(ref_returns) + 1)
                
                correlation, _ = pearsonr(current_cdf, ref_cdf)
                
                return abs(correlation)
            except:
                return 0
        
        copula = close.rolling(window=self.window).apply(calc_copula)
        return copula

class CrossMarketPhaseSynchronizationFactor(BaseFactor):
    """跨市场相位同步因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_phase_sync', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算跨市场相位同步性"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_phase_sync(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                current_phase = np.unwrap(np.angle(np.fft.fft(current_returns)))
                ref_phase = np.unwrap(np.angle(np.fft.fft(ref_returns)))
                
                phase_diff = np.abs(current_phase - ref_phase)
                
                synchronization = 1 - np.mean(phase_diff) / np.pi
                return max(0, synchronization)
            except:
                return 0
        
        phase_sync = close.rolling(window=self.window).apply(calc_phase_sync)
        return phase_sync

class CrossMarketInformationFlowFactor(BaseFactor):
    """跨市场信息流因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('cross_market_info_flow', window)
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算跨市场信息流（互信息估计）"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_info_flow(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                current_discrete = np.digitize(current_returns, np.linspace(-0.05, 0.05, 10))
                ref_discrete = np.digitize(ref_returns, np.linspace(-0.05, 0.05, 10))
                
                joint_hist, _, _ = np.histogram2d(current_discrete, ref_discrete, bins=10)
                joint_prob = joint_hist / joint_hist.sum() + 1e-10
                
                current_marginal = joint_prob.sum(axis=1)
                ref_marginal = joint_prob.sum(axis=0)
                
                mutual_info = np.sum(joint_prob * np.log(joint_prob / (current_marginal[:, np.newaxis] * ref_marginal[np.newaxis, :] + 1e-10) + 1e-10))
                
                return max(0, mutual_info)
            except:
                return 0
        
        info_flow = close.rolling(window=self.window).apply(calc_info_flow)
        return info_flow

class CrossMarketMultiscaleCorrelationFactor(BaseFactor):
    """跨市场多尺度相关性因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('cross_market_multiscale_corr', window)
        self.scales = scales or [5, 10, 20, 40]
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算多尺度相关性"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_multiscale(x):
            if len(x) < 50:
                return 0
            
            current_window = x[-50:]
            ref_window = ref_close[-50:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            correlations = []
            
            for scale in self.scales:
                if scale >= len(current_returns) // 2:
                    continue
                
                n_segments = len(current_returns) // scale
                segment_correlations = []
                
                for i in range(n_segments):
                    current_seg = current_returns[i*scale:(i+1)*scale]
                    ref_seg = ref_returns[i*scale:(i+1)*scale]
                    
                    if len(current_seg) < 5:
                        continue
                    
                    try:
                        corr, _ = pearsonr(current_seg, ref_seg)
                        segment_correlations.append(abs(corr))
                    except:
                        continue
                
                if segment_correlations:
                    correlations.append(np.mean(segment_correlations))
            
            if correlations:
                return np.mean(correlations)
            return 0
        
        multiscale = close.rolling(window=self.window).apply(calc_multiscale)
        return multiscale

class CrossMarketDynamicCorrelationFactor(BaseFactor):
    """跨市场动态相关性因子"""
    
    def __init__(self, window: int = 50, decay: float = 0.95):
        super().__init__('cross_market_dynamic_corr', window)
        self.decay = decay
        
    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """计算动态加权相关性"""
        close = df['close']
        
        if reference_df is None:
            reference_df = df.copy()
        
        ref_close = reference_df['close']
        
        def calc_dynamic_corr(x):
            if len(x) < 30:
                return 0
            
            current_window = x[-30:]
            ref_window = ref_close[-30:]
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            weighted_sum = 0
            weight_sum = 0
            
            weights = np.array([self.decay ** (len(current_returns) - i - 1) for i in range(len(current_returns))])
            weights = weights / weights.sum()
            
            for i in range(len(current_returns)):
                weighted_sum += weights[i] * current_returns[i] * ref_returns[i]
                weight_sum += weights[i]
            
            if weight_sum > 0:
                dynamic_corr = weighted_sum / weight_sum
                return dynamic_corr
            return 0
        
        dynamic_corr = close.rolling(window=self.window).apply(calc_dynamic_corr)
        return dynamic_corr
