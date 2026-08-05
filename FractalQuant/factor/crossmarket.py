"""
跨市场因子（相关性、套利机会、市场联动等）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.signal import coherence as scipy_coherence, hilbert, butter, sosfiltfilt
from .base import BaseFactor


def _aligned_price_windows(
    current_window: pd.Series,
    reference_close: pd.Series,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = pd.to_numeric(
        current_window.iloc[-length:], errors="coerce"
    )
    reference = pd.to_numeric(
        reference_close.reindex(current.index), errors="coerce"
    )
    current_values = current.to_numpy(dtype=float, copy=False)
    reference_values = reference.to_numpy(dtype=float, copy=False)
    valid = (
        np.isfinite(current_values)
        & np.isfinite(reference_values)
        & (current_values > 0)
        & (reference_values > 0)
    )
    return current_values[valid], reference_values[valid]


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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, self.correlation_window
            )
            
            if len(current_window) < 10:
                return np.nan
            
            try:
                correlation, _ = pearsonr(current_window, ref_window)
                return correlation if np.isfinite(correlation) else np.nan
            except:
                return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 20
            )
            if len(current_window) < 20:
                return np.nan
            
            price_spread = current_window - ref_window
            
            mean_spread = np.mean(price_spread)
            std_spread = np.std(price_spread)
            
            current_spread = price_spread[-1]
            
            if std_spread > 0:
                z_score = (current_spread - mean_spread) / std_spread
                return z_score
            return np.nan
        
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
                return np.nan
            
            current_series, ref_series = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_series) < 50:
                return np.nan
            
            try:
                returns_current = np.diff(np.log(current_series))
                returns_ref = np.diff(np.log(ref_series))
                
                correlation, _ = pearsonr(returns_current, returns_ref)
                return abs(correlation) if np.isfinite(correlation) else np.nan
            except:
                return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 20
            )
            if len(current_window) < 20:
                return np.nan
            
            current_return = (current_window[-1] - current_window[0]) / (current_window[0] + 1e-8)
            ref_return = (ref_window[-1] - ref_window[0]) / (ref_window[0] + 1e-8)
            
            relative_strength = current_return - ref_return
            
            return relative_strength * 100
        
        strength = close.rolling(window=self.window).apply(calc_strength)
        return strength

class CointegrationFactor(BaseFactor):
    """协整因子（Engle-Granger 两步法 + 真实 ADF t 统计量）

    输出为 -t_ADF，值越大表示协整越强（残差均值回归越显著）。
    ADF 回归：Δe_t = α + γ·e_{t-1} + δ·Δe_{t-1} + ε_t
    返回 γ / SE(γ) 的负值，自然落在有界范围，无爆炸风险。
    """

    def __init__(self, window: int = 50, min_window: int = 30, adf_maxlag: int = 1):
        super().__init__('cointegration', window)
        self.min_window = min_window
        self.adf_maxlag = adf_maxlag

    @staticmethod
    def _adf_tstat(residuals: np.ndarray, maxlag: int = 1) -> float:
        """对序列执行 ADF 检验，返回 γ 的 t 值。

        ADF 回归（含截距，固定滞后阶数 maxlag）：
            Δe_t = α + γ·e_{t-1} + Σ_{i=1}^{maxlag} δ_i·Δe_{t-i} + ε_t

        平稳（均值回归）序列的 γ < 0，t 值 < 0。
        """
        e = np.asarray(residuals, dtype=float)
        de = np.diff(e)
        n_de = len(de)
        n = n_de - maxlag          # 有效观测数
        if n < max(10, maxlag + 3):
            return np.nan

        # 因变量：Δe_{maxlag}, ..., Δe_{n_de-1}
        y = de[maxlag:]

        # 设计矩阵列：截距, e_{t-1}, Δe_{t-1}, ..., Δe_{t-maxlag}
        cols = [np.ones(n), e[maxlag:-1]]          # 截距 + e_{t-1}
        for lag in range(1, maxlag + 1):
            cols.append(de[maxlag - lag: n_de - lag])
        X = np.column_stack(cols)                  # shape (n, 2+maxlag)

        try:
            beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            if rank < X.shape[1]:
                return np.nan

            resid = y - X @ beta
            k = X.shape[1]
            sigma2 = np.dot(resid, resid) / (n - k)
            if sigma2 <= 0:
                return np.nan

            XtX_inv = np.linalg.inv(X.T @ X)
            se_gamma = np.sqrt(sigma2 * XtX_inv[1, 1])  # SE of γ (index 1)
            if se_gamma <= 0:
                return np.nan

            return float(beta[1] / se_gamma)
        except Exception:
            return np.nan

    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        close = df['close']

        if reference_df is None:
            reference_df = df.copy()

        ref_close = reference_df['close']
        min_window = self.min_window
        adf_maxlag = self.adf_maxlag

        def calc_cointegration(x):
            if len(x) < min_window:
                return np.nan

            current_series, ref_series = _aligned_price_windows(
                x, ref_close, min_window
            )
            if len(current_series) < min_window:
                return np.nan

            try:
                y_log = np.log(current_series)
                x_log = np.log(ref_series)

                # 第一步：OLS 回归求协整残差
                X_ols = np.column_stack([np.ones(len(x_log)), x_log])
                beta_ols = np.linalg.lstsq(X_ols, y_log, rcond=None)[0]
                residuals = y_log - X_ols @ beta_ols

                # 第二步：对残差执行 ADF 检验
                t_adf = CointegrationFactor._adf_tstat(residuals, maxlag=adf_maxlag)
                if not np.isfinite(t_adf):
                    return np.nan

                # 取负：值越大表示协整（均值回归）越强
                return -t_adf
            except Exception:
                return np.nan

        cointegration = close.rolling(window=self.window).apply(
            calc_cointegration, raw=False
        )
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 30
            )
            if len(current_window) < 30:
                return np.nan
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            current_vol = np.std(current_returns[-20:])
            ref_vol = np.std(ref_returns[-20:])
            
            current_corr, _ = pearsonr(
                current_returns[-20:], ref_returns[-20:]
            )
            if not np.isfinite(current_corr):
                return np.nan
            
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
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
            return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                _, coh = scipy_coherence(
                    current_returns, ref_returns, nperseg=10
                )
                
                if len(coh) > 0:
                    mean_coh = np.mean(coh)
                    return mean_coh
            except:
                pass
            
            return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
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
            
            return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
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
                return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
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
                return abs(correlation) if np.isfinite(correlation) else np.nan
            except:
                return np.nan
        
        copula = close.rolling(window=self.window).apply(calc_copula)
        return copula

class CrossMarketPhaseSynchronizationFactor(BaseFactor):
    """跨市场相位同步因子（带通 + Hilbert 瞬时相位 PLV）"""

    def __init__(
        self,
        window: int = 50,
        low_period: float = 3.0,
        high_period: float = 15.0,
        filter_order: int = 2,
        edge_trim_ratio: float = 0.1,
    ):
        super().__init__('cross_market_phase_sync', window)
        self.low_period = low_period
        self.high_period = high_period
        self.filter_order = filter_order
        self.edge_trim_ratio = edge_trim_ratio
        # 采样 1 bar/min，Nyquist=0.5 cycles/min。周期 p 分钟 -> 归一化频率 (1/p)/0.5 = 2/p。
        low_norm = 2.0 / high_period   # 高周期 -> 低频边界
        high_norm = 2.0 / low_period   # 低周期 -> 高频边界
        self._sos = butter(
            filter_order, [low_norm, high_norm], btype='band', output='sos'
        )

    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """基于 Hilbert 瞬时相位的相位锁定值 (PLV)，天然落在 [0, 1]。"""
        close = df['close']

        if reference_df is None:
            reference_df = df.copy()

        ref_close = reference_df['close']
        sos = self._sos
        min_length = self.window
        trim = max(1, int(round(self.window * self.edge_trim_ratio)))

        def calc_phase_sync(x):
            if len(x) < min_length:
                return np.nan

            current_window, ref_window = _aligned_price_windows(
                x, ref_close, min_length
            )
            if len(current_window) < min_length:
                return np.nan

            try:
                current_returns = np.diff(np.log(current_window))
                ref_returns = np.diff(np.log(ref_window))
                # 裁边后至少要留下若干点才有意义
                if current_returns.size < 3 * trim:
                    return np.nan

                # 去均值 + 零相位带通，隔离目标频段（周期 3-15 分钟）
                current_band = sosfiltfilt(
                    sos, current_returns - current_returns.mean()
                )
                ref_band = sosfiltfilt(sos, ref_returns - ref_returns.mean())

                # Hilbert 瞬时相位
                current_phase = np.angle(hilbert(current_band))
                ref_phase = np.angle(hilbert(ref_band))

                # 裁掉首尾边缘失真点后计算 PLV
                phase_diff = (current_phase - ref_phase)[trim:-trim]
                if phase_diff.size == 0:
                    return np.nan

                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                return float(plv) if np.isfinite(plv) else np.nan
            except Exception:
                return np.nan

        phase_sync = close.rolling(window=self.window).apply(
            calc_phase_sync, raw=False
        )
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
            current_returns = np.diff(np.log(current_window))
            ref_returns = np.diff(np.log(ref_window))
            
            if len(current_returns) != len(ref_returns):
                min_len = min(len(current_returns), len(ref_returns))
                current_returns = current_returns[-min_len:]
                ref_returns = ref_returns[-min_len:]
            
            try:
                bin_count = min(10, max(2, len(current_returns) // 5))
                current_discrete = np.floor(
                    stats.rankdata(current_returns)
                    / (len(current_returns) + 1)
                    * bin_count
                )
                ref_discrete = np.floor(
                    stats.rankdata(ref_returns)
                    / (len(ref_returns) + 1)
                    * bin_count
                )
                
                joint_hist, _, _ = np.histogram2d(
                    current_discrete,
                    ref_discrete,
                    bins=bin_count,
                    range=((0, bin_count), (0, bin_count)),
                )
                joint_prob = joint_hist / joint_hist.sum() + 1e-10
                
                current_marginal = joint_prob.sum(axis=1)
                ref_marginal = joint_prob.sum(axis=0)
                
                mutual_info = np.sum(joint_prob * np.log(joint_prob / (current_marginal[:, np.newaxis] * ref_marginal[np.newaxis, :] + 1e-10) + 1e-10))
                
                return max(0, mutual_info)
            except:
                return np.nan
        
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
                return np.nan
            
            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 50
            )
            if len(current_window) < 50:
                return np.nan
            
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
                        if np.std(current_seg) == 0 or np.std(ref_seg) == 0:
                            continue
                        corr, _ = pearsonr(current_seg, ref_seg)
                        if np.isfinite(corr):
                            segment_correlations.append(abs(corr))
                    except:
                        continue
                
                if segment_correlations:
                    correlations.append(np.mean(segment_correlations))
            
            if correlations:
                return np.mean(correlations)
            return np.nan
        
        multiscale = close.rolling(window=self.window).apply(calc_multiscale)
        return multiscale

class CrossMarketDynamicCorrelationFactor(BaseFactor):
    """跨市场动态相关性因子（指数衰减加权 Pearson 相关，输出范围 [-1, 1]）"""

    def __init__(self, window: int = 50, decay: float = 0.95):
        super().__init__('cross_market_dynamic_corr', window)
        self.decay = decay

    def calculate(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.Series:
        """加权 Pearson 相关：w = decay^(n-1-i)，归一化后计算加权均值、方差、协方差。"""
        close = df['close']

        if reference_df is None:
            reference_df = df.copy()

        ref_close = reference_df['close']
        decay = self.decay

        def calc_dynamic_corr(x):
            if len(x) < 30:
                return np.nan

            current_window, ref_window = _aligned_price_windows(
                x, ref_close, 30
            )
            if len(current_window) < 30:
                return np.nan

            try:
                x_ret = np.diff(np.log(current_window))
                y_ret = np.diff(np.log(ref_window))
                n = len(x_ret)

                # 指数衰减权重（最近的权重最大），归一化
                w = np.array([decay ** (n - 1 - i) for i in range(n)])
                w = w / w.sum()

                # 加权 Pearson 相关
                mean_x = np.dot(w, x_ret)
                mean_y = np.dot(w, y_ret)
                dx = x_ret - mean_x
                dy = y_ret - mean_y
                cov   = np.dot(w, dx * dy)
                var_x = np.dot(w, dx * dx)
                var_y = np.dot(w, dy * dy)

                if var_x <= 0 or var_y <= 0:
                    return np.nan

                corr = cov / np.sqrt(var_x * var_y)
                # 浮点误差保护，确保结果严格在 [-1, 1]
                corr = float(np.clip(corr, -1.0, 1.0))
                return corr if np.isfinite(corr) else np.nan
            except Exception:
                return np.nan

        dynamic_corr = close.rolling(window=self.window).apply(
            calc_dynamic_corr, raw=False
        )
        return dynamic_corr
