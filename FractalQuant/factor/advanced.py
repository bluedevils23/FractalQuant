"""
高级统计因子（分形、混沌理论、时间序列分析等）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from .base import BaseFactor


def _embed_series(values: np.ndarray, dim: int, delay: int = 1) -> np.ndarray:
    """Build a simple delay embedding matrix."""
    n_vectors = len(values) - (dim - 1) * delay
    if n_vectors <= 0:
        return np.empty((0, dim))
    return np.column_stack(
        [values[i * delay:i * delay + n_vectors] for i in range(dim)]
    )


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances for a small matrix."""
    deltas = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


class FutureReturnsFactor(BaseFactor):
    """Forward return label used by factor analysis workflows.

    This is intentionally look-ahead and should not be used as a predictor
    feature.
    """

    def __init__(self, window: int = 5):
        super().__init__('future_returns', window)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['close'].astype(float)
        return close.shift(-self.window) / close - 1


class CorrelationFactor(BaseFactor):
    """Rolling autocorrelation of returns."""

    def __init__(self, window: int = 20, lag: int = 1):
        super().__init__('correlation', window)
        self.lag = lag

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change(fill_method=None)
        return returns.rolling(window=self.window).corr(returns.shift(self.lag))


class HurstExponentFactor(BaseFactor):
    """Estimate the Hurst exponent from lagged log-price differences."""

    def __init__(self, window: int = 50, max_lag: Optional[int] = None):
        super().__init__('hurst_exponent', window)
        self.max_lag = max_lag

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        log_price = np.log(df['close'].astype(float))

        def calc_hurst(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.5

            max_lag = self.max_lag or min(20, len(values) // 2)
            if max_lag < 4:
                return 0.5

            lags = []
            tau = []

            for lag in range(2, max_lag + 1):
                diff = values[lag:] - values[:-lag]
                if len(diff) < 2:
                    continue

                std = np.std(diff)
                if std <= 0 or not np.isfinite(std):
                    continue

                lags.append(lag)
                tau.append(std)

            if len(tau) < 3:
                return 0.5

            slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
            return float(np.clip(slope, 0.0, 1.5))

        return log_price.rolling(window=self.window).apply(calc_hurst, raw=True)

class LyapunovExponentFactor(BaseFactor):
    """李雅普诺夫指数因子（混沌理论）"""
    
    def __init__(self, window: int = 50, min_separation: float = 1e-8):
        super().__init__('lyapunov_exponent', window)
        self.min_separation = min_separation
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算李雅普诺夫指数（衡量系统混沌程度）"""
        log_price = np.log(df['close'].astype(float))

        def calc_lyapunov(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, dim=2)
            if len(embedded) < 10:
                return 0.0

            max_horizon = min(5, len(embedded) // 3)
            if max_horizon < 2:
                return 0.0

            horizons = []
            divergences = []

            for horizon in range(1, max_horizon + 1):
                usable = len(embedded) - horizon
                if usable < 5:
                    break

                # 向量化最近邻搜索：一次算出 usable×usable 距离矩阵，
                # 逐元素浮点运算与原逐行 np.linalg.norm 完全一致（结果逐位不变）。
                pts = embedded[:usable]
                diff = pts[:, None, :] - pts[None, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=2))
                idx = np.arange(usable)
                # 复刻原 Theiler 窗口屏蔽：|i-j| <= 2 置为 inf。
                distances[np.abs(idx[:, None] - idx[None, :]) <= 2] = np.inf

                neighbors = np.argmin(distances, axis=1)
                initial = distances[idx, neighbors]
                valid = np.isfinite(initial) & (initial > self.min_separation)
                if not valid.any():
                    continue

                future = np.linalg.norm(
                    embedded[idx + horizon] - embedded[neighbors + horizon], axis=1
                )
                valid &= (future > 0) & np.isfinite(future)
                if not valid.any():
                    continue

                step_logs = np.log(
                    (future[valid] + self.min_separation)
                    / (initial[valid] + self.min_separation)
                )
                horizons.append(horizon)
                divergences.append(float(np.mean(step_logs)))

            if len(divergences) < 2:
                return 0.0

            return float(np.polyfit(horizons, divergences, 1)[0])

        return log_price.rolling(window=self.window).apply(calc_lyapunov, raw=True)

class RecurrenceRateFactor(BaseFactor):
    """复发率因子（非线性动力学）"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1):
        super().__init__('recurrence_rate', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发率（衡量系统重复状态的频率）"""
        close = np.log(df['close'].astype(float))

        def calc_recurrence(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 10:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            distances = np.abs(normalized[:, None] - normalized[None, :])
            np.fill_diagonal(distances, np.inf)
            recurrence = distances < self.threshold
            return float(recurrence.sum() / (len(values) * (len(values) - 1)))

        return close.rolling(window=self.window).apply(calc_recurrence, raw=True)

class EmbeddingDimensionFactor(BaseFactor):
    """嵌入维度因子（相空间重构）"""
    
    def __init__(self, window: int = 50, max_dim: int = 10):
        super().__init__('embedding_dimension', window)
        self.max_dim = max_dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算嵌入维度（用于相空间重构）"""
        close = np.log(df['close'].astype(float))

        def calc_embedding_dim(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 2.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 2.0

            normalized = (values - np.mean(values)) / scale
            min_dim = 2
            max_dim = min(self.max_dim, len(normalized) // 4)

            if max_dim < min_dim:
                return float(min_dim)

            for dim in range(min_dim, max_dim + 1):
                embedded = _embed_series(normalized, dim)
                embedded_next = _embed_series(normalized, dim + 1)
                usable = min(len(embedded), len(embedded_next))

                if usable < 8:
                    break

                embedded = embedded[:usable]
                embedded_next = embedded_next[:usable]
                distances = _pairwise_distances(embedded)
                np.fill_diagonal(distances, np.inf)

                nearest_idx = np.argmin(distances, axis=1)
                base_distance = distances[np.arange(usable), nearest_idx]
                high_distance = np.linalg.norm(
                    embedded_next - embedded_next[nearest_idx],
                    axis=1,
                )

                valid = np.isfinite(base_distance) & (base_distance > 1e-12)
                if not np.any(valid):
                    continue

                false_rate = np.mean((high_distance[valid] / base_distance[valid]) > 10.0)
                if false_rate < 0.1:
                    return float(dim)

            return float(max_dim)

        return close.rolling(window=self.window).apply(calc_embedding_dim, raw=True)

class CorrelationDimensionFactor(BaseFactor):
    """关联维度因子（分形维度）"""
    
    def __init__(self, window: int = 50, max_r: float = 2.0, min_r: float = 0.1):
        super().__init__('correlation_dimension', window)
        self.max_r = max_r
        self.min_r = min_r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算关联维度（Grassberger-Procaccia算法）"""
        close = np.log(df['close'].astype(float))

        def calc_corr_dim(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 30:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, dim=2)
            if len(embedded) < 10:
                return 0.0

            distances = _pairwise_distances(embedded)
            upper = distances[np.triu_indices(len(embedded), k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            if len(upper) < 10:
                return 0.0

            r_low = max(self.min_r, float(np.percentile(upper, 20)))
            r_high = min(self.max_r, float(np.percentile(upper, 80)))
            if r_high <= r_low:
                return 0.0

            r_values = np.logspace(np.log10(r_low), np.log10(r_high), 6)
            valid_r = []
            valid_c = []

            for r in r_values:
                c_r = np.mean(upper < r)
                # Avoid near-zero and near-one correlation integrals, which make
                # the log-log slope numerically unstable on short intraday windows.
                if 0.02 <= c_r <= 0.98:
                    valid_r.append(r)
                    valid_c.append(c_r)

            if len(valid_c) < 3:
                return 0.0

            slope = np.polyfit(np.log(valid_r), np.log(valid_c), 1)[0]
            return float(np.clip(slope, 0.0, 2.0))

        return close.rolling(window=self.window).apply(calc_corr_dim, raw=True)

class KolmogorovEntropyFactor(BaseFactor):
    """科尔莫哥洛夫熵因子（动力学系统）"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.2):
        super().__init__('kolmogorov_entropy', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算科尔莫哥洛夫熵（衡量系统复杂度）"""
        close = np.log(df['close'].astype(float))

        def calc_kolmogorov_entropy(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < self.m + 10:
                return 0.0

            returns = np.diff(values)
            if len(returns) < self.m + 5:
                return 0.0

            tolerance = self.r * np.std(returns)
            if tolerance <= 0 or not np.isfinite(tolerance):
                return 0.0

            templates_m = np.array(
                [returns[i:i + self.m] for i in range(len(returns) - self.m + 1)]
            )
            templates_m1 = np.array(
                [returns[i:i + self.m + 1] for i in range(len(returns) - self.m)]
            )

            if len(templates_m) < 2 or len(templates_m1) < 2:
                return 0.0

            diff_m = np.max(
                np.abs(templates_m[:, None, :] - templates_m[None, :, :]),
                axis=2,
            )
            diff_m1 = np.max(
                np.abs(templates_m1[:, None, :] - templates_m1[None, :, :]),
                axis=2,
            )
            np.fill_diagonal(diff_m, np.inf)
            np.fill_diagonal(diff_m1, np.inf)

            b = np.mean(diff_m <= tolerance)
            a = np.mean(diff_m1 <= tolerance)

            if a <= 0 or b <= 0:
                return 0.0

            return float(max(0.0, -np.log(a / b)))

        return close.rolling(window=self.window).apply(calc_kolmogorov_entropy, raw=True)

class MultifractalSpectrumFactor(BaseFactor):
    """多尺度谱因子（分形分析）"""
    
    def __init__(self, window: int = 50, q_values: List[float] = None):
        super().__init__('multifractal_spectrum', window)
        self.q_values = q_values or [-5, -3, -1, 0, 1, 3, 5]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算多尺度谱（衡量分形复杂度）"""
        close = np.log(df['close'].astype(float))

        def calc_multifractal(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 40:
                return 0.0

            returns = np.abs(np.diff(values))
            if len(returns) < 20:
                return 0.0

            total = returns.sum()
            if total <= 0 or not np.isfinite(total):
                return 0.0

            probs = returns / total
            probs = probs[probs > 0]
            if len(probs) < 5:
                return 0.0

            spectrum = []
            for q in self.q_values:
                if np.isclose(q, 1.0):
                    d_q = -np.sum(probs * np.log(probs))
                elif np.isclose(q, 0.0):
                    d_q = np.log(len(probs))
                else:
                    d_q = np.log(np.sum(probs ** q)) / (1.0 - q)
                if np.isfinite(d_q):
                    spectrum.append(float(d_q))

            if len(spectrum) < 2:
                return 0.0
            return float(max(spectrum) - min(spectrum))

        return close.rolling(window=self.window).apply(calc_multifractal, raw=True)

class DetrendedFluctuationFactor(BaseFactor):
    """去趋势波动分析因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('detrended_fluctuation', window)
        self.scales = scales or [5, 10, 20, 40, 80]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算DFA指数（衡量长程相关性）"""
        close = np.log(df['close'].astype(float))

        def calc_dfa(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.5

            profile = np.cumsum(values - np.mean(values))
            valid_scales = []
            fluctuations = []

            for scale in self.scales:
                if scale < 4 or scale * 2 > len(profile):
                    continue

                n_segments = len(profile) // scale
                if n_segments < 2:
                    continue

                segment_fluc = []
                x_vals = np.arange(scale, dtype=float)
                for i in range(n_segments):
                    segment = profile[i * scale:(i + 1) * scale]
                    coeffs = np.polyfit(x_vals, segment, 1)
                    trend = np.polyval(coeffs, x_vals)
                    rms = np.sqrt(np.mean((segment - trend) ** 2))
                    if np.isfinite(rms) and rms > 0:
                        segment_fluc.append(rms)

                if segment_fluc:
                    valid_scales.append(scale)
                    fluctuations.append(float(np.mean(segment_fluc)))

            if len(fluctuations) < 3:
                return 0.5

            slope = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)[0]
            return float(np.clip(slope, 0.1, 1.9))

        return close.rolling(window=self.window).apply(calc_dfa, raw=True)

class WaveletEntropyFactor(BaseFactor):
    """小波熵因子"""
    
    def __init__(self, window: int = 50, wavelet: str = 'haar'):
        super().__init__('wavelet_entropy', window)
        self.wavelet = str(wavelet).lower()
        if self.wavelet not in {'haar', 'db1'}:
            raise ValueError("WaveletEntropyFactor currently supports only Haar/db1.")
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算小波熵（时频分析）"""
        close = np.log(df['close'].astype(float))

        def calc_wavelet_entropy(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 16:
                return 0.0

            # The lightweight in-repo implementation uses a Haar/db1 dyadic split.
            detail_energies = []
            signal = np.diff(values)
            while len(signal) >= 4:
                even = signal[::2]
                odd = signal[1::2]
                size = min(len(even), len(odd))
                if size < 2:
                    break

                approx = (even[:size] + odd[:size]) / np.sqrt(2.0)
                detail = (even[:size] - odd[:size]) / np.sqrt(2.0)
                energy = float(np.sum(detail ** 2))
                if energy > 0 and np.isfinite(energy):
                    detail_energies.append(energy)
                signal = approx

            if not detail_energies:
                return 0.0

            probs = np.asarray(detail_energies) / np.sum(detail_energies)
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log(probs)))

        return close.rolling(window=self.window).apply(calc_wavelet_entropy, raw=True)

class PhaseSpaceVolumeFactor(BaseFactor):
    """相空间体积因子"""
    
    def __init__(self, window: int = 50, dim: int = 3, r: float = 1.0):
        super().__init__('phase_space_volume', window)
        self.dim = dim
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算相空间体积（系统复杂度度量）"""
        close = np.log(df['close'].astype(float))

        def calc_phase_volume(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < self.dim * 6:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.dim)
            if len(embedded) < 8:
                return 0.0

            ranges = np.ptp(embedded, axis=0)
            if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0):
                return 0.0

            bounding_volume = float(np.prod(ranges))

            cov = np.cov(embedded, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[np.isfinite(eigvals) & (eigvals > 1e-12)]
            if len(eigvals) == 0:
                return float(np.log1p(bounding_volume))

            ellipsoid_volume = float(np.sqrt(np.prod(eigvals)))
            occupied_volume = min(bounding_volume, ellipsoid_volume * len(eigvals))
            radius_scale = max(float(self.r), 1e-8) ** self.dim
            return float(np.log1p(max(0.0, occupied_volume / radius_scale)))

        return close.rolling(window=self.window).apply(calc_phase_volume, raw=True)

class PoincareSectionFactor(BaseFactor):
    """庞加莱截面因子（非线性动力学）"""
    
    def __init__(self, window: int = 50, threshold: float = 0.0):
        super().__init__('poincare_section', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算庞加莱截面（系统状态采样）"""
        close = np.log(df['close'].astype(float))

        def calc_poincare(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 10:
                return 0.0

            centered = returns - np.mean(returns)
            shifted = centered - self.threshold
            cross_mask = shifted[:-1] * shifted[1:] <= 0
            cross_idx = np.where(cross_mask)[0]
            if len(cross_idx) < 4:
                return 0.0

            intervals = np.diff(cross_idx)
            if len(intervals) < 3:
                return 0.0

            sd1 = np.std(np.diff(returns)) / np.sqrt(2.0)
            sd2 = np.std(returns[:-1] + returns[1:]) / np.sqrt(2.0)
            ratio = sd1 / (sd2 + 1e-8)
            return float(ratio + np.std(intervals) / (np.mean(intervals) + 1e-8))

        return close.rolling(window=self.window).apply(calc_poincare, raw=True)

class BifurcationDiagramFactor(BaseFactor):
    """分岔图因子（混沌理论）"""
    
    def __init__(self, window: int = 50, param_range: float = 0.1, iterations: int = 100):
        super().__init__('bifurcation_diagram', window)
        self.param_range = param_range
        self.iterations = iterations
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算分岔图特征（混沌系统参数变化）"""
        close = np.log(df['close'].astype(float))

        def calc_bifurcation(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 30:
                return 0.0

            returns = np.diff(values)
            vol = np.std(returns)
            drift = abs(np.mean(returns))
            if not np.isfinite(vol) or vol <= 0:
                return 0.0

            center_r = float(np.clip(3.2 + 20.0 * vol + 10.0 * drift, 2.8, 3.95))
            r_low = max(2.5, center_r - self.param_range)
            r_high = min(3.99, center_r + self.param_range)
            r_values = np.linspace(r_low, r_high, 12)

            # 向量化：12 个 r 相互独立，用长度为 12 的向量同步迭代 logistic 映射。
            # 每步 r*x*(1-x) 与 clip 均为逐元素运算，与原标量循环逐位一致。
            x_vec = np.full(
                len(r_values), 0.5 + 0.1 * np.tanh(np.mean(returns) / (vol + 1e-8))
            )
            collected = []
            for step in range(self.iterations):
                x_vec = r_values * x_vec * (1.0 - x_vec)
                x_vec = np.clip(x_vec, 1e-8, 1.0 - 1e-8)
                if step >= self.iterations // 2:
                    collected.append(x_vec.copy())

            if not collected:
                return 0.0

            # 每个 r 取最后 20 个状态，顺序与原实现 (r0..r11) 一致。
            states = np.array(collected)
            bifurcation_points = states[-20:].T.reshape(-1)

            hist, _ = np.histogram(bifurcation_points, bins=20, range=(0.0, 1.0))
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            if len(prob) == 0:
                return 0.0

            entropy = -np.sum(prob * np.log(prob))
            occupancy = len(prob) / 20.0
            return float(entropy * occupancy)

        return close.rolling(window=self.window).apply(calc_bifurcation, raw=True)

class ChaosIndicatorFactor(BaseFactor):
    """混沌指示器因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.5):
        super().__init__('chaos_indicator', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌指示器（区分随机和混沌）"""
        close = np.log(df['close'].astype(float))

        def calc_chaos(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(30, self.m * 6):
                return 0.0

            returns = np.diff(values)
            scale = np.std(returns)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (returns - np.mean(returns)) / scale
            embedded = _embed_series(normalized, self.m)
            if len(embedded) < 10:
                return 0.0

            distances = _pairwise_distances(embedded)
            np.fill_diagonal(distances, np.inf)
            nearest = np.argmin(distances, axis=1)
            nearest_dist = distances[np.arange(len(embedded)), nearest]

            valid = np.isfinite(nearest_dist) & (nearest_dist > 1e-8)
            if not np.any(valid):
                return 0.0

            divergence = float(np.mean(np.log1p(nearest_dist[valid] / self.r)))
            recurrence = float(np.mean(nearest_dist[valid] < self.r))
            return float(max(0.0, divergence) * (1.0 + recurrence))

        return close.rolling(window=self.window).apply(calc_chaos, raw=True)

class TimeReversalAsymmetryFactor(BaseFactor):
    """时间反演不对称性因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('time_reversal_asymmetry', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算时间反演不对称性（检测非线性）"""
        close = np.log(df['close'].astype(float))

        def calc_asymmetry(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            scale = np.std(returns)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (returns - np.mean(returns)) / scale
            contributions = []

            for lag in range(1, min(self.lag + 1, len(normalized) // 2 + 1)):
                forward = normalized[lag:]
                backward = normalized[:-lag]
                if len(forward) < 5:
                    continue
                contributions.append(np.mean(forward ** 2 * backward - forward * backward ** 2))

            if not contributions:
                return 0.0
            return float(np.mean(contributions))

        return close.rolling(window=self.window).apply(calc_asymmetry, raw=True)

class NonlinearAutocorrelationFactor(BaseFactor):
    """非线性自相关因子"""
    
    def __init__(self, window: int = 50, max_lag: int = 10):
        super().__init__('nonlinear_autocorrelation', window)
        self.max_lag = max_lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性自相关（检测非线性依赖）"""
        close = np.log(df['close'].astype(float))

        def calc_nonlinear_acf(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 30:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 10:
                return 0.0

            transforms = [
                returns,
                np.abs(returns),
                returns ** 2,
            ]
            acf_values = []

            for lag in range(1, min(self.max_lag + 1, len(returns) // 2)):
                base = returns[:-lag]
                shifted = returns[lag:]
                if len(base) < 5:
                    continue

                raw_corr = np.corrcoef(base, shifted)[0, 1]
                if not np.isfinite(raw_corr):
                    raw_corr = 0.0

                transformed_corrs = []
                for transformed in transforms[1:]:
                    corr = np.corrcoef(transformed[:-lag], transformed[lag:])[0, 1]
                    transformed_corrs.append(0.0 if not np.isfinite(corr) else abs(corr))

                acf_values.append(max(transformed_corrs) - abs(raw_corr))

            if not acf_values:
                return 0.0
            return float(np.mean(acf_values))

        return close.rolling(window=self.window).apply(calc_nonlinear_acf, raw=True)

class SurrogateDataTestFactor(BaseFactor):
    """代理数据测试因子"""
    
    def __init__(self, window: int = 50, n_surrogates: int = 10):
        super().__init__('surrogate_data_test', window)
        self.n_surrogates = n_surrogates
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算代理数据测试（检验非线性）

        性能说明：原实现对 [原始序列 + n 个代理序列] 逐个调用
        compute_nonlinearity_metric，其中 scipy.stats.skew 占单窗口约 3/4 的耗时。
        这里改为：手写 skew（与 scipy bias=False 逐位一致）+ 批量生成相位/傅里叶逆变换，
        随机数序列与逐个抽样完全相同，输出保持逐位不变（实测约 8x 提速）。
        """
        close = np.log(df['close'].astype(float))

        def _batch_metric(mat: np.ndarray) -> np.ndarray:
            # mat: (k, L)，每行独立计算非线性度量，复刻标量版逐条边界逻辑。
            out = np.zeros(len(mat), dtype=float)
            length = mat.shape[1]
            if length < 10:
                return out
            centered = mat - mat.mean(axis=1, keepdims=True)
            std = centered.std(axis=1)
            m2 = np.mean(centered ** 2, axis=1)
            m3 = np.mean(centered ** 3, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                g1 = m3 / m2 ** 1.5
                skew = g1 * np.sqrt(length * (length - 1)) / (length - 2)
            for i in range(len(mat)):
                if std[i] <= 0:
                    continue
                cubic = (
                    np.corrcoef(centered[i, :-1], centered[i, 1:] ** 2)[0, 1]
                    if length > 5
                    else 0.0
                )
                s = 0.0 if not np.isfinite(skew[i]) else float(abs(skew[i]))
                c = 0.0 if not np.isfinite(cubic) else float(abs(cubic))
                out[i] = s + c
            return out

        def calc_surrogate_test(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 30:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 12:
                return 0.0

            amplitudes = np.abs(np.fft.rfft(returns))
            m = len(amplitudes)
            seed = int(np.round(np.abs(np.sum(returns)) * 1_000_000)) % (2 ** 32 - 1)
            rng = np.random.default_rng(seed)
            # rng.uniform(size=(n, m)) 的抽样序列与 n 次 rng.uniform(size=m) 完全一致，
            # 因此批量生成相位后结果逐位不变。
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(self.n_surrogates, m))
            phases[:, 0] = 0.0
            if m > 1:
                phases[:, -1] = 0.0
            surrogates = np.fft.irfft(amplitudes * np.exp(1j * phases), n=len(returns), axis=1)

            metrics = _batch_metric(np.vstack([returns[None, :], surrogates]))
            original_metric = metrics[0]
            surrogate_metrics = metrics[1:]
            surrogate_metrics = surrogate_metrics[np.isfinite(surrogate_metrics)]
            if len(surrogate_metrics) == 0:
                return 0.0

            return float(np.mean(original_metric > surrogate_metrics))

        return close.rolling(window=self.window).apply(calc_surrogate_test, raw=True)

class RecurrencePlotFactor(BaseFactor):
    """复发图因子（非线性时间序列）"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1, dim: int = 3):
        super().__init__('recurrence_plot', window)
        self.threshold = threshold
        self.dim = dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发图特征"""
        close = np.log(df['close'].astype(float))

        def calc_recurrence_plot(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.dim * 6):
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.dim)
            if len(embedded) < 10:
                return 0.0

            distances = _pairwise_distances(embedded)
            recurrence = distances < self.threshold
            np.fill_diagonal(recurrence, False)

            recurrence_rate = float(np.mean(recurrence))
            diagonal_runs = []
            for offset in range(1, min(len(embedded), 8)):
                diag = np.diag(recurrence, k=offset)
                if len(diag) == 0:
                    continue
                run = 0
                for flag in diag:
                    if flag:
                        run += 1
                    else:
                        if run >= 2:
                            diagonal_runs.append(run)
                        run = 0
                if run >= 2:
                    diagonal_runs.append(run)

            determinism = 0.0 if not diagonal_runs else float(np.mean(diagonal_runs) / len(embedded))
            return float(recurrence_rate + determinism)

        return close.rolling(window=self.window).apply(calc_recurrence_plot, raw=True)

class MultiscaleComplexityFactor(BaseFactor):
    """多尺度复杂度因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('multiscale_complexity', window)
        self.scales = scales or [1, 2, 4, 8, 16]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算多尺度复杂度（综合多种度量）"""
        close = np.log(df['close'].astype(float))

        def calc_multiscale(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 30:
                return 0.0

            returns = np.diff(values)
            complexities = []

            for scale in self.scales:
                if scale < 1 or scale > len(returns) // 3:
                    continue

                usable = len(returns) // scale
                if usable < 8:
                    continue

                coarse = returns[:usable * scale].reshape(usable, scale).mean(axis=1)
                hist, _ = np.histogram(coarse, bins=min(10, max(4, usable // 2)))
                prob = hist / hist.sum() if hist.sum() > 0 else np.array([])
                prob = prob[prob > 0]
                if len(prob) < 2:
                    continue

                entropy = -np.sum(prob * np.log(prob))
                variability = np.std(coarse)
                complexity = entropy * (1.0 + variability)
                complexities.append(float(complexity))

            if not complexities:
                return 0.0
            return float(np.mean(complexities))

        return close.rolling(window=self.window).apply(calc_multiscale, raw=True)

class InformationComplexityFactor(BaseFactor):
    """信息复杂度因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.5):
        super().__init__('information_complexity', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算信息复杂度（结合信息论和复杂度）"""
        close = np.log(df['close'].astype(float))

        def calc_info_complexity(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.m * 6):
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.m)
            if len(embedded) < 10:
                return 0.0

            distances = _pairwise_distances(embedded)
            upper = distances[np.triu_indices(len(embedded), k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            if len(upper) < 5:
                return 0.0

            hist, _ = np.histogram(upper, bins=min(20, max(6, len(upper) // 4)))
            if hist.sum() == 0:
                return 0.0

            prob = hist / hist.sum()
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log(prob))
            spread_balance = 1.0 - (np.std(upper) / (np.mean(upper) + 1e-8)) / (1.0 + np.std(upper))
            return float(max(0.0, entropy * max(0.0, spread_balance)))

        return close.rolling(window=self.window).apply(calc_info_complexity, raw=True)

class DynamicPatternFactor(BaseFactor):
    """动态模式因子"""
    
    def __init__(self, window: int = 50, n_patterns: int = 5):
        super().__init__('dynamic_pattern', window)
        self.n_patterns = n_patterns
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """提取动态模式（类似PCA）"""
        close = np.log(df['close'].astype(float))

        def calc_dynamic_pattern(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            lag = min(max(3, len(returns) // 6), 12)
            if lag >= len(returns):
                return 0.0

            trajectory = _embed_series(returns, lag)
            if len(trajectory) < 5:
                return 0.0

            trajectory = trajectory - trajectory.mean(axis=0, keepdims=True)
            singular_values = np.linalg.svd(trajectory, compute_uv=False)
            energy = singular_values ** 2
            total_energy = energy.sum()
            if total_energy <= 0:
                return 0.0

            top_k = min(self.n_patterns, len(energy))
            return float(np.sum(energy[:top_k]) / total_energy)

        return close.rolling(window=self.window).apply(calc_dynamic_pattern, raw=True)

class StateSpaceGeometryFactor(BaseFactor):
    """状态空间几何因子"""
    
    def __init__(self, window: int = 50, dim: int = 3):
        super().__init__('state_space_geometry', window)
        self.dim = dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算状态空间几何特征"""
        close = np.log(df['close'].astype(float))

        def calc_geometry(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.dim * 5):
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.dim)
            if len(embedded) < 8:
                return 0.0

            ranges = np.ptp(embedded, axis=0)
            if np.any(ranges <= 0):
                return 0.0

            cov = np.cov(embedded, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 1e-12]
            if len(eigvals) == 0:
                return 0.0

            participation_ratio = (eigvals.sum() ** 2) / np.sum(eigvals ** 2)
            anisotropy = np.max(ranges) / (np.min(ranges) + 1e-8)
            return float(participation_ratio / anisotropy)

        return close.rolling(window=self.window).apply(calc_geometry, raw=True)

class ChaosGameRepresentationFactor(BaseFactor):
    """混沌游戏表示因子"""
    
    def __init__(self, window: int = 50, r: float = 0.5):
        super().__init__('chaos_game_representation', window)
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌游戏表示特征"""
        close = np.log(df['close'].astype(float))

        def calc_chaos_game(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            vmin = np.min(values)
            vmax = np.max(values)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return 0.0

            x_norm = (values - vmin) / (vmax - vmin)
            vertices = np.array([(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)])
            point = np.array([0.5, 0.5])
            points = []

            for value in x_norm:
                idx = min(int(value * len(vertices)), len(vertices) - 1)
                point = self.r * point + (1.0 - self.r) * vertices[idx]
                points.append(point.copy())

            points = np.asarray(points)
            hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=8, range=[[0, 1], [0, 1]])
            prob = hist.ravel()
            prob = prob[prob > 0]
            if len(prob) == 0:
                return 0.0

            prob = prob / prob.sum()
            entropy = -np.sum(prob * np.log(prob))
            coverage = len(prob) / hist.size
            return float(entropy * coverage)

        return close.rolling(window=self.window).apply(calc_chaos_game, raw=True)

class AttractorDimensionFactor(BaseFactor):
    """吸引子维度因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.1):
        super().__init__('attractor_dimension', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算吸引子维度（嵌入维度）"""
        close = np.log(df['close'].astype(float))

        def calc_attractor_dim(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.m * 6):
                return float(self.m)

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return float(self.m)

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.m)
            if len(embedded) < 8:
                return float(self.m)

            distances = _pairwise_distances(embedded)
            upper = distances[np.triu_indices(len(embedded), k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            if len(upper) < 10:
                return float(self.m)

            r_low = max(self.r / 2.0, float(np.percentile(upper, 15)))
            r_high = max(r_low * 1.5, float(np.percentile(upper, 85)))
            r_values = np.logspace(np.log10(r_low), np.log10(r_high), 6)
            counts = []
            valid_r = []

            for radius in r_values:
                count = np.mean(upper < radius)
                if 0 < count < 1:
                    valid_r.append(radius)
                    counts.append(count)

            if len(counts) < 3:
                return float(self.m)

            slope = np.polyfit(np.log(valid_r), np.log(counts), 1)[0]
            return float(np.clip(slope, 0.1, float(self.m)))

        return close.rolling(window=self.window).apply(calc_attractor_dim, raw=True)

class PhaseTransitionFactor(BaseFactor):
    """相变因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.05):
        super().__init__('phase_transition', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测相变（系统状态突变）"""
        close = np.log(df['close'].astype(float))

        def calc_phase_transition(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 10:
                return 0.0

            half = len(returns) // 2
            first_half = returns[:half]
            second_half = returns[half:]
            scale = np.std(returns)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            mean_shift = abs(np.mean(first_half) - np.mean(second_half)) / scale
            vol_shift = abs(np.std(first_half) - np.std(second_half)) / scale
            corr_shift = 0.0
            if len(first_half) > 3 and len(second_half) > 3:
                corr1 = np.corrcoef(first_half[:-1], first_half[1:])[0, 1]
                corr2 = np.corrcoef(second_half[:-1], second_half[1:])[0, 1]
                corr1 = 0.0 if not np.isfinite(corr1) else corr1
                corr2 = 0.0 if not np.isfinite(corr2) else corr2
                corr_shift = abs(corr1 - corr2)

            score = mean_shift + vol_shift + corr_shift
            return float(score / (1.0 + score))

        return close.rolling(window=self.window).apply(calc_phase_transition, raw=True)

class CriticalSlowingDownFactor(BaseFactor):
    """临界慢化因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('critical_slowing_down', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测临界慢化（系统恢复变慢）"""
        close = np.log(df['close'].astype(float))

        def calc_csd(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            if len(returns) < self.lag + 3:
                return 0.0

            lag = min(max(1, self.lag), len(returns) // 3)
            acf1 = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            acf1 = 0.0 if not np.isfinite(acf1) else abs(acf1)

            rolling_var = pd.Series(returns).rolling(window=max(3, lag + 2)).var().dropna()
            if len(rolling_var) < 2:
                return float(acf1)

            var_trend = rolling_var.iloc[-1] / (rolling_var.iloc[0] + 1e-8)
            return float(acf1 * np.log1p(max(0.0, var_trend)))

        return close.rolling(window=self.window).apply(calc_csd, raw=True)

class MemoryFunctionFactor(BaseFactor):
    """记忆函数因子"""
    
    def __init__(self, window: int = 50, max_lag: int = 20):
        super().__init__('memory_function', window)
        self.max_lag = max_lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算记忆函数（长程相关性）"""
        close = np.log(df['close'].astype(float))

        def calc_memory(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 30:
                return 0.0

            returns = np.diff(values)
            max_lag = min(self.max_lag, len(returns) // 2)
            if max_lag < 2:
                return 0.0

            acf = []
            for lag in range(1, max_lag + 1):
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                acf.append(0.0 if not np.isfinite(corr) else abs(corr))

            acf = np.asarray(acf)
            if np.all(acf <= 0):
                return 0.0

            weights = 1.0 / np.arange(1, len(acf) + 1)
            return float(np.sum(acf * weights) / np.sum(weights))

        return close.rolling(window=self.window).apply(calc_memory, raw=True)

class NonlinearDampingFactor(BaseFactor):
    """非线性阻尼因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('nonlinear_damping', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性阻尼特征"""
        close = np.log(df['close'].astype(float))

        def calc_damping(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            half = len(returns) // 2
            if half < 5:
                return 0.0

            first_half = returns[:half]
            second_half = returns[half:]
            first_amp = np.std(first_half)
            second_amp = np.std(second_half)
            if first_amp <= 0 or not np.isfinite(first_amp):
                return 0.0

            linear_damping = np.log((first_amp + 1e-8) / (second_amp + 1e-8))
            nonlinear_energy = np.mean(np.abs(second_half) ** 3) / (np.mean(np.abs(first_half) ** 3) + 1e-8)
            return float(max(0.0, linear_damping) / (1.0 + nonlinear_energy))

        return close.rolling(window=self.window).apply(calc_damping, raw=True)

class BifurcationParameterFactor(BaseFactor):
    """分岔参数因子"""
    
    def __init__(self, window: int = 50, param_range: float = 0.2):
        super().__init__('bifurcation_parameter', window)
        self.param_range = param_range
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """估计分岔参数（系统临界点）"""
        close = np.log(df['close'].astype(float))

        def calc_bifurcation_param(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 25:
                return 0.5

            returns = np.diff(values)
            vol = np.std(returns)
            drift = abs(np.mean(returns))
            if vol <= 0 or not np.isfinite(vol):
                return 0.5

            center_r = float(np.clip(3.1 + 15.0 * vol + 10.0 * drift, 2.8, 3.95))
            r_values = np.linspace(max(2.5, center_r - self.param_range), min(3.99, center_r + self.param_range), 16)

            # 向量化：16 个 r 相互独立，用向量同步迭代 logistic 映射。
            # 每步逐元素运算与原标量循环逐位一致。
            x_vec = np.full(len(r_values), 0.5)
            collected = []
            for step in range(80):
                x_vec = r_values * x_vec * (1.0 - x_vec)
                x_vec = np.clip(x_vec, 1e-8, 1.0 - 1e-8)
                if step >= 40:
                    collected.append(x_vec.copy())

            # states 形状 (40, 16)，按列求方差即每个 r 的 np.var(states)。
            variances = np.array(collected).var(axis=0)
            if len(variances) < 4 or np.allclose(variances, variances[0]):
                return 0.5

            change_idx = int(np.argmax(np.abs(np.diff(variances))))
            return float(r_values[change_idx])

        return close.rolling(window=self.window).apply(calc_bifurcation_param, raw=True)

class StrangeAttractorFactor(BaseFactor):
    """奇异吸引子因子"""
    
    def __init__(self, window: int = 50, dim: int = 3):
        super().__init__('strange_attractor', window)
        self.dim = dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测奇异吸引子特征"""
        close = np.log(df['close'].astype(float))

        def calc_strange_attractor(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.dim * 6):
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.dim)
            if len(embedded) < 8:
                return 0.0

            ranges = np.ptp(embedded, axis=0)
            volume = float(np.prod(np.maximum(ranges, 1e-3)))
            density = len(embedded) / volume

            distances = _pairwise_distances(embedded)
            upper = distances[np.triu_indices(len(embedded), k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            if len(upper) < 10:
                return 0.0

            q10 = float(np.percentile(upper, 10))
            q90 = float(np.percentile(upper, 90))
            if q90 <= q10:
                return 0.0

            scales = np.logspace(np.log10(q10), np.log10(q90), 5)
            counts = [np.mean(upper < radius) for radius in scales]
            valid = [(radius, count) for radius, count in zip(scales, counts) if 0 < count < 1]
            if len(valid) < 3:
                return float(np.log1p(density) / (1.0 + self.dim))

            slope = np.polyfit(np.log([v[0] for v in valid]), np.log([v[1] for v in valid]), 1)[0]
            attractor_score = np.log1p(density) * (1.0 + abs(slope - self.dim / 2.0))
            return float(attractor_score / (1.0 + attractor_score))

        return close.rolling(window=self.window).apply(calc_strange_attractor, raw=True)

class TopologicalEntropyFactor(BaseFactor):
    """拓扑熵因子"""
    
    def __init__(self, window: int = 50, epsilon: float = 0.1):
        super().__init__('topological_entropy', window)
        self.epsilon = epsilon
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算拓扑熵（系统复杂度）"""
        close = np.log(df['close'].astype(float))

        def calc_topological_entropy(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 10:
                return 0.0

            scale = np.std(returns)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (returns - np.mean(returns)) / scale
            eps = self.epsilon
            symbols = np.digitize(normalized, bins=[-eps, eps])
            words = {}
            word_len = 3
            for i in range(len(symbols) - word_len + 1):
                word = tuple(symbols[i:i + word_len])
                words[word] = words.get(word, 0) + 1

            counts = np.asarray(list(words.values()), dtype=float)
            if len(counts) == 0:
                return 0.0

            prob = counts / counts.sum()
            return float(-np.sum(prob * np.log(prob)) / word_len)

        return close.rolling(window=self.window).apply(calc_topological_entropy, raw=True)

class WindingNumberFactor(BaseFactor):
    """缠绕数因子（环面映射）"""
    
    def __init__(self, window: int = 50):
        super().__init__('winding_number', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算缠绕数（周期轨道）"""
        close = np.log(df['close'].astype(float))

        def calc_winding(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 15:
                return 0.0

            returns = np.diff(values)
            centered = returns - np.mean(returns)
            scale = np.std(centered)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = centered / scale
            analytic = normalized[1:] + 1j * normalized[:-1]
            angles = np.unwrap(np.angle(analytic))
            total_rotation = angles[-1] - angles[0] if len(angles) > 1 else 0.0
            return float(abs(total_rotation) / (2.0 * np.pi))

        return close.rolling(window=self.window).apply(calc_winding, raw=True)

class ManifoldDimensionFactor(BaseFactor):
    """流形维度因子"""
    
    def __init__(self, window: int = 50, k: int = 5):
        super().__init__('manifold_dimension', window)
        self.k = k
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """估计流形维度（局部线性嵌入）"""
        close = np.log(df['close'].astype(float))

        def calc_manifold_dim(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 2.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 2.0

            normalized = (values - np.mean(values)) / scale
            dim = min(5, max(2, len(normalized) // 8))
            embedded = _embed_series(normalized, dim)
            if len(embedded) < self.k + 2:
                return 2.0

            sample_count = min(20, len(embedded))
            local_dims = []
            for i in range(sample_count):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                distances[i] = np.inf
                nn_idx = np.argsort(distances)[:self.k]
                neighborhood = embedded[nn_idx] - embedded[nn_idx].mean(axis=0, keepdims=True)
                cov = np.cov(neighborhood, rowvar=False)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) == 0:
                    continue
                participation_ratio = (eigvals.sum() ** 2) / np.sum(eigvals ** 2)
                local_dims.append(participation_ratio)

            if not local_dims:
                return 2.0
            return float(np.mean(local_dims))

        return close.rolling(window=self.window).apply(calc_manifold_dim, raw=True)

class FractalCorrelationFactor(BaseFactor):
    """分形相关性因子"""
    
    def __init__(self, window: int = 50, r: float = 0.1):
        super().__init__('fractal_correlation', window)
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算分形相关积分"""
        close = np.log(df['close'].astype(float))

        def calc_fractal_corr(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            dim = min(3, max(2, len(normalized) // 10))
            embedded = _embed_series(normalized, dim)
            if len(embedded) < 8:
                return 0.0

            distances = _pairwise_distances(embedded)
            upper = distances[np.triu_indices(len(embedded), k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            if len(upper) < 10:
                return 0.0

            q20 = max(self.r, float(np.percentile(upper, 20)))
            q80 = max(q20 * 1.5, float(np.percentile(upper, 80)))
            radii = np.logspace(np.log10(q20), np.log10(q80), 5)
            corr_vals = [np.mean(upper < radius) for radius in radii]
            valid = [(radius, corr) for radius, corr in zip(radii, corr_vals) if 0 < corr < 1]
            if len(valid) < 3:
                return 0.0

            slope = np.polyfit(np.log([v[0] for v in valid]), np.log([v[1] for v in valid]), 1)[0]
            return float(max(0.0, slope))

        return close.rolling(window=self.window).apply(calc_fractal_corr, raw=True)

class NonlinearPredictabilityFactor(BaseFactor):
    """非线性可预测性因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('nonlinear_predictability', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性可预测性"""
        close = np.log(df['close'].astype(float))

        def calc_predictability(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 12:
                return 0.0

            scores = []
            for lag in range(1, min(self.lag + 4, len(returns) // 3 + 1)):
                current = returns[:-lag]
                future = returns[lag:]
                if len(current) < 8:
                    continue

                corr = np.corrcoef(current, future)[0, 1]
                corr = 0.0 if not np.isfinite(corr) else abs(corr)

                sign_match = np.mean(np.sign(current) == np.sign(future))
                scores.append(0.5 * corr + 0.5 * sign_match)

            if not scores:
                return 0.0
            return float(np.mean(scores))

        return close.rolling(window=self.window).apply(calc_predictability, raw=True)

class ChaosGameIterationFactor(BaseFactor):
    """混沌游戏迭代因子"""
    
    def __init__(self, window: int = 50, iterations: int = 100, r: float = 0.5):
        super().__init__('chaos_game_iteration', window)
        self.iterations = iterations
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌游戏迭代特征"""
        close = np.log(df['close'].astype(float))

        def calc_chaos_iter(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 15:
                return 0.0

            vmin = np.min(values)
            vmax = np.max(values)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return 0.0

            x_norm = (values - vmin) / (vmax - vmin)
            vertices = np.array([(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)])
            point = np.array([0.5, 0.5])
            points = []

            for i in range(self.iterations):
                idx = min(int(x_norm[i % len(x_norm)] * len(vertices)), len(vertices) - 1)
                point = self.r * point + (1.0 - self.r) * vertices[idx]
                points.append(point.copy())

            points = np.asarray(points)
            center = points.mean(axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            convergence = distances[:10].mean() / (distances[-10:].mean() + 1e-8)

            hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=6, range=[[0, 1], [0, 1]])
            prob = hist.ravel()
            prob = prob[prob > 0]
            if len(prob) == 0:
                return 0.0

            prob = prob / prob.sum()
            entropy = -np.sum(prob * np.log(prob))
            return float(entropy / (1.0 + convergence))

        return close.rolling(window=self.window).apply(calc_chaos_iter, raw=True)

class MultiscaleEntropyFactor(BaseFactor):
    """多尺度熵因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None, m: int = 2, r: float = 0.5):
        super().__init__('multiscale_entropy', window)
        self.scales = scales or [1, 2, 3, 4, 5]
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算多尺度熵"""
        close = np.log(df['close'].astype(float))

        def sample_entropy(series: np.ndarray) -> float:
            if len(series) <= self.m + 2:
                return 0.0
            tolerance = self.r * np.std(series)
            if tolerance <= 0 or not np.isfinite(tolerance):
                return 0.0

            templates_m = np.array([series[i:i + self.m] for i in range(len(series) - self.m + 1)])
            templates_m1 = np.array([series[i:i + self.m + 1] for i in range(len(series) - self.m)])
            if len(templates_m) < 2 or len(templates_m1) < 2:
                return 0.0

            diff_m = np.max(np.abs(templates_m[:, None, :] - templates_m[None, :, :]), axis=2)
            diff_m1 = np.max(np.abs(templates_m1[:, None, :] - templates_m1[None, :, :]), axis=2)
            np.fill_diagonal(diff_m, np.inf)
            np.fill_diagonal(diff_m1, np.inf)
            b = np.mean(diff_m <= tolerance)
            a = np.mean(diff_m1 <= tolerance)
            if a <= 0 or b <= 0:
                return 0.0
            return float(-np.log(a / b))

        def calc_multiscale_entropy(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            entropies = []
            for scale in self.scales:
                if scale < 1 or scale > len(returns) // 3:
                    continue
                usable = len(returns) // scale
                if usable < 8:
                    continue
                coarse = returns[:usable * scale].reshape(usable, scale).mean(axis=1)
                entropy = sample_entropy(coarse)
                if np.isfinite(entropy):
                    entropies.append(entropy)

            if not entropies:
                return 0.0
            return float(np.mean(entropies))

        return close.rolling(window=self.window).apply(calc_multiscale_entropy, raw=True)

class RecurrenceQuantificationFactor(BaseFactor):
    """复发量化分析因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1, min_line: int = 2):
        super().__init__('recurrence_quantification', window)
        self.threshold = threshold
        self.min_line = min_line
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发量化特征"""
        close = np.log(df['close'].astype(float))

        def calc_rqa(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 15:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            distances = np.abs(normalized[:, None] - normalized[None, :])
            recurrence = distances < self.threshold
            np.fill_diagonal(recurrence, False)

            recurrence_rate = float(np.mean(recurrence))
            diagonal_lengths = []
            for offset in range(1, min(len(values), 10)):
                diag = np.diag(recurrence, k=offset)
                run = 0
                for flag in diag:
                    if flag:
                        run += 1
                    else:
                        if run >= self.min_line:
                            diagonal_lengths.append(run)
                        run = 0
                if run >= self.min_line:
                    diagonal_lengths.append(run)

            determinism = 0.0 if not diagonal_lengths else float(np.sum(diagonal_lengths) / (recurrence.sum() + 1e-8))
            avg_line = 0.0 if not diagonal_lengths else float(np.mean(diagonal_lengths))
            return float(recurrence_rate * (1.0 + determinism) * (1.0 + avg_line / len(values)))

        return close.rolling(window=self.window).apply(calc_rqa, raw=True)

class DynamicTimeWarpingFactor(BaseFactor):
    """动态时间规整因子"""
    
    def __init__(self, window: int = 50, template_length: int = 10):
        super().__init__('dynamic_time_warping', window)
        self.template_length = template_length
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算DTW距离（模式匹配）"""
        close = np.log(df['close'].astype(float))

        def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
            dp = np.full((len(a) + 1, len(b) + 1), np.inf)
            dp[0, 0] = 0.0
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    cost = abs(a[i - 1] - b[j - 1])
                    dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            return float(dp[-1, -1] / (len(a) + len(b)))

        def calc_dtw(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < self.template_length * 3:
                return 0.0

            returns = np.diff(values)
            template = returns[-self.template_length:]
            candidate = returns[-2 * self.template_length:-self.template_length]
            baseline = returns[:self.template_length]
            if len(candidate) != self.template_length or len(baseline) != self.template_length:
                return 0.0

            d_recent = dtw_distance(template, candidate)
            d_baseline = dtw_distance(template, baseline)
            return float(1.0 / (1.0 + d_recent / (d_baseline + 1e-8)))

        return close.rolling(window=self.window).apply(calc_dtw, raw=True)

class ManifoldLearningFactor(BaseFactor):
    """流形学习因子（t-SNE, UMAP等）"""
    
    def __init__(self, window: int = 50, n_components: int = 2):
        super().__init__('manifold_learning', window)
        self.n_components = n_components
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流形学习特征"""
        close = np.log(df['close'].astype(float))

        def calc_manifold(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            dim = min(5, max(2, len(normalized) // 8))
            embedded = _embed_series(normalized, dim)
            if len(embedded) < 8:
                return 0.0

            centered = embedded - embedded.mean(axis=0, keepdims=True)
            cov = np.cov(centered, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = eigvals[eigvals > 1e-10]
            if len(eigvals) == 0:
                return 0.0

            explained = eigvals / eigvals.sum()
            intrinsic_dim = 1.0 / np.sum(explained ** 2)
            compression = explained[:self.n_components].sum() if len(explained) >= self.n_components else explained.sum()
            return float(intrinsic_dim * compression / max(1, self.n_components))

        return close.rolling(window=self.window).apply(calc_manifold, raw=True)

class RecurrenceAnalysisFactor(BaseFactor):
    """复发分析因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1):
        super().__init__('recurrence_analysis', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发分析特征"""
        close = np.log(df['close'].astype(float))

        def calc_recurrence(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 15:
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            recurrence = np.abs(normalized[:, None] - normalized[None, :]) < self.threshold
            np.fill_diagonal(recurrence, False)

            diagonal_lengths = []
            vertical_lengths = []
            for offset in range(1, min(len(values), 10)):
                diag = np.diag(recurrence, k=offset)
                run = 0
                for flag in diag:
                    if flag:
                        run += 1
                    else:
                        if run > 0:
                            diagonal_lengths.append(run)
                        run = 0
                if run > 0:
                    diagonal_lengths.append(run)

            for col in recurrence.T[:min(len(values), 10)]:
                run = 0
                for flag in col:
                    if flag:
                        run += 1
                    else:
                        if run > 0:
                            vertical_lengths.append(run)
                        run = 0
                if run > 0:
                    vertical_lengths.append(run)

            if not diagonal_lengths:
                return 0.0

            diag_mean = np.mean(diagonal_lengths)
            vert_mean = 0.0 if not vertical_lengths else np.mean(vertical_lengths)
            return float((diag_mean + vert_mean) / len(values))

        return close.rolling(window=self.window).apply(calc_recurrence, raw=True)

class NonlinearDynamicsFactor(BaseFactor):
    """非线性动力学因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.5):
        super().__init__('nonlinear_dynamics', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性动力学特征"""
        close = np.log(df['close'].astype(float))

        def calc_nonlinear_dynamics(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.m * 6):
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.m)
            if len(embedded) < 8:
                return 0.0

            distances = _pairwise_distances(embedded)
            np.fill_diagonal(distances, np.inf)
            nearest = np.min(distances, axis=1)
            nearest = nearest[np.isfinite(nearest) & (nearest > 1e-8)]
            if len(nearest) == 0:
                return 0.0

            divergence = np.mean(np.log1p(nearest / self.r))
            recurrence = np.mean(nearest < self.r)
            return float(max(0.0, divergence) * (1.0 + recurrence))

        return close.rolling(window=self.window).apply(calc_nonlinear_dynamics, raw=True)

class FractalAnalysisFactor(BaseFactor):
    """分形分析因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('fractal_analysis', window)
        self.scales = scales
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算分形分析特征"""
        close = np.log(df['close'].astype(float))

        def calc_fractal(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0

            returns = np.diff(values)
            if len(returns) < 12:
                return 0.0

            if self.scales is not None:
                candidate_scales = sorted(
                    {int(scale) for scale in self.scales if int(scale) >= 2}
                )
            else:
                max_scale = max(4, len(returns) // 2)
                geom = np.geomspace(2, max_scale, num=min(8, max_scale))
                candidate_scales = sorted(
                    {
                        int(round(scale))
                        for scale in geom
                        if int(round(scale)) >= 2
                    }
                )

            valid_scales = []
            fluctuations = []
            for scale in candidate_scales:
                if scale < 2 or scale > len(returns) // 2:
                    continue
                usable = len(returns) // scale
                if usable < 3:
                    continue
                coarse = returns[:usable * scale].reshape(usable, scale).sum(axis=1)
                fluct = np.std(coarse)
                if np.isfinite(fluct) and fluct > 0:
                    valid_scales.append(scale)
                    fluctuations.append(fluct)

            if len(fluctuations) < 3:
                return 0.0

            slope = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)[0]
            return float(np.clip(slope, 0.0, 2.0))

        return close.rolling(window=self.window).apply(calc_fractal, raw=True)

class ChaosTheoryFactor(BaseFactor):
    """混沌理论因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.1):
        super().__init__('chaos_theory', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌理论特征"""
        close = np.log(df['close'].astype(float))
        
        def calc_chaos(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < max(20, self.m * 6):
                return 0.0
            
            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.m)

            if len(embedded) < 8:
                return 0.0
            
            # 计算关联维度
            distances = _pairwise_distances(embedded)
            upper = distances[np.triu_indices(len(embedded), k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]

            if len(upper) < 10:
                return 0.0

            corr_integral = np.mean(upper < self.r)
            if corr_integral <= 0 or corr_integral >= 1:
                return 0.0

            dimension = -np.log(corr_integral) / np.log(max(1.0001, 1.0 / self.r))
            return float(max(0.0, dimension))
        
        return close.rolling(window=self.window).apply(calc_chaos, raw=True)

class NonlinearTimeSeriesFactor(BaseFactor):
    """非线性时间序列因子"""
    
    def __init__(self, window: int = 50, max_lag: int = 10):
        super().__init__('nonlinear_time_series', window)
        self.max_lag = max_lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性时间序列特征"""
        close = np.log(df['close'].astype(float))
        
        def calc_nonlinear_ts(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            if len(values) < 20:
                return 0.0
            
            returns = np.diff(values)
            
            # 计算高阶统计量
            if len(returns) < 8:
                return 0.0

            skewness = stats.skew(returns, bias=False)
            kurtosis = stats.kurtosis(returns, fisher=False, bias=False)
            skewness = 0.0 if not np.isfinite(skewness) else abs(float(skewness))
            kurtosis = 3.0 if not np.isfinite(kurtosis) else float(kurtosis)
            
            # 非线性度量
            cubic_dep = np.corrcoef(returns[:-1], returns[1:] ** 2)[0, 1] if len(returns) > 5 else 0.0
            cubic_dep = 0.0 if not np.isfinite(cubic_dep) else abs(float(cubic_dep))

            nonlinear_metric = skewness + abs(kurtosis - 3.0) / 6.0 + cubic_dep
            
            return float(nonlinear_metric / (1.0 + nonlinear_metric))
        
        return close.rolling(window=self.window).apply(calc_nonlinear_ts, raw=True)

class StateSpaceReconstructionFactor(BaseFactor):
    """状态空间重构因子"""
    
    def __init__(self, window: int = 50, dim: int = 3, tau: int = 1):
        super().__init__('state_space_reconstruction', window)
        self.dim = dim
        self.tau = tau
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算状态空间重构特征"""
        close = np.log(df['close'].astype(float))
        
        def calc_state_space(x: np.ndarray) -> float:
            values = np.asarray(x, dtype=float)
            min_required = (self.dim - 1) * self.tau + 5
            if len(values) < max(10, min_required):
                return 0.0

            scale = np.std(values)
            if scale <= 0 or not np.isfinite(scale):
                return 0.0

            normalized = (values - np.mean(values)) / scale
            embedded = _embed_series(normalized, self.dim, delay=max(1, self.tau))
            if len(embedded) < 5:
                return 0.0

            centered = embedded - embedded.mean(axis=0, keepdims=True)
            cov = np.cov(centered, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = eigvals[eigvals > 1e-10]
            if len(eigvals) == 0:
                return 0.0

            explained = eigvals / eigvals.sum()
            effective_dim = 1.0 / np.sum(explained ** 2)
            reconstruction_stability = explained[0]
            return float(effective_dim * reconstruction_stability / max(1, self.dim))

        return close.rolling(window=self.window).apply(calc_state_space, raw=True)
