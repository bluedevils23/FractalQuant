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

class LyapunovExponentFactor(BaseFactor):
    """李雅普诺夫指数因子（混沌理论）"""
    
    def __init__(self, window: int = 50, min_separation: float = 1e-8):
        super().__init__('lyapunov_exponent', window)
        self.min_separation = min_separation
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算李雅普诺夫指数（衡量系统混沌程度）"""
        log_price = np.log(df['close'])
        
        def calc_lyapunov(x):
            if len(x) < 20:
                return 0
            
            n = len(x)
            diffs = np.diff(x)
            log_diffs = np.log(np.abs(diffs) + self.min_separation)
            
            if len(log_diffs) < 10:
                return 0
                
            try:
                slope, _ = stats.linregress(np.arange(len(log_diffs)), log_diffs)[:2]
                return slope
            except:
                return 0
        
        lyapunov = log_price.rolling(window=self.window).apply(calc_lyapunov)
        return lyapunov

class RecurrenceRateFactor(BaseFactor):
    """复发率因子（非线性动力学）"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1):
        super().__init__('recurrence_rate', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发率（衡量系统重复状态的频率）"""
        close = df['close']
        
        def calc_recurrence(x):
            if len(x) < 10:
                return 0
            
            n = len(x)
            recurrence_count = 0
            
            for i in range(n - 5):
                for j in range(i + 5, n):
                    if abs(x[i] - x[j]) < self.threshold * np.std(x):
                        recurrence_count += 1
            
            total_pairs = n * (n - 1) / 2
            return recurrence_count / total_pairs if total_pairs > 0 else 0
        
        recurrence = close.rolling(window=self.window).apply(calc_recurrence)
        return recurrence

class EmbeddingDimensionFactor(BaseFactor):
    """嵌入维度因子（相空间重构）"""
    
    def __init__(self, window: int = 50, max_dim: int = 10):
        super().__init__('embedding_dimension', window)
        self.max_dim = max_dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算嵌入维度（用于相空间重构）"""
        close = df['close']
        
        def calc_embedding_dim(x):
            if len(x) < 20:
                return 5
            
            n = len(x)
            min_dim = 2
            max_dim = min(self.max_dim, n // 3)
            
            if max_dim < min_dim:
                return 5
            
            false_neighbors = []
            
            for dim in range(min_dim, max_dim + 1):
                embedded = np.array([x[i:i+dim] for i in range(n - dim)])
                
                false_count = 0
                total_count = len(embedded)
                
                for i in range(len(embedded)):
                    distances = np.linalg.norm(embedded - embedded[i], axis=1)
                    nearest_idx = np.argsort(distances)[1]
                    
                    if i + 1 < len(embedded):
                        embedded_next = np.array([x[i:i+dim+1] for i in range(n - dim - 1)])
                        if len(embedded_next) > nearest_idx:
                            dist_high = np.linalg.norm(embedded_next[i] - embedded_next[nearest_idx])
                            dist_low = distances[nearest_idx]
                            
                            if dist_high > 2 * dist_low:
                                false_count += 1
                
                false_rate = false_count / total_count if total_count > 0 else 1
                false_neighbors.append(false_rate)
                
                if false_rate < 0.1 and dim > min_dim:
                    return dim
            
            return max_dim if false_neighbors else 5
        
        embedding_dim = close.rolling(window=self.window).apply(calc_embedding_dim)
        return embedding_dim

class CorrelationDimensionFactor(BaseFactor):
    """关联维度因子（分形维度）"""
    
    def __init__(self, window: int = 50, max_r: float = 2.0, min_r: float = 0.1):
        super().__init__('correlation_dimension', window)
        self.max_r = max_r
        self.min_r = min_r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算关联维度（Grassberger-Procaccia算法）"""
        close = df['close']
        
        def calc_corr_dim(x):
            if len(x) < 30:
                return 2.0
            
            n = len(x)
            x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-8)
            
            r_values = np.logspace(np.log10(self.min_r), np.log10(self.max_r), 20)
            log_r = np.log(r_values)
            
            log_c = []
            for r in r_values:
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(x_normalized[i] - x_normalized[j]) < r:
                            count += 1
                c = count / (n * (n - 1) / 2)
                if c > 0:
                    log_c.append(np.log(c))
            
            if len(log_c) < 5:
                return 2.0
            
            try:
                slope, _ = stats.linregress(log_r[:len(log_c)], log_c)[:2]
                return max(0.1, slope)
            except:
                return 2.0
        
        corr_dim = close.rolling(window=self.window).apply(calc_corr_dim)
        return corr_dim

class KolmogorovEntropyFactor(BaseFactor):
    """科尔莫哥洛夫熵因子（动力学系统）"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.2):
        super().__init__('kolmogorov_entropy', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算科尔莫哥洛夫熵（衡量系统复杂度）"""
        close = df['close']
        
        def calc_kolmogorov_entropy(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            embedded = np.array([x[i:i+self.m] for i in range(n - self.m)])
            
            if len(embedded) < 100:
                return 0
            
            tau_values = [1, 2, 3, 5, 10]
            entropy_estimates = []
            
            for tau in tau_values:
                if tau >= len(embedded) // 2:
                    continue
                    
                match_count = 0
                total_count = len(embedded)
                
                for i in range(len(embedded) - tau):
                    for j in range(i + tau, len(embedded)):
                        if np.max(np.abs(embedded[i] - embedded[j])) < self.r:
                            match_count += 1
                
                p_match = match_count / total_count if total_count > 0 else 0
                
                if p_match > 0 and p_match < 1:
                    ks_entropy = -np.log(p_match) / tau
                    entropy_estimates.append(max(0, ks_entropy))
            
            return np.mean(entropy_estimates) if entropy_estimates else 0
        
        entropy = close.rolling(window=self.window).apply(calc_kolmogorov_entropy)
        return entropy

class MultifractalSpectrumFactor(BaseFactor):
    """多尺度谱因子（分形分析）"""
    
    def __init__(self, window: int = 50, q_values: List[float] = None):
        super().__init__('multifractal_spectrum', window)
        self.q_values = q_values or [-5, -3, -1, 0, 1, 3, 5]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算多尺度谱（衡量分形复杂度）"""
        close = df['close']
        
        def calc_multifractal(x):
            if len(x) < 100:
                return 0
            
            q_values = self.q_values
            spectrum = []
            
            for q in q_values:
                if q == 0:
                    hist, _ = np.histogram(x, bins=10)
                    prob = hist / hist.sum()
                    prob = prob[prob > 0]
                    d_q = -np.log(np.sum(prob ** q)) / (q - 1) if q != 1 else np.exp(-np.sum(prob * np.log(prob)))
                else:
                    d_q = np.log(np.sum(prob ** q)) / (q - 1) if q != 1 else 0
                
                spectrum.append(d_q)
            
            if len(spectrum) > 1:
                spectrum_width = max(spectrum) - min(spectrum)
                return spectrum_width
            return 0
        
        multifractal = close.rolling(window=self.window).apply(calc_multifractal)
        return multifractal

class DetrendedFluctuationFactor(BaseFactor):
    """去趋势波动分析因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('detrended_fluctuation', window)
        self.scales = scales or [5, 10, 20, 40, 80]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算DFA指数（衡量长程相关性）"""
        close = df['close']
        
        def calc_dfa(x):
            if len(x) < 100:
                return 0.5
            
            y = np.cumsum(x - np.mean(x))
            
            scales = self.scales
            fluc = []
            
            for scale in scales:
                if scale * 2 > len(y):
                    continue
                    
                n_segments = len(y) // scale
                segment_fluc = []
                
                for i in range(n_segments):
                    start = i * scale
                    end = start + scale
                    
                    if end > len(y):
                        continue
                    
                    x_vals = np.arange(scale)
                    y_vals = y[start:end]
                    
                    try:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        trend = slope * x_vals + intercept
                        residual = y_vals - trend
                        rms = np.sqrt(np.mean(residual ** 2))
                        segment_fluc.append(rms)
                    except:
                        continue
                
                if segment_fluc:
                    fluc.append(np.mean(segment_fluc))
            
            if len(fluc) < 3:
                return 0.5
            
            log_scales = np.log(self.scales[:len(fluc)])
            log_fluc = np.log(fluc)
            
            try:
                slope, _ = stats.linregress(log_scales, log_fluc)[:2]
                return max(0.1, min(1.9, slope))
            except:
                return 0.5
        
        dfa = close.rolling(window=self.window).apply(calc_dfa)
        return dfa

class WaveletEntropyFactor(BaseFactor):
    """小波熵因子"""
    
    def __init__(self, window: int = 50, wavelet: str = 'db4'):
        super().__init__('wavelet_entropy', window)
        self.wavelet = wavelet
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算小波熵（时频分析）"""
        close = df['close']
        
        def calc_wavelet_entropy(x):
            if len(x) < 32:
                return 0
            
            returns = np.diff(x)
            
            scales = [2, 4, 8, 16]
            entropies = []
            
            for scale in scales:
                if len(returns) < scale * 2:
                    continue
                    
                n_segments = len(returns) // scale
                segment_entropies = []
                
                for i in range(n_segments):
                    segment = returns[i*scale:(i+1)*scale]
                    if len(segment) < 5:
                        continue
                    
                    hist, _ = np.histogram(segment, bins=10)
                    prob = hist / hist.sum()
                    prob = prob[prob > 0]
                    
                    if len(prob) > 0:
                        entropy = -np.sum(prob * np.log2(prob + 1e-10))
                        segment_entropies.append(entropy)
                
                if segment_entropies:
                    entropies.append(np.mean(segment_entropies))
            
            return np.mean(entropies) if entropies else 0
        
        wavelet_entropy = close.rolling(window=self.window).apply(calc_wavelet_entropy)
        return wavelet_entropy

class PhaseSpaceVolumeFactor(BaseFactor):
    """相空间体积因子"""
    
    def __init__(self, window: int = 50, dim: int = 3, r: float = 1.0):
        super().__init__('phase_space_volume', window)
        self.dim = dim
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算相空间体积（系统复杂度度量）"""
        close = df['close']
        
        def calc_phase_volume(x):
            if len(x) < self.dim * 10:
                return 0
            
            n = len(x)
            
            embedded = np.array([x[i:i+self.dim] for i in range(n - self.dim)])
            
            if len(embedded) < 100:
                return 0
            
            distances = []
            for i in range(len(embedded)):
                for j in range(i + 1, len(embedded)):
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)
            
            if not distances:
                return 0
            
            count_in_sphere = sum(1 for d in distances if d < self.r)
            total_pairs = len(distances)
            
            volume = count_in_sphere / total_pairs if total_pairs > 0 else 0
            
            return volume * 100
        
        phase_volume = close.rolling(window=self.window).apply(calc_phase_volume)
        return phase_volume

class PoincareSectionFactor(BaseFactor):
    """庞加莱截面因子（非线性动力学）"""
    
    def __init__(self, window: int = 50, threshold: float = 0.0):
        super().__init__('poincare_section', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算庞加莱截面（系统状态采样）"""
        close = df['close']
        
        def calc_poincare(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            crossings = []
            for i in range(1, len(returns)):
                if (returns[i-1] - self.threshold) * (returns[i] - self.threshold) < 0:
                    t = (self.threshold - returns[i-1]) / (returns[i] - returns[i-1] + 1e-8)
                    crossing = returns[i-1] + t * (returns[i] - returns[i-1])
                    crossings.append(crossing)
            
            if len(crossings) < 10:
                return 0
            
            crossings = np.array(crossings)
            
            width = np.std(crossings)
            
            try:
                skewness = stats.skew(crossings)
            except:
                skewness = 0
            
            return abs(skewness) + width / (np.mean(np.abs(crossings)) + 1e-8)
        
        poincare = close.rolling(window=self.window).apply(calc_poincare)
        return poincare

class BifurcationDiagramFactor(BaseFactor):
    """分岔图因子（混沌理论）"""
    
    def __init__(self, window: int = 50, param_range: float = 0.1, iterations: int = 100):
        super().__init__('bifurcation_diagram', window)
        self.param_range = param_range
        self.iterations = iterations
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算分岔图特征（混沌系统参数变化）"""
        close = df['close']
        
        def calc_bifurcation(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            mean_return = np.mean(returns)
            
            r_values = np.linspace(mean_return - self.param_range, mean_return + self.param_range, 20)
            bifurcation_points = []
            
            for r in r_values:
                x_val = 0.5
                states = []
                
                for _ in range(self.iterations):
                    x_val = r * x_val * (1 - x_val + 1e-8)
                    if _ > self.iterations // 2:
                        states.append(x_val)
                
                if states:
                    bifurcation_points.extend(states)
            
            if not bifurcation_points:
                return 0
            
            hist, _ = np.histogram(bifurcation_points, bins=20)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            
            if len(prob) > 0:
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                return entropy / 5
            return 0
        
        bifurcation = close.rolling(window=self.window).apply(calc_bifurcation)
        return bifurcation

class ChaosIndicatorFactor(BaseFactor):
    """混沌指示器因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.5):
        super().__init__('chaos_indicator', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌指示器（区分随机和混沌）"""
        close = df['close']
        
        def calc_chaos(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            embedded = np.array([x[i:i+self.m] for i in range(n - self.m)])
            
            if len(embedded) < 100:
                return 0
            
            lyapunov_estimates = []
            
            for i in range(min(50, len(embedded))):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                nearest_idx = np.argsort(distances)[1]
                
                if nearest_idx < len(embedded):
                    separation = np.linalg.norm(embedded[nearest_idx] - embedded[i])
                    if separation > 0:
                        lyapunov_estimates.append(np.log(separation + 1e-8))
            
            if not lyapunov_estimates:
                return 0
            
            mean_lyapunov = np.mean(lyapunov_estimates)
            
            return max(0, mean_lyapunov)
        
        chaos = close.rolling(window=self.window).apply(calc_chaos)
        return chaos

class TimeReversalAsymmetryFactor(BaseFactor):
    """时间反演不对称性因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('time_reversal_asymmetry', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算时间反演不对称性（检测非线性）"""
        close = df['close']
        
        def calc_asymmetry(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            asymmetry = 0
            
            for k in range(1, self.lag + 1):
                if k >= len(returns):
                    break
                    
                forward = returns[k:]
                backward = returns[:-k]
                
                asymmetry += np.mean(forward ** 2 * backward) - np.mean(forward * backward ** 2)
            
            return asymmetry / (self.lag + 1)
        
        asymmetry = close.rolling(window=self.window).apply(calc_asymmetry)
        return asymmetry

class NonlinearAutocorrelationFactor(BaseFactor):
    """非线性自相关因子"""
    
    def __init__(self, window: int = 50, max_lag: int = 10):
        super().__init__('nonlinear_autocorrelation', window)
        self.max_lag = max_lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性自相关（检测非线性依赖）"""
        close = df['close']
        
        def calc_nonlinear_acf(x):
            if len(x) < 30:
                return 0
            
            returns = np.diff(x)
            returns_abs = np.abs(returns)
            
            acf_values = []
            
            for lag in range(1, min(self.max_lag + 1, len(returns) // 2)):
                n = len(returns) - lag
                
                acf_raw = np.corrcoef(returns[lag:], returns[:n])[0, 1] if n > 0 else 0
                acf_abs = np.corrcoef(returns_abs[lag:], returns_abs[:n])[0, 1] if n > 0 else 0
                
                nonlinear = abs(acf_abs) - abs(acf_raw)
                acf_values.append(nonlinear)
            
            return np.mean(acf_values) if acf_values else 0
        
        nonlinear_acf = close.rolling(window=self.window).apply(calc_nonlinear_acf)
        return nonlinear_acf

class SurrogateDataTestFactor(BaseFactor):
    """代理数据测试因子"""
    
    def __init__(self, window: int = 50, n_surrogates: int = 10):
        super().__init__('surrogate_data_test', window)
        self.n_surrogates = n_surrogates
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算代理数据测试（检验非线性）"""
        close = df['close']
        
        def calc_surrogate_test(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            original_metric = self._compute_nonlinearity_metric(returns)
            
            surrogate_metrics = []
            
            for _ in range(self.n_surrogates):
                fft_returns = np.fft.fft(returns)
                phases = np.random.uniform(0, 2 * np.pi, len(fft_returns))
                fft_surrogate = np.abs(fft_returns) * np.exp(1j * phases)
                surrogate = np.real(np.fft.ifft(fft_surrogate))
                
                metric = self._compute_nonlinearity_metric(surrogate)
                surrogate_metrics.append(metric)
            
            if surrogate_metrics:
                p_value = sum(1 for s in surrogate_metrics if s > original_metric) / len(surrogate_metrics)
                return 1 - p_value
            return 0
        
        def _compute_nonlinearity_metric(x):
            if len(x) < 20:
                return 0
            return abs(stats.skew(x))
        
        surrogate_test = close.rolling(window=self.window).apply(calc_surrogate_test)
        return surrogate_test

class RecurrencePlotFactor(BaseFactor):
    """复发图因子（非线性时间序列）"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1, dim: int = 3):
        super().__init__('recurrence_plot', window)
        self.threshold = threshold
        self.dim = dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发图特征"""
        close = df['close']
        
        def calc_recurrence_plot(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            if n <= self.dim:
                return 0
                
            embedded = np.array([x[i:i+self.dim] for i in range(n - self.dim)])
            
            if len(embedded) < 20:
                return 0
            
            n_points = len(embedded)
            recurrence_count = 0
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if np.linalg.norm(embedded[i] - embedded[j]) < self.threshold:
                        recurrence_count += 1
            
            total_pairs = n_points * (n_points - 1) / 2
            recurrence_rate = recurrence_count / total_pairs if total_pairs > 0 else 0
            
            diagonal_lengths = []
            for i in range(min(10, n_points)):
                length = 0
                for j in range(1, min(10, n_points - i)):
                    if np.linalg.norm(embedded[i+j] - embedded[i+j-1]) < self.threshold:
                        length += 1
                    else:
                        break
                if length > 0:
                    diagonal_lengths.append(length)
            
            avg_diagonal = np.mean(diagonal_lengths) if diagonal_lengths else 0
            
            return recurrence_rate * 100 + avg_diagonal * 0.1
        
        recurrence_plot = close.rolling(window=self.window).apply(calc_recurrence_plot)
        return recurrence_plot

class MultiscaleComplexityFactor(BaseFactor):
    """多尺度复杂度因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('multiscale_complexity', window)
        self.scales = scales or [1, 2, 4, 8, 16]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算多尺度复杂度（综合多种度量）"""
        close = df['close']
        
        def calc_multiscale(x):
            if len(x) < 100:
                return 0
            
            complexities = []
            
            for scale in self.scales:
                if scale >= len(x) // 2:
                    continue
                
                downsampled = x[::scale]
                
                if len(downsampled) < 20:
                    continue
                
                hist, _ = np.histogram(downsampled, bins=10)
                prob = hist / hist.sum()
                prob = prob[prob > 0]
                entropy = -np.sum(prob * np.log2(prob + 1e-10)) if len(prob) > 0 else 0
                
                cv = np.std(downsampled) / (np.mean(downsampled) + 1e-8)
                
                if len(downsampled) > 10:
                    acf = np.corrcoef(downsampled[:-5], downsampled[5:])[0, 1] if len(downsampled) > 5 else 0
                else:
                    acf = 0
                
                complexity = entropy * (1 + abs(acf)) * (1 + cv / 10)
                complexities.append(complexity)
            
            return np.mean(complexities) if complexities else 0
        
        multiscale_complexity = close.rolling(window=self.window).apply(calc_multiscale)
        return multiscale_complexity

class InformationComplexityFactor(BaseFactor):
    """信息复杂度因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.5):
        super().__init__('information_complexity', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算信息复杂度（结合信息论和复杂度）"""
        close = df['close']
        
        def calc_info_complexity(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            if n <= self.m:
                return 0
                
            embedded = np.array([x[i:i+self.m] for i in range(n - self.m)])
            
            if len(embedded) < 50:
                return 0
            
            distances = []
            for i in range(len(embedded)):
                for j in range(i + 1, len(embedded)):
                    distances.append(np.linalg.norm(embedded[i] - embedded[j]))
            
            if not distances:
                return 0
            
            hist, _ = np.histogram(distances, bins=20)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            
            if len(prob) < 2:
                return 0
            
            entropy = -np.sum(prob * np.log2(prob + 1e-10))
            
            mean_dist = np.mean(distances)
            variance = np.var(distances)
            
            complexity = entropy * (1 - abs(2 * mean_dist - np.min(distances) - np.max(distances)) / 
                                   (np.max(distances) - np.min(distances) + 1e-8))
            
            return complexity
        
        info_complexity = close.rolling(window=self.window).apply(calc_info_complexity)
        return info_complexity

class DynamicPatternFactor(BaseFactor):
    """动态模式因子"""
    
    def __init__(self, window: int = 50, n_patterns: int = 5):
        super().__init__('dynamic_pattern', window)
        self.n_patterns = n_patterns
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """提取动态模式（类似PCA）"""
        close = df['close']
        
        def calc_dynamic_pattern(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            lag = min(n // 3, 10)
            if lag < 2:
                return 0
            
            n_vectors = n - lag + 1
            trajectory = np.array([x[i:i+lag] for i in range(n_vectors)])
            
            try:
                U, S, Vt = np.linalg.svd(trajectory, full_matrices=False)
                
                explained_variance = S[:self.n_patterns] ** 2
                total_variance = S ** 2
                
                if np.sum(total_variance) > 0:
                    pattern_strength = np.sum(explained_variance) / np.sum(total_variance)
                    return pattern_strength
                return 0
            except:
                return 0
        
        dynamic_pattern = close.rolling(window=self.window).apply(calc_dynamic_pattern)
        return dynamic_pattern

class StateSpaceGeometryFactor(BaseFactor):
    """状态空间几何因子"""
    
    def __init__(self, window: int = 50, dim: int = 3):
        super().__init__('state_space_geometry', window)
        self.dim = dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算状态空间几何特征"""
        close = df['close']
        
        def calc_geometry(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            if n <= self.dim:
                return 0
                
            embedded = np.array([x[i:i+self.dim] for i in range(n - self.dim)])
            
            if len(embedded) < 20:
                return 0
            
            ranges = [np.max(embedded[:, i]) - np.min(embedded[:, i]) for i in range(self.dim)]
            volume = np.prod(ranges)
            
            if volume > 0:
                surface_area = 2 * np.pi * (3 * volume / (4 * np.pi)) ** (2/3) if self.dim == 3 else 2 * np.pi * (volume / np.pi)
                sphereicity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area if volume > 0 else 1
            else:
                sphereicity = 1
            
            try:
                box_size = np.min(ranges) / 10 if ranges else 0.1
                n_boxes = 0
                for i in range(self.dim):
                    min_val = np.min(embedded[:, i])
                    max_val = np.max(embedded[:, i])
                    n_boxes += int((max_val - min_val) / box_size) + 1
                
                if n_boxes > 0 and volume > 0:
                    fractal_dim = np.log(n_boxes) / np.log(1 / box_size)
                else:
                    fractal_dim = self.dim
            except:
                fractal_dim = self.dim
            
            return sphereicity * fractal_dim
        
        state_space_geom = close.rolling(window=self.window).apply(calc_geometry)
        return state_space_geom

class ChaosGameRepresentationFactor(BaseFactor):
    """混沌游戏表示因子"""
    
    def __init__(self, window: int = 50, r: float = 0.5):
        super().__init__('chaos_game_representation', window)
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌游戏表示特征"""
        close = df['close']
        
        def calc_chaos_game(x):
            if len(x) < 50:
                return 0
            
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            
            n_points = 1000
            points = []
            x_val, y_val = 0.5, 0.5
            
            for i in range(n_points):
                vertex_idx = np.random.randint(0, 3)
                
                vertices = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
                vx, vy = vertices[vertex_idx]
                
                x_val = self.r * x_val + (1 - self.r) * vx
                y_val = self.r * y_val + (1 - self.r) * vy
                
                if i > 100:
                    points.append((x_val, y_val))
            
            if not points:
                return 0
            
            points = np.array(points)
            
            from scipy.spatial.distance import pdist
            
            if len(points) > 10:
                distances = pdist(points)
                cluster_degree = np.std(distances) / (np.mean(distances) + 1e-8)
            else:
                cluster_degree = 0
            
            hist_2d, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=10)
            prob = hist_2d.flatten()
            prob = prob[prob > 0]
            uniformity = -np.sum(prob * np.log2(prob + 1e-10)) / np.log2(len(prob)) if len(prob) > 0 else 0
            
            return cluster_degree * uniformity
        
        chaos_game = close.rolling(window=self.window).apply(calc_chaos_game)
        return chaos_game

class AttractorDimensionFactor(BaseFactor):
    """吸引子维度因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.1):
        super().__init__('attractor_dimension', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算吸引子维度（嵌入维度）"""
        close = df['close']
        
        def calc_attractor_dim(x):
            if len(x) < 50:
                return 2.0
            
            n = len(x)
            
            if n <= self.m:
                return 2.0
                
            embedded = np.array([x[i:i+self.m] for i in range(n - self.m)])
            
            if len(embedded) < 100:
                return 2.0
            
            r_values = np.logspace(np.log10(self.r / 10), np.log10(self.r * 10), 10)
            log_r = np.log(r_values)
            log_n = []
            
            for r in r_values:
                count = 0
                for i in range(len(embedded)):
                    for j in range(i + 1, len(embedded)):
                        if np.linalg.norm(embedded[i] - embedded[j]) < r:
                            count += 1
                log_n.append(np.log(count + 1))
            
            if len(log_n) < 5:
                return 2.0
            
            try:
                slope, _ = stats.linregress(log_r, log_n)[:2]
                return max(0.1, slope)
            except:
                return 2.0
        
        attractor_dim = close.rolling(window=self.window).apply(calc_attractor_dim)
        return attractor_dim

class PhaseTransitionFactor(BaseFactor):
    """相变因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.05):
        super().__init__('phase_transition', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测相变（系统状态突变）"""
        close = df['close']
        
        def calc_phase_transition(x):
            if len(x) < 30:
                return 0
            
            returns = np.diff(x)
            
            n = len(returns)
            half = n // 2
            
            first_half = returns[:half]
            second_half = returns[half:]
            
            mean_diff = abs(np.mean(first_half) - np.mean(second_half)) / (np.std(returns) + 1e-8)
            var_diff = abs(np.var(first_half) - np.var(second_half)) / (np.var(returns) + 1e-8)
            skew_diff = abs(stats.skew(first_half) - stats.skew(second_half)) / (stats.skew(returns) + 1e-8)
            
            phase_transition = mean_diff + var_diff + abs(skew_diff)
            
            return min(phase_transition, 10) / 10
        
        phase_transition = close.rolling(window=self.window).apply(calc_phase_transition)
        return phase_transition

class CriticalSlowingDownFactor(BaseFactor):
    """临界慢化因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('critical_slowing_down', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测临界慢化（系统恢复变慢）"""
        close = df['close']
        
        def calc_csd(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            n = len(returns)
            
            acf = []
            for k in range(1, min(self.lag * 5, n // 2)):
                if k >= n:
                    break
                corr = np.corrcoef(returns[k:], returns[:n-k])[0, 1] if n-k > 0 else 0
                acf.append(abs(corr))
            
            if not acf:
                return 0
            
            acf = np.array(acf)
            
            if len(acf) > 1:
                try:
                    decay_rate = -np.log(abs(acf[-1]) + 1e-8) / len(acf) if acf[-1] != 0 else 0
                    return decay_rate * 10
                except:
                    return 0
            return 0
        
        csd = close.rolling(window=self.window).apply(calc_csd)
        return csd

class MemoryFunctionFactor(BaseFactor):
    """记忆函数因子"""
    
    def __init__(self, window: int = 50, max_lag: int = 20):
        super().__init__('memory_function', window)
        self.max_lag = max_lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算记忆函数（长程相关性）"""
        close = df['close']
        
        def calc_memory(x):
            if len(x) < 100:
                return 0
            
            returns = np.diff(x)
            
            n = len(returns)
            acf = []
            
            for k in range(1, min(self.max_lag + 1, n // 2)):
                if k >= n:
                    break
                corr = np.corrcoef(returns[k:], returns[:n-k])[0, 1] if n-k > 0 else 0
                acf.append(corr)
            
            if not acf:
                return 0
            
            acf = np.array(acf)
            
            if len(acf) > 1:
                try:
                    decay_rate = -np.log(abs(acf[-1]) + 1e-8) / len(acf) if acf[-1] != 0 else 0
                    return decay_rate * 10
                except:
                    return 0
            return 0
        
        memory = close.rolling(window=self.window).apply(calc_memory)
        return memory

class NonlinearDampingFactor(BaseFactor):
    """非线性阻尼因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('nonlinear_damping', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性阻尼特征"""
        close = df['close']
        
        def calc_damping(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            n = len(returns)
            
            if n > 20:
                first_half = returns[:n//2]
                second_half = returns[n//2:]
                
                first_amplitude = np.std(first_half)
                second_amplitude = np.std(second_half)
                
                if first_amplitude > 0:
                    damping = np.log(first_amplitude / (second_amplitude + 1e-8)) / (n // 2)
                    return max(0, damping) * 10
            return 0
        
        damping = close.rolling(window=self.window).apply(calc_damping)
        return damping

class BifurcationParameterFactor(BaseFactor):
    """分岔参数因子"""
    
    def __init__(self, window: int = 50, param_range: float = 0.2):
        super().__init__('bifurcation_parameter', window)
        self.param_range = param_range
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """估计分岔参数（系统临界点）"""
        close = df['close']
        
        def calc_bifurcation_param(x):
            if len(x) < 50:
                return 0.5
            
            returns = np.diff(x)
            mean_return = np.mean(returns)
            
            r_values = np.linspace(mean_return - self.param_range, mean_return + self.param_range, 30)
            
            stat_values = []
            
            for r in r_values:
                x_val = 0.5
                states = []
                
                for _ in range(100):
                    x_val = r * x_val * (1 - x_val + 1e-8)
                    if _ > 50:
                        states.append(x_val)
                
                if states:
                    stat_values.append({
                        'r': r,
                        'mean': np.mean(states),
                        'var': np.var(states),
                        'max': np.max(states),
                        'min': np.min(states)
                    })
            
            if len(stat_values) < 10:
                return 0.5
            
            variances = [s['var'] for s in stat_values]
            
            if len(variances) > 5:
                var_changes = np.diff(variances)
                
                max_change_idx = np.argmax(np.abs(var_changes))
                
                normalized_param = max_change_idx / len(variances)
                return normalized_param
            
            return 0.5
        
        bifurcation_param = close.rolling(window=self.window).apply(calc_bifurcation_param)
        return bifurcation_param

class StrangeAttractorFactor(BaseFactor):
    """奇异吸引子因子"""
    
    def __init__(self, window: int = 50, dim: int = 3):
        super().__init__('strange_attractor', window)
        self.dim = dim
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测奇异吸引子特征"""
        close = df['close']
        
        def calc_strange_attractor(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            if n <= self.dim:
                return 0
                
            embedded = np.array([x[i:i+self.dim] for i in range(n - self.dim)])
            
            if len(embedded) < 100:
                return 0
            
            ranges = [np.max(embedded[:, i]) - np.min(embedded[:, i]) for i in range(self.dim)]
            volume = np.prod([max(r, 0.01) for r in ranges])
            
            point_density = len(embedded) / (volume + 1e-8)
            
            scales = [0.1, 0.2, 0.5, 1.0, 2.0]
            n_points = []
            
            for scale in scales:
                count = 0
                for i in range(len(embedded)):
                    for j in range(i + 1, len(embedded)):
                        if np.linalg.norm(embedded[i] - embedded[j]) < scale:
                            count += 1
                n_points.append(count)
            
            if len(n_points) > 2:
                try:
                    log_scales = np.log(scales)
                    log_points = np.log([max(p, 1) for p in n_points])
                    slope, _ = stats.linregress(log_scales, log_points)[:2]
                    fractal_dim = -slope
                except:
                    fractal_dim = self.dim
            else:
                fractal_dim = self.dim
            
            strange_attractor = point_density ** (1/self.dim) * (1 + abs(fractal_dim - self.dim))
            
            return min(strange_attractor, 10) / 10
        
        strange_attractor = close.rolling(window=self.window).apply(calc_strange_attractor)
        return strange_attractor

class TopologicalEntropyFactor(BaseFactor):
    """拓扑熵因子"""
    
    def __init__(self, window: int = 50, epsilon: float = 0.1):
        super().__init__('topological_entropy', window)
        self.epsilon = epsilon
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算拓扑熵（系统复杂度）"""
        close = df['close']
        
        def calc_topological_entropy(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            dim = min(3, n // 10)
            if dim < 2:
                return 0
                
            embedded = np.array([x[i:i+dim] for i in range(n - dim)])
            
            if len(embedded) < 50:
                return 0
            
            epsilon_values = [self.epsilon / 2, self.epsilon, self.epsilon * 2, self.epsilon * 4]
            n_values = []
            
            for eps in epsilon_values:
                covered = np.zeros(len(embedded), dtype=bool)
                count = 0
                
                for i in range(len(embedded)):
                    if not covered[i]:
                        for j in range(len(embedded)):
                            if not covered[j] and np.linalg.norm(embedded[i] - embedded[j]) < eps:
                                covered[j] = True
                        count += 1
                
                n_values.append(count)
            
            if len(n_values) < 3:
                return 0
            
            log_n = np.log([max(v, 1) for v in n_values])
            log_eps = np.log(epsilon_values)
            
            try:
                slope, _ = stats.linregress(log_eps, log_n)[:2]
                return max(0, -slope)
            except:
                return 0
        
        topological_entropy = close.rolling(window=self.window).apply(calc_topological_entropy)
        return topological_entropy

class WindingNumberFactor(BaseFactor):
    """缠绕数因子（环面映射）"""
    
    def __init__(self, window: int = 50):
        super().__init__('winding_number', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算缠绕数（周期轨道）"""
        close = df['close']
        
        def calc_winding(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            n = len(returns)
            
            theta = 2 * np.pi * returns / (np.max(returns) - np.min(returns) + 1e-8)
            r = np.linspace(0, 1, n)
            
            dtheta = np.diff(theta)
            dtheta = np.where(dtheta > np.pi, dtheta - 2*np.pi, dtheta)
            dtheta = np.where(dtheta < -np.pi, dtheta + 2*np.pi, dtheta)
            
            total_rotation = np.sum(dtheta)
            winding_number = total_rotation / (2 * np.pi)
            
            return abs(winding_number)
        
        winding = close.rolling(window=self.window).apply(calc_winding)
        return winding

class ManifoldDimensionFactor(BaseFactor):
    """流形维度因子"""
    
    def __init__(self, window: int = 50, k: int = 5):
        super().__init__('manifold_dimension', window)
        self.k = k
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """估计流形维度（局部线性嵌入）"""
        close = df['close']
        
        def calc_manifold_dim(x):
            if len(x) < 30:
                return 2.0
            
            n = len(x)
            
            dim = min(5, n // 5)
            if dim < 2:
                return 2.0
                
            embedded = np.array([x[i:i+dim] for i in range(n - dim)])
            
            if len(embedded) < 20:
                return 2.0
            
            errors = []
            
            for i in range(min(50, len(embedded))):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                nearest_idx = np.argsort(distances)[1:self.k+1]
                
                if len(nearest_idx) < 2:
                    continue
                
                try:
                    X = embedded[nearest_idx]
                    y = embedded[i]
                    
                    X_centered = X - np.mean(X, axis=0)
                    y_centered = y - np.mean(X, axis=0)
                    
                    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    
                    projection = U @ (U.T @ y_centered)
                    error = np.linalg.norm(y_centered - projection)
                    errors.append(error)
                except:
                    continue
            
            if not errors:
                return 2.0
            
            mean_error = np.mean(errors)
            
            dimension = 1 / (1 + mean_error)
            
            return dimension * self.k
        
        manifold_dim = close.rolling(window=self.window).apply(calc_manifold_dim)
        return manifold_dim

class FractalCorrelationFactor(BaseFactor):
    """分形相关性因子"""
    
    def __init__(self, window: int = 50, r: float = 0.1):
        super().__init__('fractal_correlation', window)
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算分形相关积分"""
        close = df['close']
        
        def calc_fractal_corr(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            dim = min(3, n // 10)
            if dim < 2:
                return 0
                
            embedded = np.array([x[i:i+dim] for i in range(n - dim)])
            
            if len(embedded) < 50:
                return 0
            
            count = 0
            total_pairs = 0
            
            for i in range(len(embedded)):
                for j in range(i + 1, len(embedded)):
                    if np.linalg.norm(embedded[i] - embedded[j]) < self.r:
                        count += 1
                    total_pairs += 1
            
            if total_pairs == 0:
                return 0
            
            correlation_integral = count / total_pairs
            
            if correlation_integral > 0:
                dimension = np.log(correlation_integral + 1e-8) / np.log(self.r)
                return max(0.1, -dimension)
            return 0
        
        fractal_corr = close.rolling(window=self.window).apply(calc_fractal_corr)
        return fractal_corr

class NonlinearPredictabilityFactor(BaseFactor):
    """非线性可预测性因子"""
    
    def __init__(self, window: int = 50, lag: int = 1):
        super().__init__('nonlinear_predictability', window)
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性可预测性"""
        close = df['close']
        
        def calc_predictability(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            n = len(returns)
            
            errors = []
            
            for h in range(1, min(self.lag + 3, n // 5)):
                y_true = returns[h:]
                y_pred = returns[:-h]
                
                if len(y_true) < 10:
                    continue
                
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                errors.append({
                    'lag': h,
                    'mae': mae,
                    'rmse': rmse
                })
            
            if not errors:
                return 0
            
            rmse_values = [e['rmse'] for e in errors]
            mae_values = [e['mae'] for e in errors]
            
            rmse_normalized = 1 - (np.min(rmse_values) / (np.mean(rmse_values) + 1e-8))
            mae_normalized = 1 - (np.min(mae_values) / (np.mean(mae_values) + 1e-8))
            
            predictability = (rmse_normalized + mae_normalized) / 2
            
            return predictability
        
        predictability = close.rolling(window=self.window).apply(calc_predictability)
        return predictability

class ChaosGameIterationFactor(BaseFactor):
    """混沌游戏迭代因子"""
    
    def __init__(self, window: int = 50, iterations: int = 100, r: float = 0.5):
        super().__init__('chaos_game_iteration', window)
        self.iterations = iterations
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌游戏迭代特征"""
        close = df['close']
        
        def calc_chaos_iter(x):
            if len(x) < 50:
                return 0
            
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            
            points = []
            x_val, y_val = 0.5, 0.5
            
            for i in range(self.iterations):
                vertex_idx = np.random.randint(0, 3)
                
                vertices = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
                vx, vy = vertices[vertex_idx]
                
                x_val = self.r * x_val + (1 - self.r) * vx
                y_val = self.r * y_val + (1 - self.r) * vy
                
                points.append((x_val, y_val))
            
            if len(points) < 10:
                return 0
            
            points = np.array(points)
            
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            
            if len(distances) > 10:
                early_dist = np.mean(distances[:10])
                late_dist = np.mean(distances[-10:])
                
                if early_dist > 0:
                    convergence = early_dist / (late_dist + 1e-8)
                else:
                    convergence = 1
            else:
                convergence = 1
            
            hist, _ = np.histogram2d(points[:, 0], points[:, 1], bins=5)
            prob = hist.flatten()
            prob = prob[prob > 0]
            
            if len(prob) > 0:
                uniformity = -np.sum(prob * np.log2(prob + 1e-10)) / np.log2(len(prob))
            else:
                uniformity = 0
            
            return convergence * uniformity
        
        chaos_iter = close.rolling(window=self.window).apply(calc_chaos_iter)
        return chaos_iter

class MultiscaleEntropyFactor(BaseFactor):
    """多尺度熵因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None, m: int = 2, r: float = 0.5):
        super().__init__('multiscale_entropy', window)
        self.scales = scales or [1, 2, 3, 4, 5]
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算多尺度熵"""
        close = df['close']
        
        def calc_multiscale_entropy(x):
            if len(x) < 100:
                return 0
            
            entropies = []
            
            for scale in self.scales:
                downsampled = x[::scale]
                
                if len(downsampled) < 20:
                    continue
                
                n = len(downsampled)
                
                if n <= self.m:
                    continue
                    
                embedded = np.array([downsampled[i:i+self.m] for i in range(n - self.m)])
                
                if len(embedded) < 10:
                    continue
                
                count_m = 0
                count_m1 = 0
                
                for i in range(len(embedded)):
                    for j in range(i + 1, len(embedded)):
                        if np.max(np.abs(embedded[i] - embedded[j])) < self.r:
                            count_m += 1
                            
                            if i + self.m < n and j + self.m < n:
                                if np.abs(downsampled[i+self.m] - downsampled[j+self.m]) < self.r:
                                    count_m1 += 1
                
                if count_m > 0:
                    entropy = -np.log(count_m1 / count_m + 1e-8)
                    entropies.append(entropy)
            
            return np.mean(entropies) if entropies else 0
        
        multiscale_entropy = close.rolling(window=self.window).apply(calc_multiscale_entropy)
        return multiscale_entropy

class RecurrenceQuantificationFactor(BaseFactor):
    """复发量化分析因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1, min_line: int = 2):
        super().__init__('recurrence_quantification', window)
        self.threshold = threshold
        self.min_line = min_line
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发量化特征"""
        close = df['close']
        
        def calc_rqa(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            recurrence = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(x[i] - x[j]) < self.threshold:
                        recurrence[i, j] = 1
                        recurrence[j, i] = 1
            
            total_pairs = n * (n - 1) / 2
            recurrence_points = np.sum(recurrence) / 2
            recurrence_rate = recurrence_points / total_pairs if total_pairs > 0 else 0
            
            diagonal_lengths = []
            for i in range(n):
                length = 0
                for j in range(1, min(10, n - i)):
                    if recurrence[i, i+j]:
                        length += 1
                    else:
                        break
                if length >= self.min_line:
                    diagonal_lengths.append(length)
            
            if diagonal_lengths:
                determinism = np.sum(diagonal_lengths) / recurrence_points if recurrence_points > 0 else 0
            else:
                determinism = 0
            
            avg_diagonal = np.mean(diagonal_lengths) if diagonal_lengths else 0
            
            max_diagonal = max(diagonal_lengths) if diagonal_lengths else 0
            
            rqa_index = recurrence_rate * (1 + determinism) * (1 + avg_diagonal / 10)
            
            return min(rqa_index, 1)
        
        rqa = close.rolling(window=self.window).apply(calc_rqa)
        return rqa

class DynamicTimeWarpingFactor(BaseFactor):
    """动态时间规整因子"""
    
    def __init__(self, window: int = 50, template_length: int = 10):
        super().__init__('dynamic_time_warping', window)
        self.template_length = template_length
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算DTW距离（模式匹配）"""
        close = df['close']
        
        def calc_dtw(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            if n < self.template_length * 2:
                return 0
            
            template = x[-self.template_length:]
            
            m = len(template)
            
            dtw = np.zeros((m + 1, n - m + 2))
            dtw[0, 1:] = np.inf
            dtw[1:, 0] = np.inf
            
            for i in range(1, m + 1):
                for j in range(1, n - m + 2):
                    cost = abs(template[i-1] - x[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
            
            dtw_distance = dtw[m, n - m + 1] / m
            
            predictability = 1 / (1 + dtw_distance)
            
            return predictability
        
        dtw = close.rolling(window=self.window).apply(calc_dtw)
        return dtw

class ManifoldLearningFactor(BaseFactor):
    """流形学习因子（t-SNE, UMAP等）"""
    
    def __init__(self, window: int = 50, n_components: int = 2):
        super().__init__('manifold_learning', window)
        self.n_components = n_components
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流形学习特征"""
        close = df['close']
        
        def calc_manifold(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            dim = min(5, n // 5)
            if dim < 2:
                return 0
                
            embedded = np.array([x[i:i+dim] for i in range(n - dim)])
            
            if len(embedded) < 30:
                return 0
            
            eigenvalues_list = []
            
            for i in range(min(50, len(embedded))):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                nearest_idx = np.argsort(distances)[1:6]
                
                if len(nearest_idx) < 2:
                    continue
                
                try:
                    X = embedded[nearest_idx]
                    X_centered = X - np.mean(X, axis=0)
                    
                    cov_matrix = np.cov(X_centered.T)
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    eigenvalues = np.sort(eigenvalues)[::-1]
                    
                    explained_variance = eigenvalues / np.sum(eigenvalues)
                    manifold_dim = np.sum(explained_variance ** 2)
                    
                    eigenvalues_list.append(1 / manifold_dim)
                except:
                    continue
            
            return np.mean(eigenvalues_list) if eigenvalues_list else 0
        
        manifold_learning = close.rolling(window=self.window).apply(calc_manifold)
        return manifold_learning

class RecurrenceAnalysisFactor(BaseFactor):
    """复发分析因子"""
    
    def __init__(self, window: int = 50, threshold: float = 0.1):
        super().__init__('recurrence_analysis', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算复发分析特征"""
        close = df['close']
        
        def calc_recurrence(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            recurrence = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(x[i] - x[j]) < self.threshold:
                        recurrence[i, j] = 1
                        recurrence[j, i] = 1
            
            diagonal_lengths = []
            for i in range(n):
                length = 0
                for j in range(1, min(10, n - i)):
                    if recurrence[i, i+j]:
                        length += 1
                    else:
                        break
                if length > 0:
                    diagonal_lengths.append(length)
            
            if not diagonal_lengths:
                return 0
            
            avg_length = np.mean(diagonal_lengths)
            max_length = max(diagonal_lengths)
            
            return avg_length * max_length / 100
        
        recurrence_analysis = close.rolling(window=self.window).apply(calc_recurrence)
        return recurrence_analysis

class NonlinearDynamicsFactor(BaseFactor):
    """非线性动力学因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.5):
        super().__init__('nonlinear_dynamics', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性动力学特征"""
        close = df['close']
        
        def calc_nonlinear_dynamics(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            if n <= self.m:
                return 0
                
            embedded = np.array([x[i:i+self.m] for i in range(n - self.m)])
            
            if len(embedded) < 50:
                return 0
            
            lyapunov_estimates = []
            
            for i in range(min(50, len(embedded))):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                nearest_idx = np.argsort(distances)[1]
                
                if nearest_idx < len(embedded):
                    separation = np.linalg.norm(embedded[nearest_idx] - embedded[i])
                    if separation > 0:
                        lyapunov_estimates.append(np.log(separation + 1e-8))
            
            if not lyapunov_estimates:
                return 0
            
            mean_lyapunov = np.mean(lyapunov_estimates)
            
            return max(0, mean_lyapunov)
        
        nonlinear_dynamics = close.rolling(window=self.window).apply(calc_nonlinear_dynamics)
        return nonlinear_dynamics

class FractalAnalysisFactor(BaseFactor):
    """分形分析因子"""
    
    def __init__(self, window: int = 50, scales: List[int] = None):
        super().__init__('fractal_analysis', window)
        self.scales = scales or [5, 10, 20, 40, 80]
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算分形分析特征"""
        close = df['close']
        
        def calc_fractal(x):
            if len(x) < 100:
                return 0
            
            log_price = np.log(x)
            
            variances = []
            for scale in self.scales:
                if scale >= len(log_price) // 2:
                    continue
                
                n_segments = len(log_price) // scale
                segment_vars = []
                
                for i in range(n_segments):
                    segment = log_price[i*scale:(i+1)*scale]
                    if len(segment) > 1:
                        segment_vars.append(np.var(segment))
                
                if segment_vars:
                    variances.append(np.mean(segment_vars))
            
            if len(variances) < 3:
                return 0
            
            try:
                log_scales = np.log(self.scales[:len(variances)])
                log_vars = np.log(variances)
                slope, _ = stats.linregress(log_scales, log_vars)[:2]
                fractal_dim = 0.5 * (1 - slope)
                return max(0.1, min(2.0, fractal_dim))
            except:
                return 0
        
        fractal_analysis = close.rolling(window=self.window).apply(calc_fractal)
        return fractal_analysis

class ChaosTheoryFactor(BaseFactor):
    """混沌理论因子"""
    
    def __init__(self, window: int = 50, m: int = 3, r: float = 0.1):
        super().__init__('chaos_theory', window)
        self.m = m
        self.r = r
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算混沌理论特征"""
        close = df['close']
        
        def calc_chaos(x):
            if len(x) < 50:
                return 0
            
            n = len(x)
            
            if n <= self.m:
                return 0
                
            embedded = np.array([x[i:i+self.m] for i in range(n - self.m)])
            
            if len(embedded) < 50:
                return 0
            
            # 计算关联维度
            distances = []
            for i in range(len(embedded)):
                for j in range(i + 1, len(embedded)):
                    distances.append(np.linalg.norm(embedded[i] - embedded[j]))
            
            if not distances:
                return 0
            
            count = sum(1 for d in distances if d < self.r)
            correlation_integral = count / len(distances)
            
            if correlation_integral > 0:
                dimension = np.log(correlation_integral + 1e-8) / np.log(self.r)
                return max(0.1, -dimension)
            return 0
        
        chaos_theory = close.rolling(window=self.window).apply(calc_chaos)
        return chaos_theory

class NonlinearTimeSeriesFactor(BaseFactor):
    """非线性时间序列因子"""
    
    def __init__(self, window: int = 50, max_lag: int = 10):
        super().__init__('nonlinear_time_series', window)
        self.max_lag = max_lag
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算非线性时间序列特征"""
        close = df['close']
        
        def calc_nonlinear_ts(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            # 计算高阶统计量
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # 非线性度量
            nonlinear_metric = abs(skewness) + abs(kurtosis - 3) / 6
            
            return min(nonlinear_metric, 5) / 5
        
        nonlinear_ts = close.rolling(window=self.window).apply(calc_nonlinear_ts)
        return nonlinear_ts

class StateSpaceReconstructionFactor(BaseFactor):
    """状态空间重构因子"""
    
    def __init__(self, window: int = 50, dim: int = 3, tau: int = 1):
        super().__init__('state_space_reconstruction', window)
        self.dim = dim
        self.tau = tau
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算状态空间重构特征"""
        close = df['close']
        
        def calc_state_space(x):
            if len(x) < 30:
                return 0
            
            n = len(x)
            
            if n <= self.dim:
                return 0