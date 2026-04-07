"""
机器学习因子（预测模型、聚类、降维等）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, FastICA
from sklearn.covariance import EllipticEnvelope
from .base import BaseFactor

class MLForecastFactor(BaseFactor):
    """机器学习预测因子"""
    
    def __init__(self, window: int = 50, model_type: str = 'linear', forecast_steps: int = 5):
        super().__init__('ml_forecast', window)
        self.model_type = model_type
        self.forecast_steps = forecast_steps
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用机器学习模型预测未来价格"""
        close = df['close']
        
        def predict_price(x):
            if len(x) < self.window:
                return 0
            
            returns = np.diff(x)
            X = np.arange(len(returns)).reshape(-1, 1)
            
            try:
                if self.model_type == 'linear':
                    model = LinearRegression()
                elif self.model_type == 'ridge':
                    model = Ridge(alpha=1.0)
                elif self.model_type == 'lasso':
                    model = Lasso(alpha=1.0)
                elif self.model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=10, random_state=42)
                elif self.model_type == 'gb':
                    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
                else:
                    model = LinearRegression()
                
                model.fit(X[:-self.forecast_steps], returns[:-self.forecast_steps])
                prediction = model.predict(X[-self.forecast_steps:])
                
                current_price = x[-1]
                predicted_price = current_price + np.sum(prediction)
                
                return (predicted_price - current_price) / (current_price + 1e-8)
            except:
                return 0
        
        forecast = close.rolling(window=self.window).apply(predict_price)
        return forecast

class MLAnomalyDetectionFactor(BaseFactor):
    """机器学习异常检测因子"""
    
    def __init__(self, window: int = 50, contamination: float = 0.1):
        super().__init__('ml_anomaly', window)
        self.contamination = contamination
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用异常检测算法识别市场异常"""
        close = df['close']
        
        def detect_anomaly(x):
            if len(x) < 30:
                return 0
            
            returns = np.diff(x).reshape(-1, 1)
            
            try:
                scaler = StandardScaler()
                returns_scaled = scaler.fit_transform(returns)
                
                if len(returns_scaled) < 20:
                    return 0
                
                detector = EllipticEnvelope(contamination=self.contamination, random_state=42)
                detector.fit(returns_scaled)
                
                predictions = detector.predict(returns_scaled)
                anomaly_scores = -detector.score_samples(returns_scaled)
                
                current_score = anomaly_scores[-1]
                mean_score = np.mean(anomaly_scores[:-1])
                
                return (current_score - mean_score) / (np.std(anomaly_scores[:-1]) + 1e-8)
            except:
                return 0
        
        anomaly = close.rolling(window=self.window).apply(detect_anomaly)
        return anomaly

class ClusteringRegimeFactor(BaseFactor):
    """聚类市场状态因子"""
    
    def __init__(self, window: int = 100, n_clusters: int = 3):
        super().__init__('clustering_regime', window)
        self.n_clusters = n_clusters
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用聚类算法识别市场状态"""
        close = df['close']
        
        def cluster_regime(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            features = []
            for i in range(len(returns) - 10):
                window_returns = returns[i:i+10]
                if len(window_returns) < 10:
                    continue
                
                feature = [
                    np.mean(window_returns),
                    np.std(window_returns),
                    np.max(window_returns),
                    np.min(window_returns),
                    np.sum(window_returns > 0),
                    np.sum(window_returns < 0),
                    np.argmax(window_returns),
                    np.argmin(window_returns),
                    np.percentile(window_returns, 25),
                    np.percentile(window_returns, 75)
                ]
                features.append(feature)
            
            if len(features) < self.n_clusters * 5:
                return 0
            
            try:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_scaled)
                
                current_regime = labels[-1]
                regime_distribution = np.bincount(labels, minlength=self.n_clusters) / len(labels)
                
                regime_entropy = -np.sum(regime_distribution * np.log2(regime_distribution + 1e-10))
                
                return regime_entropy
            except:
                return 0
        
        regime = close.rolling(window=self.window).apply(cluster_regime)
        return regime

class DimensionReductionFactor(BaseFactor):
    """降维因子（PCA/ICA）"""
    
    def __init__(self, window: int = 50, n_components: int = 2, method: str = 'pca'):
        super().__init__('dimension_reduction', window)
        self.n_components = n_components
        self.method = method
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用降维技术提取主要成分"""
        close = df['close']
        
        def reduce_dimension(x):
            if len(x) < 30:
                return 0
            
            returns = np.diff(x)
            
            features = []
            for i in range(len(returns) - 5):
                window_returns = returns[i:i+5]
                if len(window_returns) < 5:
                    continue
                features.append(window_returns)
            
            if len(features) < 10:
                return 0
            
            try:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                if self.method == 'pca':
                    reducer = PCA(n_components=self.n_components, random_state=42)
                else:
                    reducer = FastICA(n_components=self.n_components, random_state=42)
                
                reduced = reducer.fit_transform(features_scaled)
                
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                
                return explained_variance
            except:
                return 0
        
        reduction = close.rolling(window=self.window).apply(reduce_dimension)
        return reduction

class EnsemblePredictorFactor(BaseFactor):
    """集成预测因子"""
    
    def __init__(self, window: int = 50, forecast_steps: int = 5):
        super().__init__('ensemble_predictor', window)
        self.forecast_steps = forecast_steps
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """集成多个模型的预测结果"""
        close = df['close']
        
        def ensemble_predict(x):
            if len(x) < self.window:
                return 0
            
            returns = np.diff(x)
            X = np.arange(len(returns)).reshape(-1, 1)
            
            predictions = []
            
            try:
                models = [
                    LinearRegression(),
                    Ridge(alpha=1.0),
                    Lasso(alpha=1.0)
                ]
                
                for model in models:
                    try:
                        model.fit(X[:-self.forecast_steps], returns[:-self.forecast_steps])
                        pred = model.predict(X[-self.forecast_steps:])
                        predictions.append(np.mean(pred))
                    except:
                        continue
                
                if not predictions:
                    return 0
                
                ensemble_pred = np.mean(predictions)
                current_price = x[-1]
                
                return ensemble_pred / (np.std(returns) + 1e-8)
            except:
                return 0
        
        ensemble = close.rolling(window=self.window).apply(ensemble_predict)
        return ensemble

class NeuralNetPredictorFactor(BaseFactor):
    """神经网络预测因子（简化实现）"""
    
    def __init__(self, window: int = 50, hidden_size: int = 10, epochs: int = 50):
        super().__init__('neural_net_predictor', window)
        self.hidden_size = hidden_size
        self.epochs = epochs
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用简单神经网络预测"""
        close = df['close']
        
        def neural_predict(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            try:
                X = np.arange(len(returns)).reshape(-1, 1)
                y = returns
                
                X_train, y_train = X[:-5], y[:-5]
                
                weights = np.random.randn(X_train.shape[1], self.hidden_size) * 0.1
                bias = np.zeros(self.hidden_size)
                output_weights = np.random.randn(self.hidden_size, 1) * 0.1
                
                for _ in range(self.epochs):
                    hidden = np.tanh(X_train @ weights + bias)
                    output = hidden @ output_weights
                    
                    error = y_train.reshape(-1, 1) - output
                    output_weights += 0.01 * hidden.T @ error
                    hidden_error = error @ output_weights.T * (1 - hidden ** 2)
                    weights += 0.01 * X_train.T @ hidden_error
                    bias += 0.01 * np.sum(hidden_error, axis=0)
                
                hidden_final = np.tanh(X[-1] @ weights + bias)
                prediction = hidden_final @ output_weights
                
                current_price = x[-1]
                predicted_price = current_price + prediction[0]
                
                return (predicted_price - current_price) / (current_price + 1e-8)
            except:
                return 0
        
        neural = close.rolling(window=self.window).apply(neural_predict)
        return neural

class SupportVectorForecastFactor(BaseFactor):
    """支持向量预测因子"""
    
    def __init__(self, window: int = 50, forecast_steps: int = 5, kernel: str = 'rbf'):
        super().__init__('sv_forecast', window)
        self.forecast_steps = forecast_steps
        self.kernel = kernel
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用支持向量机预测"""
        close = df['close']
        
        def sv_predict(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            X = np.arange(len(returns)).reshape(-1, 1)
            
            try:
                model = SVR(kernel=self.kernel, C=1.0, epsilon=0.1)
                model.fit(X[:-self.forecast_steps], returns[:-self.forecast_steps])
                
                prediction = model.predict(X[-self.forecast_steps:])
                
                current_price = x[-1]
                predicted_price = current_price + np.sum(prediction)
                
                return (predicted_price - current_price) / (current_price + 1e-8)
            except:
                return 0
        
        sv = close.rolling(window=self.window).apply(sv_predict)
        return sv

class FeatureImportanceFactor(BaseFactor):
    """特征重要性因子"""
    
    def __init__(self, window: int = 50, n_estimators: int = 100):
        super().__init__('feature_importance', window)
        self.n_estimators = n_estimators
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算特征重要性作为因子"""
        close = df['close']
        
        def calc_importance(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            features = []
            for i in range(len(returns) - 10):
                window_returns = returns[i:i+10]
                if len(window_returns) < 10:
                    continue
                features.append(window_returns)
            
            if len(features) < 20:
                return 0
            
            try:
                X = np.array(features[:-1])
                y = np.array([returns[i+10] for i in range(len(returns) - 10)])
                
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
                model.fit(X, y)
                
                importance = model.feature_importances_
                
                return np.max(importance)
            except:
                return 0
        
        importance = close.rolling(window=self.window).apply(calc_importance)
        return importance

class AutoencoderAnomalyFactor(BaseFactor):
    """自编码器异常检测因子"""
    
    def __init__(self, window: int = 50, encoding_dim: int = 3, epochs: int = 50):
        super().__init__('autoencoder_anomaly', window)
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用自编码器检测异常"""
        close = df['close']
        
        def autoencoder_anomaly(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            features = []
            for i in range(len(returns) - 10):
                window_returns = returns[i:i+10]
                if len(window_returns) < 10:
                    continue
                features.append(window_returns)
            
            if len(features) < 20:
                return 0
            
            try:
                X = np.array(features)
                
                input_dim = X.shape[1]
                encoding_dim = min(self.encoding_dim, input_dim - 1)
                
                weights_input = np.random.randn(input_dim, encoding_dim) * 0.1
                bias_encoder = np.zeros(encoding_dim)
                weights_decoder = np.random.randn(encoding_dim, input_dim) * 0.1
                bias_decoder = np.zeros(input_dim)
                
                for _ in range(self.epochs):
                    encoded = np.tanh(X @ weights_input + bias_encoder)
                    decoded = encoded @ weights_decoder + bias_decoder
                    
                    error = X - decoded
                    weights_decoder += 0.01 * encoded.T @ error
                    bias_decoder += 0.01 * np.sum(error, axis=0)
                    encoded_error = error @ weights_decoder.T * (1 - encoded ** 2)
                    weights_input += 0.01 * X.T @ encoded_error
                    bias_encoder += 0.01 * np.sum(encoded_error, axis=0)
                
                reconstructed = np.tanh(X @ weights_input + bias_encoder) @ weights_decoder + bias_decoder
                reconstruction_error = np.mean((X - reconstructed) ** 2, axis=1)
                
                current_error = reconstruction_error[-1]
                mean_error = np.mean(reconstruction_error[:-1])
                
                return (current_error - mean_error) / (np.std(reconstruction_error[:-1]) + 1e-8)
            except:
                return 0
        
        anomaly = close.rolling(window=self.window).apply(autoencoder_anomaly)
        return anomaly

class GaussianProcessFactor(BaseFactor):
    """高斯过程预测因子"""
    
    def __init__(self, window: int = 50, forecast_steps: int = 5):
        super().__init__('gaussian_process', window)
        self.forecast_steps = forecast_steps
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用高斯过程预测"""
        close = df['close']
        
        def gp_predict(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            X = np.arange(len(returns)).reshape(-1, 1)
            
            try:
                X_train, y_train = X[:-self.forecast_steps], returns[:-self.forecast_steps]
                
                kernel_var = np.var(y_train)
                length_scale = 5.0
                
                K = kernel_var * np.exp(-0.5 * np.linalg.norm(X_train[:, np.newaxis] - X_train[np.newaxis, :], axis=2) ** 2 / length_scale ** 2)
                K += 0.1 * np.eye(len(X_train))
                
                K_star = kernel_var * np.exp(-0.5 * np.linalg.norm(X[-self.forecast_steps:, np.newaxis] - X_train[np.newaxis, :], axis=2) ** 2 / length_scale ** 2)
                
                try:
                    alpha = np.linalg.solve(K, y_train)
                    prediction = K_star @ alpha
                except:
                    prediction = np.zeros(self.forecast_steps)
                
                current_price = x[-1]
                predicted_price = current_price + np.sum(prediction)
                
                return (predicted_price - current_price) / (current_price + 1e-8)
            except:
                return 0
        
        gp = close.rolling(window=self.window).apply(gp_predict)
        return gp

class RegressionQuantileFactor(BaseFactor):
    """分位数回归因子"""
    
    def __init__(self, window: int = 50, quantile: float = 0.5):
        super().__init__('quantile_regression', window)
        self.quantile = quantile
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """使用分位数回归预测"""
        close = df['close']
        
        def quantile_reg(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            X = np.arange(len(returns)).reshape(-1, 1)
            
            try:
                n = len(X)
                beta = np.zeros(2)
                
                for _ in range(100):
                    predictions = X @ beta
                    residuals = returns - predictions
                    
                    weights = np.where(residuals > 0, self.quantile, 1 - self.quantile)
                    
                    X_weighted = X * weights.reshape(-1, 1)
                    y_weighted = returns * weights
                    
                    try:
                        beta = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
                    except:
                        break
                
                future_X = np.array([[len(X), 1]])
                prediction = future_X @ beta
                
                current_price = x[-1]
                predicted_price = current_price + prediction[0]
                
                return (predicted_price - current_price) / (current_price + 1e-8)
            except:
                return 0
        
        quantile = close.rolling(window=self.window).apply(quantile_reg)
        return quantile
