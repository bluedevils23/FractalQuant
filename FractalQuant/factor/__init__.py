"""
因子模块
"""
from .base import BaseFactor
from .price import ReturnsFactor, PriceMomentumFactor, VolumePriceTrendFactor
from .volatility import HistoricalVolatilityFactor, ParkinsonVolatilityFactor, GARCHVolatilityFactor
from .trend import MACDFactor, RSIFactor, EMAFactor, BollingerBandsFactor
from .orderbook import OrderBookImbalanceFactor, OrderBookPressureFactor

# 高级统计因子
from .advanced import (
    FutureReturnsFactor, CorrelationFactor, HurstExponentFactor,
    LyapunovExponentFactor, RecurrenceRateFactor, EmbeddingDimensionFactor,
    CorrelationDimensionFactor, KolmogorovEntropyFactor, MultifractalSpectrumFactor,
    DetrendedFluctuationFactor, WaveletEntropyFactor, PhaseSpaceVolumeFactor,
    PoincareSectionFactor, BifurcationDiagramFactor, ChaosIndicatorFactor,
    TimeReversalAsymmetryFactor, NonlinearAutocorrelationFactor
)

# 机器学习因子
from .ml import (
    MLForecastFactor, MLAnomalyDetectionFactor, ClusteringRegimeFactor,
    DimensionReductionFactor, EnsemblePredictorFactor, NeuralNetPredictorFactor,
    SupportVectorForecastFactor, FeatureImportanceFactor, AutoencoderAnomalyFactor,
    GaussianProcessFactor, RegressionQuantileFactor
)

# 微观结构因子
from .microstructure import (
    OrderFlowImbalanceFactor, LiquidityRatioFactor, VolumeWeightedPriceFactor,
    OrderBookPressureFactor, TradeSizeDistributionFactor, VolatilityAdjustedVolumeFactor,
    PriceVelocityFactor, MomentumAccelerationFactor, VolumeSpikeFactor,
    LiquidityShockFactor, OrderBookAsymmetryFactor, TradeDirectionPersistenceFactor,
    MarketImpactFactor, LiquidityDepthFactor, OrderFlowSignificanceFactor,
    VolumeClusteringFactor, PriceVolumeDecouplingFactor, MarketEfficiencyFactor,
    LiquidityMigrationFactor
)

# 跨市场因子
from .crossmarket import (
    CrossMarketCorrelationFactor, ArbitrageOpportunityFactor, MarketLinkageFactor,
    RelativeStrengthFactor, CointegrationFactor, CrossMarketVolatilityFactor,
    MarketRegimeSwitchFactor, CrossMarketEntropyFactor, CrossMarketCoherenceFactor,
    CrossMarketGrangerFactor, CrossMarketJointDistributionFactor, CrossMarketCopulaFactor,
    CrossMarketPhaseSynchronizationFactor, CrossMarketInformationFlowFactor,
    CrossMarketMultiscaleCorrelationFactor, CrossMarketDynamicCorrelationFactor
)

# 组合器
from .combiner import FactorCombiner, FactorScore, MultiFactorSignal

# 选择器
from .selector import FactorSelector, WeightOptimizer, FactorEnsemble, FactorBacktest

__all__ = [
    # 基础
    'BaseFactor',
    
    # 价格因子
    'ReturnsFactor', 'PriceMomentumFactor', 'VolumePriceTrendFactor',
    
    # 波动率因子
    'HistoricalVolatilityFactor', 'ParkinsonVolatilityFactor', 'GARCHVolatilityFactor',
    
    # 趋势因子
    'MACDFactor', 'RSIFactor', 'EMAFactor', 'BollingerBandsFactor',
    
    # 订单簿因子
    'OrderBookImbalanceFactor', 'OrderBookPressureFactor',
    
    # 高级统计因子
    'FutureReturnsFactor', 'CorrelationFactor', 'HurstExponentFactor',
    'LyapunovExponentFactor', 'RecurrenceRateFactor', 'EmbeddingDimensionFactor',
    'CorrelationDimensionFactor', 'KolmogorovEntropyFactor', 'MultifractalSpectrumFactor',
    'DetrendedFluctuationFactor', 'WaveletEntropyFactor', 'PhaseSpaceVolumeFactor',
    'PoincareSectionFactor', 'BifurcationDiagramFactor', 'ChaosIndicatorFactor',
    'TimeReversalAsymmetryFactor', 'NonlinearAutocorrelationFactor',
    
    # 机器学习因子
    'MLForecastFactor', 'MLAnomalyDetectionFactor', 'ClusteringRegimeFactor',
    'DimensionReductionFactor', 'EnsemblePredictorFactor', 'NeuralNetPredictorFactor',
    'SupportVectorForecastFactor', 'FeatureImportanceFactor', 'AutoencoderAnomalyFactor',
    'GaussianProcessFactor', 'RegressionQuantileFactor',
    
    # 微观结构因子
    'OrderFlowImbalanceFactor', 'LiquidityRatioFactor', 'VolumeWeightedPriceFactor',
    'OrderBookPressureFactor', 'TradeSizeDistributionFactor', 'VolatilityAdjustedVolumeFactor',
    'PriceVelocityFactor', 'MomentumAccelerationFactor', 'VolumeSpikeFactor',
    'LiquidityShockFactor', 'OrderBookAsymmetryFactor', 'TradeDirectionPersistenceFactor',
    'MarketImpactFactor', 'LiquidityDepthFactor', 'OrderFlowSignificanceFactor',
    'VolumeClusteringFactor', 'PriceVolumeDecouplingFactor', 'MarketEfficiencyFactor',
    'LiquidityMigrationFactor',
    
    # 跨市场因子
    'CrossMarketCorrelationFactor', 'ArbitrageOpportunityFactor', 'MarketLinkageFactor',
    'RelativeStrengthFactor', 'CointegrationFactor', 'CrossMarketVolatilityFactor',
    'MarketRegimeSwitchFactor', 'CrossMarketEntropyFactor', 'CrossMarketCoherenceFactor',
    'CrossMarketGrangerFactor', 'CrossMarketJointDistributionFactor', 'CrossMarketCopulaFactor',
    'CrossMarketPhaseSynchronizationFactor', 'CrossMarketInformationFlowFactor',
    'CrossMarketMultiscaleCorrelationFactor', 'CrossMarketDynamicCorrelationFactor',
    
    # 组合器
    'FactorCombiner', 'FactorScore', 'MultiFactorSignal',
    
    # 选择器
    'FactorSelector', 'WeightOptimizer', 'FactorEnsemble', 'FactorBacktest',
]
