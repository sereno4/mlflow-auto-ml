#!/usr/bin/env python3
"""
Configuração do Evidently para drift detection.
Define thresholds e colunas a monitorar.
"""

from typing import List, Dict

# Colunas numéricas a monitorar para drift
NUMERIC_COLUMNS: List[str] = [
    'latency_ms',
    'memory_usage_mb',
    'cpu_percent',
    'grok_tokens_used',
    'tokens_per_ms',
    'memory_per_token',
]

# Colunas categóricas a monitorar
CATEGORICAL_COLUMNS: List[str] = [
    'has_error',
    'high_latency',
    'is_peak_hour',
    'is_weekend',
]

# Thresholds para drift
DRIFT_THRESHOLDS: Dict[str, float] = {
    'dataset_drift_ratio': 0.5,  # Se >50% das features driftaram → alerta
    'psi_threshold': 0.2,        # Population Stability Index
    'ks_threshold': 0.1,         # Kolmogorov-Smirnov test p-value
}

# Regras de sanity check
SANITY_CHECKS: Dict[str, Dict] = {
    'latency_ms': {'min': 0, 'max': 10000},
    'memory_usage_mb': {'min': 0, 'max': 10000},
    'cpu_percent': {'min': 0, 'max': 100},
    'grok_tokens_used': {'min': 0, 'max': 10000},
}
