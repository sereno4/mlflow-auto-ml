#!/usr/bin/env python3
"""
Sanity Checks (Regra de Ouro).
5 checks essenciais antes de promover modelo para Production.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03-features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '04-train'))

from feature_engineering import get_features_from_db, engineer_features
from train_xgboost import train_model, evaluate_model, split_data


class NumpyEncoder(json.JSONEncoder):
    """Encoder para serializar numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SanityChecker:
    """
    5 Checks da Regra de Ouro:
    1. Não constante (predições variam)
    2. Não nulo (sem NaN/None)
    3. Classe balanceada (não 100% uma classe)
    4. Métrica mínima (accuracy >= threshold)
    5. Latência aceitável (< 30s treino, < 10ms predição)
    """
    
    THRESHOLDS = {
        'min_accuracy': 0.6,
        'min_f1': 0.5,
        'max_train_time': 30.0,
        'max_predict_time': 0.01,
        'max_null_ratio': 0.05,
        'min_class_ratio': 0.1,
    }
    
    def __init__(self):
        self.results = {}
        self.passed = False
    
    def check_1_not_constant(self, model, X_test: pd.DataFrame) -> bool:
        """Check 1: Predições não são constantes."""
        print("\n🧪 Check 1: Not Constant")
        preds = model.predict(X_test)
        unique_preds = np.unique(preds)
        
        is_constant = len(unique_preds) == 1
        self.results['not_constant'] = {
            'passed': bool(not is_constant),
            'unique_predictions': int(len(unique_preds)),
            'values': [int(v) for v in unique_preds]
        }
        
        status = "✅ PASS" if not is_constant else "❌ FAIL"
        print(f"   {status}: {len(unique_preds)} classes únicas")
        return not is_constant
    
    def check_2_not_null(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Check 2: Sem valores nulos."""
        print("\n🧪 Check 2: Not Null")
        null_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        y_null = y.isnull().sum() / len(y)
        
        passed = null_ratio <= self.THRESHOLDS['max_null_ratio'] and y_null == 0
        
        self.results['not_null'] = {
            'passed': bool(passed),
            'X_null_ratio': float(null_ratio),
            'y_null_ratio': float(y_null)
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: X_null={null_ratio:.4%}, y_null={y_null:.4%}")
        return passed
    
    def check_3_class_balance(self, y: pd.Series) -> bool:
        """Check 3: Classes balanceadas."""
        print("\n🧪 Check 3: Class Balance")
        class_counts = y.value_counts(normalize=True)
        min_ratio = float(class_counts.min())
        
        passed = min_ratio >= self.THRESHOLDS['min_class_ratio']
        
        self.results['class_balance'] = {
            'passed': bool(passed),
            'class_distribution': {str(k): float(v) for k, v in class_counts.items()},
            'min_ratio': min_ratio
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: min_ratio={min_ratio:.2%}")
        return passed
    
    def check_4_metric_threshold(self, metrics: Dict[str, float]) -> bool:
        """Check 4: Métricas acima do threshold."""
        print("\n🧪 Check 4: Metric Threshold")
        acc = float(metrics.get('accuracy', 0))
        f1 = float(metrics.get('f1_score', 0))
        
        acc_ok = acc >= self.THRESHOLDS['min_accuracy']
        f1_ok = f1 >= self.THRESHOLDS['min_f1']
        passed = acc_ok and f1_ok
        
        self.results['metric_threshold'] = {
            'passed': bool(passed),
            'accuracy': acc,
            'f1_score': f1,
            'threshold_accuracy': self.THRESHOLDS['min_accuracy'],
            'threshold_f1': self.THRESHOLDS['min_f1']
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: acc={acc:.4f} (min {self.THRESHOLDS['min_accuracy']}), f1={f1:.4f} (min {self.THRESHOLDS['min_f1']})")
        return passed
    
    def check_5_latency(self, train_time: float, predict_time: float) -> bool:
        """Check 5: Latência aceitável."""
        print("\n🧪 Check 5: Latency")
        train_ok = train_time <= self.THRESHOLDS['max_train_time']
        predict_ok = predict_time <= self.THRESHOLDS['max_predict_time']
        passed = train_ok and predict_ok
        
        self.results['latency'] = {
            'passed': bool(passed),
            'train_time': float(train_time),
            'predict_time': float(predict_time),
            'max_train': self.THRESHOLDS['max_train_time'],
            'max_predict': self.THRESHOLDS['max_predict_time']
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: train={train_time:.3f}s (max {self.THRESHOLDS['max_train_time']}s), predict={predict_time*1000:.2f}ms (max {self.THRESHOLDS['max_predict_time']*1000:.0f}ms)")
        return passed
    
    def run_all_checks(self, model, X_train, X_test, y_train, y_test, metrics, train_time, predict_time) -> bool:
        """Executa todos os 5 checks."""
        print("=" * 60)
        print("🧪 SANITY CHECKS: Regra de Ouro (5 Checks)")
        print("=" * 60)
        
        checks = [
            self.check_1_not_constant(model, X_test),
            self.check_2_not_null(X_train, y_train),
            self.check_3_class_balance(y_train),
            self.check_4_metric_threshold(metrics),
            self.check_5_latency(train_time, predict_time),
        ]
        
        self.passed = all(checks)
        
        print("\n" + "=" * 60)
        if self.passed:
            print("✅ TODOS OS CHECKS PASSARAM! Modelo apto para Production.")
        else:
            print(f"❌ {sum(not c for c in checks)}/5 checks falharam.")
        print("=" * 60)
        
        return self.passed
    
    def get_report(self) -> Dict:
        """Retorna relatório completo."""
        return {
            'passed': bool(self.passed),
            'checks': self.results,
            'timestamp': pd.Timestamp.now().isoformat()
        }


def run_sanity_pipeline() -> Tuple[bool, Dict]:
    """Pipeline completo de sanity check."""
    import time
    
    print("🚀 Iniciando Sanity Pipeline...")
    
    df = get_features_from_db("vw_features_v1")
    df = engineer_features(df)
    
    feature_cols = [c for c in df.columns if c not in ['event_id', 'agent_id', 'target', 'has_error']]
    X = df[feature_cols]
    y = df['target']
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    t0 = time.time()
    model = train_model(X_train, y_train)
    train_time = time.time() - t0
    
    t0 = time.time()
    _ = model.predict(X_test.iloc[:1])
    predict_time = time.time() - t0
    
    metrics = evaluate_model(model, X_test, y_test)
    
    checker = SanityChecker()
    passed = checker.run_all_checks(model, X_train, X_test, y_train, y_test, metrics, train_time, predict_time)
    
    return passed, checker.get_report()


if __name__ == "__main__":
    passed, report = run_sanity_pipeline()
    
    report_path = os.path.expanduser("~/.mlflow_artifacts/sanity_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\n📄 Relatório salvo em: {report_path}")
    
    sys.exit(0 if passed else 1)
