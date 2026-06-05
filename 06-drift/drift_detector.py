#!/usr/bin/env python3
"""
Drift Detector usando Evidently.
Compara dados atuais com baseline de referência.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Adicionar paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03-features'))
sys.path.insert(0, os.path.dirname(__file__))

from evidently_config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, DRIFT_THRESHOLDS

# Evidently imports
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️  Evidently não instalado. Usando fallback estatístico.")

from feature_engineering import get_features_from_db, engineer_features


class DriftDetector:
    """
    Detecta drift entre dados de referência e dados atuais.
    """
    
    def __init__(self, reference_path: Optional[str] = None):
        self.reference_path = reference_path or "/tmp/reference_data.csv"
        self.drift_report_path = "/tmp/drift_report.json"
    
    def _load_reference(self) -> pd.DataFrame:
        """
        Carrega dados de referência (baseline).
        Se não existir, cria a partir do DB.
        """
        if os.path.exists(self.reference_path):
            return pd.read_csv(self.reference_path)
        
        print("📊 Criando baseline de referência...")
        df = get_features_from_db("vw_features_v1")
        df = engineer_features(df)
        df.to_csv(self.reference_path, index=False)
        print(f"💾 Baseline salvo em {self.reference_path}")
        return df
    
    def _load_current(self) -> pd.DataFrame:
        """
        Carrega dados atuais do PostgreSQL.
        """
        df = get_features_from_db("vw_features_v1")
        df = engineer_features(df)
        return df
    
    def _fallback_drift_check(self, reference: pd.DataFrame, current: pd.DataFrame) -> Dict:
        """
        Fallback quando Evidently não está disponível.
        Usa testes estatísticos simples (KS + média).
        """
        from scipy import stats
        
        results = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_ratio': 0.0,
            'details': {}
        }
        
        numeric_cols = [c for c in NUMERIC_COLUMNS if c in reference.columns and c in current.columns]
        
        drifted = 0
        for col in numeric_cols:
            ref_data = reference[col].dropna()
            cur_data = current[col].dropna()
            
            if len(ref_data) < 2 or len(cur_data) < 2:
                continue
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)
            
            # Mean diff
            mean_diff = abs(ref_data.mean() - cur_data.mean()) / (ref_data.std() + 1e-10)
            
            is_drifted = p_value < DRIFT_THRESHOLDS['ks_threshold'] or mean_diff > 0.5
            
            results['details'][col] = {
                'ks_stat': float(ks_stat),
                'p_value': float(p_value),
                'mean_diff': float(mean_diff),
                'drifted': is_drifted
            }
            
            if is_drifted:
                drifted += 1
                results['drifted_features'].append(col)
        
        total = len(numeric_cols)
        results['drift_ratio'] = drifted / total if total > 0 else 0
        results['drift_detected'] = results['drift_ratio'] > DRIFT_THRESHOLDS['dataset_drift_ratio']
        
        return results
    
    def detect_drift(self) -> Dict:
        """
        Executa detecção de drift completa.
        Retorna dict com resultado e metadados.
        """
        print("🔍 Detectando drift...")
        
        reference = self._load_reference()
        current = self._load_current()
        
        print(f"📊 Referência: {len(reference)} amostras | Atual: {len(current)} amostras")
        
        if not EVIDENTLY_AVAILABLE:
            print("⚠️  Usando fallback estatístico (Evidently não instalado)")
            results = self._fallback_drift_check(reference, current)
        else:
            # Evidently report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            
            # Extrair resultados
            report_dict = report.as_dict()
            
            drifted_features = []
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'DataDriftTable':
                    for feature, data in metric.get('result', {}).get('drift_by_columns', {}).items():
                        if data.get('drift_detected'):
                            drifted_features.append(feature)
            
            total_features = len(report_dict.get('metrics', [{}])[0].get('result', {}).get('drift_by_columns', {}))
            drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0
            
            results = {
                'drift_detected': drift_ratio > DRIFT_THRESHOLDS['dataset_drift_ratio'],
                'drifted_features': drifted_features,
                'drift_ratio': drift_ratio,
                'details': report_dict
            }
            
            # Salvar report
            report.save_json(self.drift_report_path)
            print(f"📄 Report salvo em {self.drift_report_path}")
        
        # Print resumo
        status = "🚨 DRIFT DETECTADO" if results['drift_detected'] else "✅ Sem drift"
        print(f"{status} (ratio: {results['drift_ratio']:.2%})")
        
        if results['drifted_features']:
            print(f"   Features com drift: {results['drifted_features']}")
        
        return results
    
    def should_retrain(self) -> bool:
        """
        Decide se deve re-treinar baseado no drift.
        """
        results = self.detect_drift()
        return results['drift_detected']
    
    def update_reference(self):
        """
        Atualiza baseline de referência com dados atuais.
        """
        current = self._load_current()
        current.to_csv(self.reference_path, index=False)
        print(f"🔄 Baseline atualizado: {self.reference_path}")


def main():
    """
    CLI para drift detection.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Drift Detector')
    parser.add_argument('--update-baseline', action='store_true', help='Atualiza baseline')
    parser.add_argument('--check-only', action='store_true', help='Só verifica, não decide')
    args = parser.parse_args()
    
    detector = DriftDetector()
    
    if args.update_baseline:
        detector.update_reference()
        return
    
    results = detector.detect_drift()
    
    if not args.check_only:
        if results['drift_detected']:
            print("\n🚨 AÇÃO: Re-treinamento necessário!")
            sys.exit(1)  # Exit code 1 = drift detectado
        else:
            print("\n✅ AÇÃO: Nenhuma ação necessária")
            sys.exit(0)


if __name__ == "__main__":
    main()
