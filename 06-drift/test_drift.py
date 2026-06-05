#!/usr/bin/env python3
"""
TESTE ETAPA 6: Drift Detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    try:
        from drift_detector import DriftDetector
        from evidently_config import NUMERIC_COLUMNS, DRIFT_THRESHOLDS
        print("✅ Imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import falhou: {e}")
        return False

def test_config():
    try:
        from evidently_config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, DRIFT_THRESHOLDS
        assert len(NUMERIC_COLUMNS) > 0, "NUMERIC_COLUMNS vazio"
        assert 'dataset_drift_ratio' in DRIFT_THRESHOLDS, "Threshold de drift faltando"
        print(f"✅ Config OK: {len(NUMERIC_COLUMNS)} num, {len(CATEGORICAL_COLUMNS)} cat")
        return True
    except Exception as e:
        print(f"❌ Config falhou: {e}")
        return False

def test_detector_init():
    try:
        from drift_detector import DriftDetector
        detector = DriftDetector()
        print("✅ DriftDetector inicializado")
        return True
    except Exception as e:
        print(f"❌ Init falhou: {e}")
        return False

def test_load_reference():
    try:
        from drift_detector import DriftDetector
        detector = DriftDetector()
        ref = detector._load_reference()
        assert len(ref) > 0, "Referência vazia"
        print(f"✅ Load reference OK: {len(ref)} amostras")
        return True
    except Exception as e:
        print(f"❌ Load reference falhou: {e}")
        return False

def test_detect_drift():
    try:
        from drift_detector import DriftDetector
        detector = DriftDetector()
        results = detector.detect_drift()
        
        assert 'drift_detected' in results, "Resultado sem flag drift_detected"
        assert 'drift_ratio' in results, "Resultado sem drift_ratio"
        
        status = "DRIFT" if results['drift_detected'] else "OK"
        print(f"✅ Detect drift OK: {status} (ratio={results['drift_ratio']:.2%})")
        return True
    except Exception as e:
        print(f"❌ Detect drift falhou: {e}")
        return False

def test_should_retrain():
    try:
        from drift_detector import DriftDetector
        detector = DriftDetector()
        should = detector.should_retrain()
        print(f"✅ Should retrain OK: {should}")
        return True
    except Exception as e:
        print(f"❌ Should retrain falhou: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 6: Drift Detection")
    print("=" * 60)
    
    tests = [
        ("Imports", test_import),
        ("Config", test_config),
        ("Detector Init", test_detector_init),
        ("Load Reference", test_load_reference),
        ("Detect Drift", test_detect_drift),
        ("Should Retrain", test_should_retrain),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- Testando: {name} ---")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed}/{len(tests)} testes passaram")
    print("=" * 60)
    
    if passed == len(tests):
        print("🎉 ETAPA 6 COMPLETA! Pronto para Etapa 7.")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
