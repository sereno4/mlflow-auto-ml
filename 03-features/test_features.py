#!/usr/bin/env python3
"""
TESTE ETAPA 3: Feature Engineering + MLflow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    try:
        from feature_engineering import (
            get_features_from_db,
            engineer_features,
            compute_feature_hash,
            log_features_to_mlflow,
            prepare_train_test
        )
        print("✅ Imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import falhou: {e}")
        return False

def test_extract_features():
    try:
        from feature_engineering import get_features_from_db
        df = get_features_from_db("vw_features_v1")
        assert len(df) > 0, "DataFrame vazio"
        assert 'latency_ms' in df.columns, "Coluna latency_ms faltando"
        print(f"✅ Extração OK: {len(df)} registros, {len(df.columns)} colunas")
        return True
    except Exception as e:
        print(f"❌ Extração falhou: {e}")
        return False

def test_engineer_features():
    try:
        from feature_engineering import get_features_from_db, engineer_features
        df = get_features_from_db("vw_features_v1")
        df = engineer_features(df)
        
        assert 'tokens_per_ms' in df.columns, "Feature tokens_per_ms faltando"
        assert 'is_peak_hour' in df.columns, "Feature is_peak_hour faltando"
        assert 'target' in df.columns, "Target faltando"
        
        print(f"✅ Engenharia OK: {len(df.columns)} features")
        return True
    except Exception as e:
        print(f"❌ Engenharia falhou: {e}")
        return False

def test_mlflow_logging():
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Verificar se consegue listar experimentos
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow OK: {len(experiments)} experimentos encontrados")
        return True
    except Exception as e:
        print(f"❌ MLflow falhou: {e}")
        return False

def test_feature_hash():
    try:
        from feature_engineering import get_features_from_db, engineer_features, compute_feature_hash
        df = get_features_from_db("vw_features_v1")
        df = engineer_features(df)
        h = compute_feature_hash(df)
        
        assert len(h) == 16, "Hash deve ter 16 caracteres"
        print(f"✅ Hash OK: {h}")
        return True
    except Exception as e:
        print(f"❌ Hash falhou: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 3: Feature Engineering + MLflow")
    print("=" * 60)
    
    tests = [
        ("Imports", test_import),
        ("Extração", test_extract_features),
        ("Engenharia", test_engineer_features),
        ("MLflow", test_mlflow_logging),
        ("Hash", test_feature_hash),
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
        print("🎉 ETAPA 3 COMPLETA! Pronto para Etapa 4.")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
