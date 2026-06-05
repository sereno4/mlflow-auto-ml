#!/usr/bin/env python3
"""
TESTE ETAPA 5: MLflow Registry
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    try:
        from registry_manager import RegistryManager, promote_pipeline
        from log_experiment import log_experiment
        print("✅ Imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import falhou: {e}")
        return False

def test_registry_manager_init():
    try:
        from registry_manager import RegistryManager
        manager = RegistryManager()
        print("✅ RegistryManager inicializado")
        return True
    except Exception as e:
        print(f"❌ Init falhou: {e}")
        return False

def test_list_versions():
    try:
        from registry_manager import RegistryManager
        manager = RegistryManager()
        versions = manager.list_versions()
        print(f"✅ List versions OK: {len(versions)} versões")
        return True
    except Exception as e:
        print(f"❌ List versions falhou: {e}")
        return False

def test_log_experiment():
    try:
        from log_experiment import log_experiment
        run_id = log_experiment(
            "test_run",
            {"param1": "value1"},
            {"metric1": 0.95}
        )
        assert run_id is not None, "Run ID não retornado"
        print(f"✅ Log experiment OK: run_id={run_id[:8]}...")
        return True
    except Exception as e:
        print(f"❌ Log experiment falhou: {e}")
        return False

def test_model_uri():
    try:
        from registry_manager import RegistryManager
        manager = RegistryManager()
        uri = manager.get_model_uri("Production")
        print(f"✅ Get model URI OK: {uri}")
        return True
    except Exception as e:
        print(f"❌ Get model URI falhou: {e}")
        return False

def test_mlflow_connection():
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow connection OK: {len(experiments)} experimentos")
        return True
    except Exception as e:
        print(f"❌ MLflow connection falhou: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 5: MLflow Registry")
    print("=" * 60)
    
    tests = [
        ("Imports", test_import),
        ("Registry Init", test_registry_manager_init),
        ("List Versions", test_list_versions),
        ("Log Experiment", test_log_experiment),
        ("Model URI", test_model_uri),
        ("MLflow Connection", test_mlflow_connection),
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
        print("🎉 ETAPA 5 COMPLETA! Pronto para Etapa 6.")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
