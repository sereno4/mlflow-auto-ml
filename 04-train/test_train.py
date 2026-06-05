#!/usr/bin/env python3
"""
TESTE ETAPA 4: Treinamento XGBoost
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    try:
        from train_xgboost import (
            load_features, split_data, train_model,
            evaluate_model, log_training_to_mlflow
        )
        print("✅ Imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import falhou: {e}")
        return False

def test_load_features():
    try:
        from train_xgboost import load_features
        X, y = load_features()
        
        assert len(X) > 0, "DataFrame vazio"
        assert len(X) == len(y), "X e y com tamanhos diferentes"
        assert y.nunique() >= 1, "Target precisa ter pelo menos 1 classe"
        
        print(f"✅ Load features OK: {X.shape[0]} amostras, {X.shape[1]} features")
        return True
    except Exception as e:
        print(f"❌ Load features falhou: {e}")
        return False

def test_split_data():
    try:
        from train_xgboost import load_features, split_data
        X, y = load_features()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert len(X_train) > 0, "Treino vazio"
        assert len(X_test) > 0, "Teste vazio"
        assert len(X_train) + len(X_test) == len(X), "Split incorreto"
        
        print(f"✅ Split OK: train={len(X_train)}, test={len(X_test)}")
        return True
    except Exception as e:
        print(f"❌ Split falhou: {e}")
        return False

def test_train_model():
    try:
        from train_xgboost import load_features, split_data, train_model
        X, y = load_features()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        model = train_model(X_train, y_train)
        
        assert hasattr(model, 'predict'), "Modelo não tem método predict"
        assert hasattr(model, 'predict_proba'), "Modelo não tem predict_proba"
        
        print("✅ Treinamento OK: modelo treinado")
        return True
    except Exception as e:
        print(f"❌ Treinamento falhou: {e}")
        return False

def test_evaluate_model():
    try:
        from train_xgboost import load_features, split_data, train_model, evaluate_model
        X, y = load_features()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for m in required_metrics:
            assert m in metrics, f"Métrica {m} faltando"
            assert 0 <= metrics[m] <= 1, f"Métrica {m} fora do range [0,1]"
        
        print(f"✅ Avaliação OK: accuracy={metrics['accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Avaliação falhou: {e}")
        return False

def test_mlflow_logging():
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        experiments = mlflow.search_experiments()
        
        # Verificar se experimento existe
        exp_names = [e.name for e in experiments]
        assert "mlflow-auto-ml" in exp_names, "Experimento não encontrado"
        
        print(f"✅ MLflow OK: {len(experiments)} experimentos")
        return True
    except Exception as e:
        print(f"❌ MLflow falhou: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 4: Treinamento XGBoost")
    print("=" * 60)
    
    tests = [
        ("Imports", test_import),
        ("Load Features", test_load_features),
        ("Split Data", test_split_data),
        ("Train Model", test_train_model),
        ("Evaluate Model", test_evaluate_model),
        ("MLflow Logging", test_mlflow_logging),
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
        print("🎉 ETAPA 4 COMPLETA! Pronto para Etapa 5.")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
