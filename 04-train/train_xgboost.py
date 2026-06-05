#!/usr/bin/env python3
"""
Treinamento XGBoost para ML Pipeline.
Classificação: prever se haverá erro no agente.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Configuração
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
EXPERIMENT_NAME = "mlflow-auto-ml"
FEATURES_PATH = "/tmp/features_ready.csv"

# Adicionar path do feature engineering
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03-features'))

def load_features(path: str = FEATURES_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega features preparadas. Se não existir, roda feature engineering.
    """
    if not os.path.exists(path):
        print("⚠️  CSV não encontrado. Rodando feature engineering...")
        from feature_engineering import run_feature_engineering
        df = run_feature_engineering()
    else:
        df = pd.read_csv(path)
    
    # Identificar colunas
    exclude_cols = ['event_id', 'agent_id', 'target', 'has_error']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # Garantir tipos numéricos
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.astype(int)
    
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
    """
    Split temporal (não aleatório para dados de séries temporais).
    """
    split_idx = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:]
    )

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Treina modelo XGBoost com hiperparâmetros otimizados para CPU.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Avalia modelo com múltiplas métricas.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5,
    }
    
    return metrics

def log_training_to_mlflow(
    model: xgb.XGBClassifier,
    metrics: Dict[str, float],
    feature_names: list,
    params: Dict[str, Any],
    run_name: str = "xgboost_training"
) -> str:
    """
    Loga treinamento completo no MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=run_name):
        # Log hiperparâmetros
        mlflow.log_params(params)
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log feature names
        feature_dict = {'features': feature_names}
        
        artifact_dir = os.path.expanduser("~/.mlflow_artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        
        feature_names_path = os.path.join(artifact_dir, "feature_names.json")
        with open(feature_names_path, 'w') as f:
            json.dump(feature_dict, f, indent=2)
        
        try:
            mlflow.log_artifact(feature_names_path, 'model')
        except Exception as e:
            print(f"⚠️  Warning: Não foi possível logar artifact: {e}")
        
        # Log modelo com assinatura
        try:
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name="agent-error-predictor"
            )
        except Exception as e:
            print(f"⚠️  Warning: Não foi possível registrar modelo: {e}")
            # Salvar localmente como fallback
            model.save_model(os.path.join(artifact_dir, "xgboost_model.json"))
            print(f"💾 Modelo salvo localmente em: {artifact_dir}/xgboost_model.json")
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Modelo logado no MLflow: run_id={run_id}")
        return run_id

def main():
    print("🚀 Treinamento XGBoost...")
    
    # 1. Carregar features (auto-extrai se necessário)
    X, y = load_features()
    print(f"📊 Features carregadas: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"📦 Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # 3. Treinar
    print("⚙️  Treinando modelo...")
    model = train_model(X_train, y_train)
    print("✅ Treinamento concluído")
    
    # 4. Avaliar
    print("📈 Avaliando modelo...")
    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # 5. Log no MLflow
    params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
    }
    
    run_id = log_training_to_mlflow(model, metrics, list(X.columns), params)
    
    # 6. Salvar localmente
    model.save_model('/tmp/xgboost_model.json')
    print("💾 Modelo salvo em /tmp/xgboost_model.json")
    
    return run_id, metrics

if __name__ == "__main__":
    main()
