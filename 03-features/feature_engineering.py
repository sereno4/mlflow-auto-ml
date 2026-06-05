#!/usr/bin/env python3
"""
Feature Engineering para ML Pipeline.
Extrai features do PostgreSQL e prepara para treino.
"""

import os
import hashlib
import json
import warnings
import pandas as pd
from typing import Tuple, Dict
from sqlalchemy import create_engine
import mlflow

# Configuração
DB_URL = "postgresql://mlflow:mlflow123@postgres:5432/mlflow_db"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
FEATURES_PATH = "/tmp/features_ready.csv"

# Criar engine SQLAlchemy (elimina UserWarning)
engine = create_engine(DB_URL)

def get_features_from_db(view_name: str = "vw_features_v1") -> pd.DataFrame:
    """
    Extrai features do PostgreSQL via SQLAlchemy.
    """
    query = f"""
    SELECT 
        event_id,
        agent_id,
        latency_ms,
        memory_usage_mb,
        cpu_percent,
        grok_tokens_used,
        has_error,
        high_latency,
        hour_of_day,
        day_of_week,
        user_action
    FROM feature_store.{view_name}
    WHERE latency_ms IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de features adicionais.
    """
    df = df.copy()
    
    # Features numéricas derivadas
    df['tokens_per_ms'] = df['grok_tokens_used'] / (df['latency_ms'] + 1)
    df['memory_per_token'] = df['memory_usage_mb'] / (df['grok_tokens_used'] + 1)
    df['cpu_memory_ratio'] = df['cpu_percent'] / (df['memory_usage_mb'] + 1)
    
    # Features categóricas
    df['is_peak_hour'] = df['hour_of_day'].apply(
        lambda x: 1 if x in [9, 10, 11, 14, 15, 16] else 0
    )
    df['is_weekend'] = df['day_of_week'].apply(
        lambda x: 1 if x in [0, 6] else 0
    )
    
    # One-hot encoding para user_action
    df = pd.get_dummies(df, columns=['user_action'], prefix='action')
    
    # Target: prever se haverá erro (classificação)
    df['target'] = df['has_error']
    
    return df

def compute_feature_hash(df: pd.DataFrame) -> str:
    """
    Computa hash das features para versionamento.
    """
    feature_cols = sorted([c for c in df.columns if c not in ['event_id', 'agent_id', 'target']])
    feature_str = ','.join(feature_cols)
    return hashlib.sha256(feature_str.encode()).hexdigest()[:16]

def prepare_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara dados para treino.
    """
    feature_cols = [c for c in df.columns if c not in ['event_id', 'agent_id', 'target', 'has_error']]
    
    X = df[feature_cols]
    y = df['target']
    
    # Split temporal (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def log_features_to_mlflow(
    df: pd.DataFrame,
    view_name: str,
    feature_hash: str,
    experiment_name: str = "mlflow-auto-ml"
) -> str:
    """
    Loga metadados das features no MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"feature_engineering_{view_name}"):
        # Parâmetros
        mlflow.log_param("feature_view", view_name)
        mlflow.log_param("feature_hash", feature_hash)
        mlflow.log_param("n_features", len([c for c in df.columns if c not in ['event_id', 'agent_id', 'target']]))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_errors", df['target'].sum())
        mlflow.log_param("error_rate", float(df['target'].mean()))
        
        # Métricas das features
        mlflow.log_metric("avg_latency", float(df['latency_ms'].mean()))
        mlflow.log_metric("avg_memory", float(df['memory_usage_mb'].mean()))
        mlflow.log_metric("avg_cpu", float(df['cpu_percent'].mean()))
        
        # Artifact: dicionário de features (salva em local com permissão)
        feature_dict = {
            'features': [c for c in df.columns if c not in ['event_id', 'agent_id', 'target']],
            'categorical': [c for c in df.columns if c.startswith('action_')],
            'numerical': [c for c in df.columns if not c.startswith('action_') and c not in ['event_id', 'agent_id', 'target']],
            'target': 'target'
        }
        
        # Usar pasta temporária do usuário em vez de /tmp para evitar permissões
        artifact_dir = os.path.expanduser("~/.mlflow_artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        
        feature_dict_path = os.path.join(artifact_dir, "feature_dict.json")
        with open(feature_dict_path, 'w') as f:
            json.dump(feature_dict, f, indent=2)
        
        # Log artifact com tratamento de erro
        try:
            mlflow.log_artifact(feature_dict_path, 'features')
        except Exception as e:
            print(f"⚠️  Warning: Não foi possível logar artifact no MLflow: {e}")
            print(f"💾 Artifact salvo localmente em: {feature_dict_path}")
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Features logadas no MLflow: run_id={run_id}")
        return run_id

def run_feature_engineering(view_name: str = "vw_features_v1") -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    Retorna o DataFrame processado.
    """
    print("🔧 Feature Engineering...")
    
    # Extrair features
    df = get_features_from_db(view_name)
    print(f"📊 Dados brutos: {len(df)} registros")
    
    # Engenharia
    df = engineer_features(df)
    print(f"🔧 Features criadas: {len(df.columns)} colunas")
    
    # Hash
    feature_hash = compute_feature_hash(df)
    print(f"🔑 Feature hash: {feature_hash}")
    
    # Log no MLflow (com tratamento de erro)
    try:
        run_id = log_features_to_mlflow(df, view_name, feature_hash)
    except Exception as e:
        print(f"⚠️  Warning: MLflow logging falhou: {e}")
        run_id = None
    
    # Preparar treino
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    print(f"📦 Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Salvar localmente para próxima etapa
    df.to_csv(FEATURES_PATH, index=False)
    print(f"💾 Features salvas em {FEATURES_PATH}")
    
    return df

if __name__ == "__main__":
    run_feature_engineering()
