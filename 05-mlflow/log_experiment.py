#!/usr/bin/env python3
"""
Log de Experimento no MLflow.
Wrapper simples para logar métricas, parâmetros e artefatos.
"""

import os
import json
import mlflow
from typing import Dict, Any, Optional

MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

def log_experiment(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, str]] = None,
    experiment_name: str = "mlflow-auto-ml"
) -> str:
    """
    Loga um experimento completo no MLflow.
    
    Args:
        run_name: Nome da run
        params: Dicionário de parâmetros
        metrics: Dicionário de métricas
        artifacts: Dict {nome: caminho_arquivo}
        experiment_name: Nome do experimento
    
    Returns:
        run_id da run criada
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Log parâmetros
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log métricas
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log artefatos
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    try:
                        mlflow.log_artifact(path, name)
                    except Exception as e:
                        print(f"⚠️  Não foi possível logar artifact {name}: {e}")
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Experimento logado: run_id={run_id}")
        return run_id


if __name__ == "__main__":
    # Teste simples
    params = {"model": "xgboost", "n_estimators": 100}
    metrics = {"accuracy": 0.95, "f1": 0.94}
    
    run_id = log_experiment("test_experiment", params, metrics)
    print(f"Run ID: {run_id}")
