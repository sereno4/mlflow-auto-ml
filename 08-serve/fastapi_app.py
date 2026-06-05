#!/usr/bin/env python3
"""
FastAPI Serving para modelo ML.
Sempre serve o modelo com tag "Production" do MLflow Registry.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import xgboost as xgb

# Configuração
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "agent-error-predictor")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-serving")

# ============ PYDANTIC MODELS ============

class PredictionRequest(BaseModel):
    latency_ms: float
    memory_usage_mb: float
    cpu_percent: float
    grok_tokens_used: float
    hour_of_day: float = 12.0
    day_of_week: float = 1.0
    user_action: str = "chat"

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    model_stage: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    model_stage: Optional[str]
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    model_stage: str
    feature_names: List[str]
    last_loaded: Optional[str]


# ============ MODEL MANAGER ============

class ModelManager:
    """
    Gerencia carregamento e reload do modelo do MLflow.
    """
    
    # Features EXATAS do treinamento (ordem importa!)
    FEATURE_COLUMNS = [
        'latency_ms', 'memory_usage_mb', 'cpu_percent', 'grok_tokens_used',
        'high_latency', 'hour_of_day', 'day_of_week',
        'tokens_per_ms', 'memory_per_token', 'cpu_memory_ratio',
        'is_peak_hour', 'is_weekend',
        'action_chat', 'action_file_upload', 'action_tool_call'
    ]
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_stage = None
        self.last_loaded = None
        self.start_time = datetime.now()
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._load_model()
    
    def _engineer_features(self, data: Dict) -> pd.DataFrame:
        """
        Aplica MESMA engenharia de features do treinamento.
        """
        df = pd.DataFrame([data])
        
        # Features derivadas (igual ao train_xgboost)
        df['tokens_per_ms'] = df['grok_tokens_used'] / (df['latency_ms'] + 1)
        df['memory_per_token'] = df['memory_usage_mb'] / (df['grok_tokens_used'] + 1)
        df['cpu_memory_ratio'] = df['cpu_percent'] / (df['memory_usage_mb'] + 1)
        
        # Features categóricas (igual ao train_xgboost)
        df['is_peak_hour'] = df['hour_of_day'].apply(
            lambda x: 1 if x in [9, 10, 11, 14, 15, 16] else 0
        )
        df['is_weekend'] = df['day_of_week'].apply(
            lambda x: 1 if x in [0, 6] else 0
        )
        
        # high_latency (igual ao train_xgboost)
        df['high_latency'] = (df['latency_ms'] > 1000).astype(int)
        
        # One-hot encoding para user_action (igual ao train_xgboost)
        df = pd.get_dummies(df, columns=['user_action'], prefix='action')
        
        # Garantir TODAS as colunas esperadas pelo modelo
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        
        # Reordenar EXATAMENTE como no treino
        df = df[self.FEATURE_COLUMNS]
        
        # Garantir tipos numéricos
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return df
    
    def _load_model(self) -> bool:
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            prod_version = None
            
            for v in versions:
                if v.current_stage == MODEL_STAGE:
                    prod_version = v
                    break
            
            if not prod_version:
                logger.warning(f"Nenhum modelo em {MODEL_STAGE}")
                if versions:
                    prod_version = versions[0]
                else:
                    raise Exception("Nenhuma versão no registry")
            
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = prod_version.version
            self.model_stage = prod_version.current_stage
            self.last_loaded = datetime.now().isoformat()
            
            logger.info(f"✅ Modelo carregado: v{self.model_version} ({self.model_stage})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar do registry: {e}")
            return self._load_fallback()
    
    def _load_fallback(self) -> bool:
        fallback_paths = [
            "/tmp/xgboost_model.json",
            os.path.expanduser("~/.mlflow_artifacts/xgboost_model.json"),
        ]
        
        for path in fallback_paths:
            if os.path.exists(path):
                try:
                    self.model = xgb.XGBClassifier()
                    self.model.load_model(path)
                    self.model_version = "fallback"
                    self.model_stage = "local"
                    self.last_loaded = datetime.now().isoformat()
                    logger.info(f"✅ Fallback: {path}")
                    return True
                except Exception as e:
                    logger.warning(f"Fallback falhou: {e}")
        
        return False
    
    def reload(self) -> bool:
        logger.info("🔄 Reloading...")
        return self._load_model()
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if self.model is None:
            raise HTTPException(status_code=503, detail="Modelo não carregado")
        
        data = request.dict()
        df = self._engineer_features(data)
        
        try:
            if hasattr(self.model, 'predict_proba'):
                prediction = int(self.model.predict(df)[0])
                probability = float(self.model.predict_proba(df)[0][1])
            else:
                preds = self.model.predict(df)
                prediction = int(preds[0])
                probability = float(prediction)
            
            return PredictionResponse(
                prediction=prediction,
                probability=probability,
                model_version=self.model_version or "unknown",
                model_stage=self.model_stage or "unknown",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")


# ============ FASTAPI APP ============

app = FastAPI(
    title="ML Auto-ML Serving",
    description="API de serving para modelo de predição de erros",
    version="1.0.0"
)

model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI iniciado")


@app.get("/")
async def root():
    return {
        "service": "ml-auto-ml-serving",
        "status": "running",
        "model": MODEL_NAME,
        "mlflow_uri": MLFLOW_TRACKING_URI
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = (datetime.now() - model_manager.start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        model_version=model_manager.model_version,
        model_stage=model_manager.model_stage,
        uptime_seconds=uptime
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=model_manager.model_version,
        model_stage=model_manager.model_stage,
        feature_names=model_manager.FEATURE_COLUMNS,
        last_loaded=model_manager.last_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return model_manager.predict(request)


@app.post("/reload")
async def reload_model():
    success = model_manager.reload()
    if success:
        return {"status": "reloaded", "version": model_manager.model_version}
    raise HTTPException(status_code=500, detail="Falha ao recarregar")


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    return [model_manager.predict(r) for r in requests]


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
