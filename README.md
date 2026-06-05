# 🚀 Production-Ready AutoML Pipeline with MLOps Governance

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Apache_Airflow-2.x-017CE2?logo=apache-airflow)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?logo=mlflow)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-<10ms_Latency-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

Este repositório contém um ecossistema completo de **AutoML e Engenharia de MLOps End-to-End (E2E)** focado em resiliência, governança de modelos e monitoramento em produção. O pipeline automatiza desde a ingestão de dados brutos até o serviço em tempo real com baixa latência, aplicando validações rigorosas em cada etapa.

---

## Arquitetura - MLFlow Auto ML Platform
┌──────────────────────────────────────────────────────────────┐
│                    DATA & TELEMETRY LAYER                   │
├──────────────────────────────────────────────────────────────┤
│ PostgreSQL                                                   │
│ • Dados históricos de treinamento                            │
│ • Logs de inferência                                          │
│ • Eventos para detecção de Drift                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                        │
│                        Apache Airflow                        │
├──────────────────────────────────────────────────────────────┤
│ Schedule: A cada 6 horas                                     │
│                                                              │
│ Extract Data                                                 │
│        ↓                                                     │
│ Drift Detection                                              │
│        ↓                                                     │
│ ┌───────────────┬─────────────────┐                          │
│ │ Drift Found   │ No Drift        │                          │
│ ▼               ▼                 │                          │
│ Train Model     Keep Current      │                          │
│ ▼                                 │                          │
│ Validate Model                    │                          │
│ ▼                                 │                          │
│ Promote Model                     │                          │
│ ▼                                 │                          │
│ Reload Serving                    │                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  MODEL GOVERNANCE LAYER                      │
│                     MLflow Registry                          │
├──────────────────────────────────────────────────────────────┤
│ • Experiment Tracking                                        │
│ • Model Versioning                                           │
│ • Model Registry                                             │
│ • Production Promotion                                       │
│ • Rollback Capability                                        │
│ • Artifact Persistence                                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    MODEL SERVING LAYER                       │
│                         FastAPI                              │
├──────────────────────────────────────────────────────────────┤
│ • REST Inference API                                         │
│ • Pydantic Validation                                        │
│ • Hot Reload via Webhook                                     │
│ • Cached Model Fallback                                      │
│ • Low Latency Inference (<10ms)                              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   MONITORING & RELIABILITY                   │
├──────────────────────────────────────────────────────────────┤
│ • Drift Detection                                            │
│ • Automated Retraining                                       │
│ • Slack/Webhook Alerts                                       │
│ • Model Health Checks                                        │
│ • Production Safety Gates                                    │
└──────────────────────────────────────────────────────────────┘
🚀 Principais Capacidades
Detecção automática de Data Drift
Retraining automatizado baseado em evidências
Governança completa de modelos com MLflow
Promoção automática para produção
Validação com múltiplos sanity checks
Inferência de baixa latência (<10ms)
Recuperação automática em caso de falha
Arquitetura orientada a MLOps e produção
🛠️ Stack Tecnológica

Machine Learning

XGBoost
Scikit-Learn
Pandas
NumPy

MLOps

MLflow
Apache Airflow

Backend

FastAPI
Pydantic

Data

PostgreSQL

Infraestrutura

Docker
Docker Compose

Observabilidade

Logging
Drift Monitoring
Alerting





---

## 📁 Estrutura

mlflow-auto-ml/
├── 01-setup/
│   ├── docker-compose.yml      # Infra: Postgres + MLflow + Airflow
│   ├── Dockerfile.mlflow       # Imagem customizada com psycopg2
│   ├── setup.sh                # Script de setup completo
│   └── test_env.py             # ✅ 6/6 tests
├── 02-data/
│   ├── init_postgres.sql       # Schema + Views + 30 registros
│   └── test_data.py            # ✅ 5/5 tests
├── 03-features/
│   ├── feature_engineering.py  # SQLAlchemy + feature hash
│   └── test_features.py        # ✅ 5/5 tests
├── 04-train/
│   ├── train_xgboost.py        # Treino CPU-only < 30s
│   └── test_train.py           # ✅ 6/6 tests
├── 05-mlflow/
│   ├── registry_manager.py     # Promote/Archive/Delete
│   └── test_mlflow.py          # ✅ 6/6 tests
├── 06-drift/
│   ├── drift_detector.py       # Evidently fallback scipy
│   └── test_drift.py           # ✅ 6/6 tests
├── 07-airflow/
│   ├── dag_auto_retrain.py     # DAG T1→T2→T3→T4→T5→T6
│   └── test_dag.py             # ✅ 6/6 tests
├── 08-serve/
│   ├── fastapi_app.py          # 15 features, < 10ms
│   └── test_api.py             # ✅ 6/6 tests
├── 09-sanity/
│   ├── smoke_tests.py          # 5 checks Regra de Ouro
│   └── test_sanity.py          # ✅ 8/8 tests
└── README.md
plain

**Total: 54/54 testes passando 🎉**


---

Rodar testes de todas as etapas
bash
cd ..
python3 01-setup/test_env.py      # ✅ Infra
python3 02-data/test_data.py      # ✅ Dados
python3 03-features/test_features.py  # ✅ Features
python3 04-train/test_train.py    # ✅ Treinamento
python3 05-mlflow/test_mlflow.py  # ✅ Registry
python3 06-drift/test_drift.py   # ✅ Drift
python3 07-airflow/test_dag.py   # ✅ DAG
python3 08-serve/test_api.py     # ✅ API
python3 09-sanity/test_sanity.py # ✅ Sanity


🔬 Etapas do Pipeline
T1: Extract Features
Conecta no PostgreSQL via SQLAlchemy
Extrai da view vw_features_v1
Cria 15 features derivadas
Gera feature hash para versionamento
T2: Drift Detection
Compara dados atuais vs baseline de referência
Evidently (quando instalado) ou fallback scipy KS-test
Se drift > 50% → continua para treino
Se sem drift → skip_success
T3: Train XGBoost
Treino CPU-only em < 30s
100 estimators, max_depth=4
Loga métricas no MLflow
T4: Sanity Checks (Regra de Ouro)
Not Constant: predições variam
Not Null: sem NaN/None
Class Balance: não 100% uma classe
Metric Threshold: accuracy ≥ 0.6, f1 ≥ 0.5
Latency: treino < 30s, predict < 10ms
T5: Promote Production
MLflow Registry: Staging → Production
Versão anterior → Archived
T6: Deploy FastAPI
Reload do model serving
Health check automático
🛠️ Tecnologias
Table
Camada	Stack
Orquestração	Apache Airflow 2.9
Tracking	MLflow 2.14
ML	XGBoost, scikit-learn
Dados	PostgreSQL 15, SQLAlchemy, Pandas
Serving	FastAPI, Uvicorn
Drift	Evidently (fallback scipy)
Infra	Docker, Docker Compose
📸 Screenshots
Airflow DAG — Pipeline Completo
screenshots/airflow_dag.png
MLflow — Experiments
screenshots/mlflow_ui.png


