MLFlow Auto ML Platform

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Apache_Airflow-2.x-017CE2?logo=apache-airflow)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?logo=mlflow)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-<10ms_Latency-009688?logo=fastapi)](https://fastapi.tiangolo.com/)


Plataforma AutoML pronta para produção com governança MLOps, detecção automática de drift, re-treinamento inteligente e deploy contínuo de modelos de Machine Learning.








🎯 Visão Geral

Este projeto implementa um pipeline completo de Machine Learning em produção, automatizando todo o ciclo de vida do modelo:

Extração de dados
Engenharia de features
Treinamento
Detecção de drift
Validação
Governança
Deploy
Monitoramento

O objetivo é demonstrar uma arquitetura MLOps real, resiliente e reproduzível.

🏗️ Arquitetura
┌──────────────────────────────────────────────┐
│           DATA & TELEMETRY LAYER             │
├──────────────────────────────────────────────┤
│ PostgreSQL                                   │
│ • Training Data                              │
│ • Inference Logs                             │
│ • Drift Monitoring Events                    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│            ORCHESTRATION LAYER               │
│              Apache Airflow                  │
├──────────────────────────────────────────────┤
│ Extract Data                                 │
│        ↓                                     │
│ Drift Detection                              │
│        ↓                                     │
│  Drift?                                      │
│   ├─ Yes → Train → Validate → Promote       │
│   └─ No  → Keep Current Model               │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│           GOVERNANCE LAYER                   │
│             MLflow Registry                  │
├──────────────────────────────────────────────┤
│ • Experiment Tracking                        │
│ • Model Versioning                           │
│ • Registry                                   │
│ • Promotion                                  │
│ • Rollback                                   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│            SERVING LAYER                     │
│                FastAPI                       │
├──────────────────────────────────────────────┤
│ • REST API                                   │
│ • Pydantic Validation                        │
│ • Hot Reload                                 │
│ • Cached Model Fallback                      │
│ • <10ms Inference                            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│        MONITORING & RELIABILITY              │
├──────────────────────────────────────────────┤
│ • Drift Detection                            │
│ • Automated Retraining                       │
│ • Slack Alerts                               │
│ • Health Checks                              │
│ • Production Safety Gates                    │
└──────────────────────────────────────────────┘
🚀 Principais Capacidades

✅ Detecção automática de Data Drift

✅ Retraining automatizado baseado em evidências

✅ Governança completa com MLflow

✅ Promoção automática para produção

✅ Rollback seguro de modelos

✅ Inferência de baixa latência (<10ms)

✅ Monitoramento contínuo

✅ Pipeline reproduzível de ponta a ponta

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
Dados
PostgreSQL
SQLAlchemy
Infraestrutura
Docker
Docker Compose
Observabilidade
Drift Monitoring
Logging
Alerting
📁 Estrutura do Projeto
mlflow-auto-ml/
│
├── 01-setup/
│   ├── docker-compose.yml
│   ├── Dockerfile.mlflow
│   ├── setup.sh
│   └── test_env.py
│
├── 02-data/
│   ├── init_postgres.sql
│   └── test_data.py
│
├── 03-features/
│   ├── feature_engineering.py
│   └── test_features.py
│
├── 04-train/
│   ├── train_xgboost.py
│   └── test_train.py
│
├── 05-mlflow/
│   ├── registry_manager.py
│   └── test_mlflow.py
│
├── 06-drift/
│   ├── drift_detector.py
│   └── test_drift.py
│
├── 07-airflow/
│   ├── dag_auto_retrain.py
│   └── test_dag.py
│
├── 08-serve/
│   ├── fastapi_app.py
│   └── test_api.py
│
├── 09-sanity/
│   ├── smoke_tests.py
│   └── test_sanity.py
│
└── README.md
🧪 Qualidade
Total de Testes: 54/54 ✅

Infraestrutura ............ 6/6
Dados ..................... 5/5
Features .................. 5/5
Treinamento ............... 6/6
MLflow Registry ........... 6/6
Drift Detection ........... 6/6
Airflow DAG ............... 6/6
FastAPI Serving ........... 6/6
Sanity Checks ............. 8/8
🔬 Pipeline de Treinamento
T1 — Extração e Features
Conexão PostgreSQL via SQLAlchemy
Extração da View vw_features_v1
Criação de 15 features derivadas
Versionamento por hash
T2 — Drift Detection
Evidently AI
KS-Test (fallback)
Comparação com baseline
T3 — Treinamento
XGBoost
CPU Only
Tempo médio inferior a 30 segundos
T4 — Sanity Checks
Predições não constantes
Ausência de NaN
Balanceamento de classes
F1 Score mínimo
Latência validada
T5 — Promoção
Registry MLflow
Produção automática
Arquivamento da versão anterior
T6 — Deploy
FastAPI Hot Reload
Health Check
Modelo ativo em produção
📈 Resultados
Pipeline totalmente automatizado
Deploy contínuo de modelos
Inferência inferior a 10ms
Governança completa do ciclo de vida
54/54 testes passando
Arquitetura pronta para produção
📜 Licença

MIT License
