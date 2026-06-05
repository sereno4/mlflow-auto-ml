#!/usr/bin/env python3
"""
Gera dados de amostra e cria estrutura no PostgreSQL.
"""
import os
import random
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DB_URL", "postgresql://mlflow:mlflow123@localhost:5432/mlflow_db")
engine = create_engine(DB_URL)

def create_schema_and_data():
    with engine.connect() as conn:
        # Schema
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS feature_store"))
        conn.execute(text("COMMIT"))

        # Tabela base
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feature_store.agent_events (
                event_id        SERIAL PRIMARY KEY,
                agent_id        VARCHAR(50),
                latency_ms      FLOAT,
                memory_usage_mb FLOAT,
                cpu_percent     FLOAT,
                grok_tokens_used INT,
                has_error       BOOLEAN,
                high_latency    BOOLEAN,
                hour_of_day     INT,
                day_of_week     INT,
                user_action     VARCHAR(50),
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("COMMIT"))

        # Dados fake
        n = 1000
        random.seed(42)
        np.random.seed(42)
        actions = ['click', 'scroll', 'submit', 'navigate', 'search']
        rows = []
        for i in range(n):
            latency = np.random.exponential(200)
            rows.append({
                'agent_id':        f'agent_{random.randint(1,10):02d}',
                'latency_ms':      round(latency, 2),
                'memory_usage_mb': round(np.random.normal(512, 128), 2),
                'cpu_percent':     round(np.random.uniform(5, 95), 2),
                'grok_tokens_used':random.randint(100, 4000),
                'has_error':       random.random() < 0.1,
                'high_latency':    latency > 500,
                'hour_of_day':     random.randint(0, 23),
                'day_of_week':     random.randint(0, 6),
                'user_action':     random.choice(actions),
            })

        df = pd.DataFrame(rows)
        # Inserir só se tabela vazia
        count = conn.execute(text("SELECT COUNT(*) FROM feature_store.agent_events")).scalar()
        if count == 0:
            df.to_sql('agent_events', engine, schema='feature_store',
                      if_exists='append', index=False)
            print(f"✅ {n} linhas inseridas")
        else:
            print(f"ℹ️  Tabela já tem {count} linhas, pulando inserção")

        # View
        conn.execute(text("DROP VIEW IF EXISTS feature_store.vw_features_v1"))
        conn.execute(text("""
            CREATE VIEW feature_store.vw_features_v1 AS
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
            FROM feature_store.agent_events
        """))
        conn.execute(text("COMMIT"))
        print("✅ View feature_store.vw_features_v1 criada")

if __name__ == "__main__":
    create_schema_and_data()
    print("✅ Setup completo!")
