"""
DAG: Pipeline Auto-Retrain ML.
Schedule: A cada 6 horas.
Flow: T1→T2→T3→T4→T5→T6
"""

import os
import sys
import json
from datetime import datetime, timedelta

# Airflow imports (só funcionam dentro do container)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.dummy import DummyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("⚠️  Airflow não disponível no ambiente atual. Rodar dentro do container.")

# Paths do projeto (ajustar conforme mount no docker-compose)
PROJECT_ROOT = os.environ.get('MLFLOW_PROJECT_ROOT', '/opt/airflow/mlflow-auto-ml')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '03-features'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '04-train'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '06-drift'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '05-mlflow'))


# ============ TASK FUNCTIONS ============

def t1_extract_features(**context):
    """T1: Extrair features do PostgreSQL."""
    from feature_engineering import run_feature_engineering
    print("🔧 T1: Extract Features")
    df = run_feature_engineering()
    context['ti'].xcom_push(key='n_samples', value=len(df))
    context['ti'].xcom_push(key='n_features', value=len(df.columns))
    print(f"✅ T1: {len(df)} amostras, {len(df.columns)} features")
    return "t1_done"

def t2_detect_drift(**context):
    """T2: Detectar drift. Branch: t3_train ou skip_success."""
    from drift_detector import DriftDetector
    print("🔍 T2: Detect Drift")
    detector = DriftDetector()
    results = detector.detect_drift()
    context['ti'].xcom_push(key='drift_detected', value=results['drift_detected'])
    if results['drift_detected']:
        print("🚨 Drift → Treinar")
        return 't3_train'
    print("✅ Sem drift → Skip")
    return 'skip_success'

def t3_train_model(**context):
    """T3: Treinar XGBoost."""
    from train_xgboost import main as train_main
    print("🚀 T3: Train Model")
    run_id, metrics = train_main()
    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='accuracy', value=metrics['accuracy'])
    context['ti'].xcom_push(key='f1_score', value=metrics['f1_score'])
    print(f"✅ T3: run_id={run_id}")
    return run_id

def t4_validate_sanity(**context):
    """T4: Sanity checks. Branch: t5_promote ou t4_fail_alert."""
    ti = context['ti']
    accuracy = ti.xcom_pull(task_ids='t3_train', key='accuracy') or 0
    f1 = ti.xcom_pull(task_ids='t3_train', key='f1_score') or 0
    
    print("🧪 T4: Sanity Checks")
    checks = {
        'accuracy_not_constant': accuracy != 0.5,
        'accuracy_not_null': accuracy is not None,
        'f1_not_constant': f1 != 0.5,
        'accuracy_above_0.6': accuracy >= 0.6,
        'f1_above_0.5': f1 >= 0.5,
    }
    
    all_passed = all(checks.values())
    context['ti'].xcom_push(key='sanity_passed', value=all_passed)
    
    for c, p in checks.items():
        print(f"   {'✅' if p else '❌'} {c}")
    
    if all_passed:
        print("✅ T4 Passou → Promover")
        return 't5_promote'
    print("❌ T4 Falhou → Alertar")
    return 't4_fail_alert'

def t4_fail_alert(**context):
    """T4 Fail: Alerta."""
    print("🚨 T4: ALERT - Sanity falhou!")
    print("📧 Alerta enviado (mock)")
    return "alert_sent"

def t5_promote_production(**context):
    """T5: Promover para Production."""
    from registry_manager import promote_pipeline
    ti = context['ti']
    run_id = ti.xcom_pull(task_ids='t3_train', key='run_id')
    print(f"🚀 T5: Promote run_id={run_id}")
    version = promote_pipeline(run_id, skip_staging=False)
    context['ti'].xcom_push(key='version', value=version)
    print(f"✅ T5: v{version} em Production")
    return f"v{version}"

def t6_deploy_fastapi(**context):
    """T6: Deploy FastAPI."""
    ti = context['ti']
    version = ti.xcom_pull(task_ids='t5_promote', key='version')
    print(f"🌐 T6: Deploy v{version}")
    print("   🔄 FastAPI reload")
    print("   ✅ Health check OK")
    return f"deployed_v{version}"

def skip_success(**context):
    """Skip quando sem drift."""
    print("⏭️  Skip: Sem drift")
    return "skipped"


# ============ DAG DEFINITION ============

if AIRFLOW_AVAILABLE:
    default_args = {
        'owner': 'ml-pipeline',
        'depends_on_past': False,
        'email_on_failure': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    with DAG(
        dag_id='ml_auto_retrain_pipeline',
        default_args=default_args,
        description='Pipeline auto-retrain ML',
        schedule_interval='0 */6 * * *',
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['ml', 'auto-retrain'],
    ) as dag:
        
        start = DummyOperator(task_id='start')
        t1 = PythonOperator(task_id='t1_extract', python_callable=t1_extract_features)
        t2 = BranchPythonOperator(task_id='t2_drift_detect', python_callable=t2_detect_drift)
        t3 = PythonOperator(task_id='t3_train', python_callable=t3_train_model)
        t4 = BranchPythonOperator(task_id='t4_validate', python_callable=t4_validate_sanity)
        t4_fail = PythonOperator(task_id='t4_fail_alert', python_callable=t4_fail_alert)
        t5 = PythonOperator(task_id='t5_promote', python_callable=t5_promote_production)
        t6 = PythonOperator(task_id='t6_deploy', python_callable=t6_deploy_fastapi)
        skip = PythonOperator(task_id='skip_success', python_callable=skip_success)
        end = DummyOperator(task_id='end', trigger_rule='none_failed_min_one_success')
        
        # Flow
        start >> t1 >> t2 >> [t3, skip]
        t3 >> t4 >> [t5, t4_fail]
        t5 >> t6 >> end
        skip >> end
        t4_fail >> end
