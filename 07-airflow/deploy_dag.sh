#!/bin/bash
# Deploy DAG para container Airflow

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAG_FILE="$PROJECT_ROOT/07-airflow/dag_auto_retrain.py"
AIRFLOW_CONTAINER="mlflow-airflow-webserver-1"  # ajuste conforme seu compose
AIRFLOW_DAGS_PATH="/opt/airflow/dags"

echo "🚀 Deploy DAG para Airflow..."

# Verificar container
if ! docker ps | grep -q airflow; then
    echo "❌ Container Airflow não encontrado. Verifique o nome:"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep airflow
    echo ""
    echo "Ajuste AIRFLOW_CONTAINER no script e tente novamente."
    exit 1
fi

# Encontrar container correto
CONTAINER=$(docker ps --format "{{.Names}}" | grep airflow | head -1)
echo "📦 Container encontrado: $CONTAINER"

# Criar pasta de DAGs no container
docker exec $CONTAINER mkdir -p $AIRFLOW_DAGS_PATH

# Copiar DAG
docker cp "$DAG_FILE" "$CONTAINER:$AIRFLOW_DAGS_PATH/"
echo "📄 DAG copiado: $DAG_FILE → $CONTAINER:$AIRFLOW_DAGS_PATH/"

# Setar variável de ambiente para o projeto
docker exec $CONTAINER bash -c "echo 'export MLFLOW_PROJECT_ROOT=/opt/airflow/mlflow-auto-ml' >> /opt/airflow/airflow.env" 2>/dev/null || true

# Validar DAG
echo "🔍 Validando DAG..."
docker exec $CONTAINER python -c "
import sys
sys.path.insert(0, '$AIRFLOW_DAGS_PATH')
from dag_auto_retrain import dag
print(f'✅ DAG válido: {dag.dag_id}')
print(f'   Tasks: {len(dag.tasks)}')
print(f'   Schedule: {dag.schedule_interval}')
" || echo "⚠️  Validação falhou (pode precisar de dependências)"

echo "✅ Deploy concluído!"
echo ""
echo "Para ver o DAG na UI:"
echo "   http://localhost:8080"
