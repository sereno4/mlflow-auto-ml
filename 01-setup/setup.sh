#!/bin/bash
set -e

# Detectar docker compose v2 ou v1
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
    echo "✅ Usando Docker Compose v2"
elif docker-compose version &>/dev/null; then
    COMPOSE="docker-compose"
    echo "⚠️ Usando Docker Compose v1"
else
    echo "❌ Docker Compose não encontrado!"
    echo "Instale: sudo apt install docker-compose-plugin"
    exit 1
fi

echo ""
echo "========================================"
echo "🧹 LIMPANDO AMBIENTE"
echo "========================================"

$COMPOSE down -v 2>/dev/null || true
docker stop mlflow-postgres mlflow-server airflow-webserver airflow-scheduler airflow-init 2>/dev/null || true
docker rm -f mlflow-postgres mlflow-server airflow-webserver airflow-scheduler airflow-init 2>/dev/null || true
docker volume prune -f 2>/dev/null || true

mkdir -p mlruns mlartifacts
chmod 777 mlruns mlartifacts 2>/dev/null || true

echo ""
echo "========================================"
echo "🐳 SUBINDO CONTAINERS"
echo "========================================"

# Subir PostgreSQL primeiro
$COMPOSE up -d postgres

echo "⏳ Aguardando PostgreSQL..."
sleep 10

until docker exec mlflow-postgres pg_isready -U mlflow -d mlflow_db 2>/dev/null; do
    echo "   ⏳ PostgreSQL ainda não pronto..."
    sleep 3
done
echo "✅ PostgreSQL pronto!"

# Build e subir MLflow (com psycopg2)
echo "🔨 Buildando MLflow com psycopg2..."
$COMPOSE up --build -d mlflow
echo "⏳ Aguardando MLflow..."
sleep 10

# Verificar se MLflow subiu
until curl -s http://localhost:5000 >/dev/null 2>&1; do
    echo "   ⏳ MLflow ainda não responde..."
    sleep 3
done
echo "✅ MLflow pronto!"

# Airflow init
$COMPOSE up airflow-init
echo "✅ Airflow init completo!"

# Airflow webserver + scheduler
$COMPOSE up -d airflow-webserver airflow-scheduler

echo ""
echo "========================================"
echo "🔍 VERIFICANDO SERVIÇOS"
echo "========================================"

$COMPOSE ps

echo ""
echo "Testando conectividade:"
echo -n "   PostgreSQL (5432): "
timeout 3 bash -c "cat < /dev/null > /dev/tcp/localhost/5432" 2>/dev/null && echo "✅ OK" || echo "❌ OFF"

echo -n "   MLflow (5000): "
timeout 3 bash -c "cat < /dev/null > /dev/tcp/localhost/5000" 2>/dev/null && echo "✅ OK" || echo "❌ OFF"

echo -n "   Airflow (8080): "
timeout 3 bash -c "cat < /dev/null > /dev/tcp/localhost/8080" 2>/dev/null && echo "✅ OK" || echo "❌ OFF"

echo ""
echo "========================================"
echo "🎉 SETUP COMPLETO!"
echo "========================================"
echo ""
echo "URLs:"
echo "   MLflow UI:    http://localhost:5000"
echo "   Airflow UI:   http://localhost:8080"
echo "   Airflow Login: admin / admin"
echo ""
echo "Logs:"
echo "   $COMPOSE logs -f mlflow"
echo "   $COMPOSE logs -f airflow-webserver"
