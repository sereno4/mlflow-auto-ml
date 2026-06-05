#!/bin/bash
echo "========================================"
echo "🔍 DIAGNÓSTICO MLflow Auto-ML"
echo "========================================"

echo ""
echo "📦 CONTAINERS:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -E "mlflow|airflow|postgres" || echo "   Nenhum container encontrado"

echo ""
echo "📊 LOGS MLflow (últimas 30 linhas):"
docker logs --tail 30 mlflow-server 2>&1 || echo "   ❌ mlflow-server não encontrado"

echo ""
echo "📊 LOGS Airflow Webserver (últimas 30 linhas):"
docker logs --tail 30 airflow-webserver 2>&1 || echo "   ❌ airflow-webserver não encontrado"

echo ""
echo "📊 LOGS Airflow Init:"
docker logs airflow-init 2>&1 | tail -20 || echo "   ❌ airflow-init não encontrado"

echo ""
echo "🌐 CONECTIVIDADE:"
echo -n "   PostgreSQL (5432): "
timeout 3 bash -c "cat < /dev/null > /dev/tcp/localhost/5432" 2>/dev/null && echo "✅ OK" || echo "❌ OFF"
echo -n "   MLflow (5000): "
timeout 3 bash -c "cat < /dev/null > /dev/tcp/localhost/5000" 2>/dev/null && echo "✅ OK" || echo "❌ OFF"
echo -n "   Airflow (8080): "
timeout 3 bash -c "cat < /dev/null > /dev/tcp/localhost/8080" 2>/dev/null && echo "✅ OK" || echo "❌ OFF"

echo ""
echo "🔐 AIRFLOW USERS:"
docker exec airflow-webserver airflow users list 2>/dev/null || echo "   ❌ Não foi possível listar"

echo ""
echo "🗄️ POSTGRES DBs:"
docker exec mlflow-postgres psql -U mlflow -l 2>/dev/null || echo "   ❌ Não conectou"
