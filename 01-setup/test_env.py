#!/usr/bin/env python3
"""
TESTE ETAPA 1: Ambiente e Infraestrutura
Valida se Docker Compose subiu corretamente.
"""

import sys
import time
import requests
import subprocess

def test_docker_running():
    """Verifica se Docker está rodando."""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker está rodando")
            return True
        else:
            print("❌ Docker não está respondendo")
            return False
    except Exception as e:
        print(f"❌ Docker não encontrado: {e}")
        return False

def test_compose_file():
    """Verifica se docker-compose.yml é válido."""
    try:
        result = subprocess.run(
            ['docker', 'compose', '-f', '01-setup/docker-compose.yml', 'config'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ docker-compose.yml é válido")
            return True
        else:
            print(f"❌ docker-compose.yml inválido: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erro ao validar compose: {e}")
        return False

def test_mlflow_up():
    """Verifica se MLflow responde."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:5000', timeout=2)
            if response.status_code == 200:
                print("✅ MLflow server está rodando (localhost:5000)")
                return True
        except:
            pass
        time.sleep(1)
        print(f"  ⏳ Aguardando MLflow... ({i+1}/{max_retries})")
    
    print("❌ MLflow não respondeu após 30s")
    return False

def test_postgres_up():
    """Verifica se PostgreSQL responde."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="mlflow",
            password="mlflow123",
            dbname="mlflow_db"
        )
        conn.close()
        print("✅ PostgreSQL está acessível")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL não acessível: {e}")
        return False

def test_airflow_up():
    """Verifica se Airflow Webserver responde."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8080', timeout=2)
            if response.status_code in [200, 302]:
                print("✅ Airflow Webserver está rodando (localhost:8080)")
                return True
        except:
            pass
        time.sleep(1)
        print(f"  ⏳ Aguardando Airflow... ({i+1}/{max_retries})")
    
    print("❌ Airflow não respondeu após 30s")
    return False

def test_python_imports():
    """Verifica se bibliotecas Python instalam."""
    try:
        import sklearn, xgboost, pandas, numpy
        import mlflow, evidently, fastapi, psycopg2
        print("✅ Todas as bibliotecas Python importam corretamente")
        return True
    except ImportError as e:
        print(f"❌ Biblioteca faltando: {e}")
        print("   Execute: pip install -r 01-setup/requirements.txt")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 1: Ambiente e Infraestrutura")
    print("=" * 60)
    
    tests = [
        ("Docker", test_docker_running),
        ("Compose File", test_compose_file),
        ("Python Imports", test_python_imports),
        ("PostgreSQL", test_postgres_up),
        ("MLflow", test_mlflow_up),
        ("Airflow", test_airflow_up),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- Testando: {name} ---")
        if test_func():
            passed += 1
        else:
            if name in ["MLflow", "Airflow", "PostgreSQL"]:
                print("   💡 Dica: Execute 'docker compose -f 01-setup/docker-compose.yml up -d'")
    
    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed}/{len(tests)} testes passaram")
    print("=" * 60)
    
    if passed == len(tests):
        print("🎉 ETAPA 1 COMPLETA! Pronto para Etapa 2.")
        sys.exit(0)
    else:
        print("⚠️  Alguns serviços não estão prontos. Verifique os logs.")
        sys.exit(1)
