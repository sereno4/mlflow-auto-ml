#!/usr/bin/env python3
"""
TESTE ETAPA 2: Dados e Feature Store
Valida schema, dados e views.
"""

import sys
import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'mlflow',
    'password': 'mlflow123',
    'dbname': 'mlflow_db'
}

def test_schema_exists():
    """Verifica se schema foi criado."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'feature_store';")
        result = cur.fetchone()
        conn.close()
        if result:
            print("✅ Schema 'feature_store' existe")
            return True
        else:
            print("❌ Schema 'feature_store' não encontrado")
            return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_tables_exist():
    """Verifica se tabelas foram criadas."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'feature_store' 
            AND table_type = 'BASE TABLE';
        """)
        tables = [row[0] for row in cur.fetchall()]
        conn.close()
        
        expected = ['raw_events', 'reference_data']
        missing = [t for t in expected if t not in tables]
        
        if not missing:
            print(f"✅ Tabelas OK: {tables}")
            return True
        else:
            print(f"❌ Tabelas faltando: {missing}")
            return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_views_exist():
    """Verifica se views foram criadas."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.views 
            WHERE table_schema = 'feature_store';
        """)
        views = [row[0] for row in cur.fetchall()]
        conn.close()
        
        expected = ['vw_features_v1', 'vw_features_v2', 'vw_monitoring']
        missing = [v for v in expected if v not in views]
        
        if not missing:
            print(f"✅ Views OK: {views}")
            return True
        else:
            print(f"❌ Views faltando: {missing}")
            return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_data_count():
    """Verifica se dados foram inseridos."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feature_store.raw_events;")
        count = cur.fetchone()[0]
        conn.close()
        
        if count >= 10:
            print(f"✅ Dados inseridos: {count} registros")
            return True
        else:
            print(f"❌ Poucos dados: {count} registros")
            return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_feature_view_query():
    """Testa query na view de features."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT agent_id, has_error, high_latency, hour_of_day 
            FROM feature_store.vw_features_v1 
            LIMIT 3;
        """)
        rows = cur.fetchall()
        conn.close()
        
        if len(rows) == 3:
            print(f"✅ View vw_features_v1 retorna dados: {len(rows)} rows")
            return True
        else:
            print(f"❌ View não retorna dados esperados")
            return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 2: Dados e Feature Store")
    print("=" * 60)
    
    tests = [
        ("Schema", test_schema_exists),
        ("Tabelas", test_tables_exist),
        ("Views", test_views_exist),
        ("Dados", test_data_count),
        ("Feature View", test_feature_view_query),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- Testando: {name} ---")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed}/{len(tests)} testes passaram")
    print("=" * 60)
    
    if passed == len(tests):
        print("🎉 ETAPA 2 COMPLETA! Pronto para Etapa 3.")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
