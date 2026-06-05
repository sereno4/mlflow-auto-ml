#!/usr/bin/env python3
"""
TESTE ETAPA 7: Airflow DAG (sem importar airflow)
"""

import sys
import os
import ast

def test_file_exists():
    dag_path = os.path.join(os.path.dirname(__file__), 'dag_auto_retrain.py')
    assert os.path.exists(dag_path), "DAG file não encontrado"
    print("✅ DAG file existe")
    return True

def test_syntax_valid():
    dag_path = os.path.join(os.path.dirname(__file__), 'dag_auto_retrain.py')
    with open(dag_path, 'r') as f:
        code = f.read()
    ast.parse(code)
    print("✅ Syntax válida")
    return True

def test_tasks_defined():
    dag_path = os.path.join(os.path.dirname(__file__), 'dag_auto_retrain.py')
    with open(dag_path, 'r') as f:
        code = f.read()
    
    expected_tasks = ['t1_extract_features', 't2_detect_drift', 't3_train_model',
                      't4_validate_sanity', 't5_promote_production', 't6_deploy_fastapi',
                      't4_fail_alert', 'skip_success']
    
    for task in expected_tasks:
        assert f"def {task}" in code, f"Task {task} não definida"
    
    print(f"✅ Tasks definidas: {len(expected_tasks)} tasks")
    return True

def test_dag_id():
    dag_path = os.path.join(os.path.dirname(__file__), 'dag_auto_retrain.py')
    with open(dag_path, 'r') as f:
        code = f.read()
    assert "dag_id='ml_auto_retrain_pipeline'" in code, "DAG ID incorreto"
    assert "schedule_interval" in code, "Schedule não definido"
    print("✅ DAG ID e schedule OK")
    return True

def test_flow_structure():
    dag_path = os.path.join(os.path.dirname(__file__), 'dag_auto_retrain.py')
    with open(dag_path, 'r') as f:
        code = f.read()
    
    # Verificar branches
    assert "BranchPythonOperator" in code, "BranchPythonOperator não usado"
    assert "t2 >> [t3, skip]" in code or "t2 >> [t3, skip]" in code.replace(" ", ""), "Flow T2 incorreto"
    assert "t4 >> [t5, t4_fail]" in code or "t4 >> [t5, t4_fail]" in code.replace(" ", ""), "Flow T4 incorreto"
    
    print("✅ Flow structure OK")
    return True

def test_xcom_usage():
    dag_path = os.path.join(os.path.dirname(__file__), 'dag_auto_retrain.py')
    with open(dag_path, 'r') as f:
        code = f.read()
    
    assert "xcom_push" in code, "xcom_push não usado"
    assert "xcom_pull" in code, "xcom_pull não usado"
    print("✅ XCom usage OK")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 7: Airflow DAG (Static Analysis)")
    print("=" * 60)
    
    tests = [
        ("File Exists", test_file_exists),
        ("Syntax Valid", test_syntax_valid),
        ("Tasks Defined", test_tasks_defined),
        ("DAG ID", test_dag_id),
        ("Flow Structure", test_flow_structure),
        ("XCom Usage", test_xcom_usage),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- Testando: {name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} falhou: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed}/{len(tests)} testes passaram")
    print("=" * 60)
    
    if passed == len(tests):
        print("🎉 ETAPA 7 COMPLETA! Pronto para Etapa 8.")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
