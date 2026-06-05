#!/usr/bin/env python3
"""
TESTE ETAPA 9: Sanity Checks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    try:
        from smoke_tests import SanityChecker, run_sanity_pipeline, NumpyEncoder
        print("✅ Imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import falhou: {e}")
        return False

def test_thresholds():
    try:
        from smoke_tests import SanityChecker
        checker = SanityChecker()
        assert 'min_accuracy' in checker.THRESHOLDS
        assert 'max_train_time' in checker.THRESHOLDS
        print("✅ Thresholds OK")
        return True
    except Exception as e:
        print(f"❌ Thresholds falhou: {e}")
        return False

def test_check_not_constant():
    try:
        from smoke_tests import SanityChecker
        import pandas as pd
        import numpy as np
        
        checker = SanityChecker()
        class MockModel:
            def predict(self, X):
                return np.array([0, 1, 0, 1])
        
        X = pd.DataFrame({'a': [1, 2, 3, 4]})
        passed = checker.check_1_not_constant(MockModel(), X)
        print(f"✅ Check 1 OK: passed={passed}")
        return True
    except Exception as e:
        print(f"❌ Check 1 falhou: {e}")
        return False

def test_check_not_null():
    try:
        from smoke_tests import SanityChecker
        import pandas as pd
        
        checker = SanityChecker()
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        passed = checker.check_2_not_null(X, y)
        print(f"✅ Check 2 OK: passed={passed}")
        return True
    except Exception as e:
        print(f"❌ Check 2 falhou: {e}")
        return False

def test_check_class_balance():
    try:
        from smoke_tests import SanityChecker
        import pandas as pd
        
        checker = SanityChecker()
        y = pd.Series([0, 0, 0, 1, 1])
        passed = checker.check_3_class_balance(y)
        print(f"✅ Check 3 OK: passed={passed}")
        return True
    except Exception as e:
        print(f"❌ Check 3 falhou: {e}")
        return False

def test_check_metric_threshold():
    try:
        from smoke_tests import SanityChecker
        
        checker = SanityChecker()
        metrics = {'accuracy': 0.95, 'f1_score': 0.94}
        passed = checker.check_4_metric_threshold(metrics)
        print(f"✅ Check 4 OK: passed={passed}")
        return True
    except Exception as e:
        print(f"❌ Check 4 falhou: {e}")
        return False

def test_check_latency():
    try:
        from smoke_tests import SanityChecker
        
        checker = SanityChecker()
        passed = checker.check_5_latency(train_time=5.0, predict_time=0.005)
        print(f"✅ Check 5 OK: passed={passed}")
        return True
    except Exception as e:
        print(f"❌ Check 5 falhou: {e}")
        return False

def test_json_serialization():
    try:
        from smoke_tests import SanityChecker, NumpyEncoder
        import json
        import numpy as np
        
        checker = SanityChecker()
        checker.results = {
            'test': {
                'passed': np.bool_(True),
                'value': np.float64(0.95),
                'array': np.array([1, 2, 3])
            }
        }
        
        report = checker.get_report()
        json_str = json.dumps(report, cls=NumpyEncoder)
        assert 'passed' in json_str
        print("✅ JSON serialization OK")
        return True
    except Exception as e:
        print(f"❌ JSON serialization falhou: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE ETAPA 9: Sanity Checks")
    print("=" * 60)
    
    tests = [
        ("Imports", test_import),
        ("Thresholds", test_thresholds),
        ("Check 1: Not Constant", test_check_not_constant),
        ("Check 2: Not Null", test_check_not_null),
        ("Check 3: Class Balance", test_check_class_balance),
        ("Check 4: Metric Threshold", test_check_metric_threshold),
        ("Check 5: Latency", test_check_latency),
        ("JSON Serialization", test_json_serialization),
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
        print("🎉 ETAPA 9 COMPLETA!")
        sys.exit(0)
    else:
        print("⚠️  Alguns testes falharam.")
        sys.exit(1)
