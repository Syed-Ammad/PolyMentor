"""
Test cases for Python ErrorDetector.
8 comprehensive tests covering syntax, logical, runtime, and semantic errors.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.python_error_detector import PythonErrorDetector


class TestPythonErrorDetector(unittest.TestCase):
    """Test cases for Python error detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = PythonErrorDetector()
    
    def test_syntax_error_missing_colon(self):
        """Test syntax error: missing colon."""
        code = "if x > 5\n    print('x is greater')"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "missing_colon")
        self.assertEqual(result["line"], 1)
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "PY_SYN_002")
    
    def test_syntax_error_invalid_syntax(self):
        """Test syntax error: invalid syntax from ast.parse."""
        code = "def hello()\n    print('Hello World')"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "python_syntax_error")
        self.assertEqual(result["line"], 1)
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "PY_SYN_001")
    
    def test_runtime_error_division_by_zero(self):
        """Test runtime error: division by zero risk."""
        code = "result = 100 / x"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "division_by_zero_risk")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "PY_RT_001")
    
    def test_runtime_error_key_error_risk(self):
        """Test runtime error: key error risk."""
        code = "value = my_dict['key']"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "key_error_risk")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "PY_RT_003")
    
    def test_logical_error_infinite_loop(self):
        """Test logical error: infinite loop."""
        code = """while True:
    print('This will run forever')
    print('Still running')"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "infinite_loop")
        self.assertEqual(result["line"], 1)
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "PY_LOG_003")
    
    def test_semantic_error_shadowing_builtin(self):
        """Test semantic error: shadowing builtin."""
        code = "list = [1, 2, 3]"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "shadowing_builtin")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "PY_SEM_002")
    
    def test_semantic_error_bare_except(self):
        """Test semantic error: bare except."""
        code = """try:
    risky_operation()
except:
    pass"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "bare_except")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "PY_SEM_005")
    
    def test_correct_code_no_error(self):
        """Test correct code with no errors."""
        code = """def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci of 10 is {result}")"""
        result = self.detector.detect(code)
        
        self.assertFalse(result["has_error"])
        self.assertEqual(result["error_type"], None)
        self.assertEqual(result["subtype"], None)
        self.assertEqual(result["line"], None)
        self.assertEqual(result["message"], "No obvious error detected")
        self.assertEqual(result["language"], "python")


if __name__ == '__main__':
    unittest.main()
