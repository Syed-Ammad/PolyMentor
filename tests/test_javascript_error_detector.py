"""
Test cases for JavaScript ErrorDetector.
8 comprehensive tests covering syntax, logical, runtime, and semantic errors.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.javascript_error_detector import JavaScriptErrorDetector


class TestJavaScriptErrorDetector(unittest.TestCase):
    """Test cases for JavaScript error detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = JavaScriptErrorDetector()
    
    def test_syntax_error_missing_semicolon(self):
        """Test syntax error: missing semicolon."""
        code = "let x = 5\nlet y = 10;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "missing_semicolon")
        self.assertEqual(result["line"], 1)
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "JS_SYN_002")
    
    def test_syntax_error_unmatched_brackets(self):
        """Test syntax error: unmatched brackets."""
        code = "function hello() {\n    console.log('Hello World';\n}"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "unmatched_brackets")
        self.assertEqual(result["severity"], "high")
    
    def test_runtime_error_division_by_zero(self):
        """Test runtime error: division by zero."""
        code = "let result = 100 / 0;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "division_by_zero_risk")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JS_RT_001")
    
    def test_runtime_error_undefined_variable(self):
        """Test runtime error: undefined variable."""
        code = "console.log(undefinedVar);"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "undefined_variable_risk")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "JS_RT_002")
    
    def test_logical_error_assignment_in_condition(self):
        """Test logical error: assignment in condition."""
        code = "if (x = 5) {\n    console.log('x is 5');\n}"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "assignment_in_condition")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JS_LOG_001")
    
    def test_logical_error_loose_equality(self):
        """Test logical error: loose equality."""
        code = "if (x == 5) {\n    console.log('x is 5');\n}"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "loose_equality_warning")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "JS_LOG_002")
    
    def test_semantic_error_var_usage(self):
        """Test semantic error: var usage."""
        code = "var x = 5;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "var_usage_warning")
        self.assertEqual(result["severity"], "low")
        self.assertEqual(result["rule_id"], "JS_SEM_001")
    
    def test_correct_code_no_error(self):
        """Test correct code with no errors."""
        code = """const numbers = [1, 2, 3, 4, 5];
let sum = 0;

for (const num of numbers) {
    sum += num;
}

console.log(`Sum: ${sum}`);"""
        result = self.detector.detect(code)
        
        self.assertFalse(result["has_error"])
        self.assertEqual(result["error_type"], None)
        self.assertEqual(result["subtype"], None)
        self.assertEqual(result["line"], None)
        self.assertEqual(result["message"], "No obvious error detected")
        self.assertEqual(result["language"], "javascript")


if __name__ == '__main__':
    unittest.main()
