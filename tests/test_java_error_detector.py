"""
Test cases for Java ErrorDetector.
8 comprehensive tests covering syntax, logical, runtime, and semantic errors.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.java_error_detector import JavaErrorDetector


class TestJavaErrorDetector(unittest.TestCase):
    """Test cases for Java error detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = JavaErrorDetector()
    
    def test_syntax_error_missing_semicolon(self):
        """Test syntax error: missing semicolon."""
        code = """int x = 5
int y = 10;"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "missing_semicolon")
        self.assertEqual(result["line"], 1)
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_SYN_002")
    
    def test_syntax_error_unmatched_brackets(self):
        """Test syntax error: unmatched brackets."""
        code = "public class Test {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "missing_brace")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_SYN_005")
    
    def test_runtime_error_division_by_zero(self):
        """Test runtime error: division by zero."""
        code = "int result = 100 / 0;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "division_by_zero_risk")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_RT_001")
    
    def test_runtime_error_null_pointer_risk(self):
        """Test runtime error: null pointer risk."""
        code = """String str = null;
int length = str.length();"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "null_pointer_risk")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_RT_002")
    
    def test_logical_error_assignment_in_condition(self):
        """Test logical error: assignment in condition."""
        code = "if (x = 5) {\n    System.out.println(\"x is 5\");\n}"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "assignment_in_condition")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_LOG_001")
    
    def test_logical_error_infinite_loop(self):
        """Test logical error: infinite loop."""
        code = """while (true) {
    System.out.println("Running forever");
}"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "infinite_loop")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_LOG_003")
    
    def test_semantic_error_string_comparison(self):
        """Test semantic error: string comparison with ==."""
        code = """String str1 = "hello";
String str2 = "hello";
if (str1 == str2) {
    System.out.println("Equal");
}"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "string_comparison_with_double_equals")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_SEM_001")
    
    def test_correct_code_no_error(self):
        """Test correct code with no errors."""
        code = """import java.util.ArrayList;
import java.util.List;

public class Calculator {
    public static void main(String[] args) {
        List<Integer> numbers = new ArrayList<>();
        
        for (int i = 1; i <= 10; i++) {
            numbers.add(i * 2);
        }
        
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        
        System.out.println("Sum: " + sum);
    }
}"""
        result = self.detector.detect(code)
        
        self.assertFalse(result["has_error"])
        self.assertEqual(result["error_type"], None)
        self.assertEqual(result["subtype"], None)
        self.assertEqual(result["line"], None)
        self.assertEqual(result["message"], "No obvious error detected")
        self.assertEqual(result["language"], "java")


if __name__ == '__main__':
    unittest.main()
