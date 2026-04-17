"""
Integration tests for main ErrorDetector router.
6 comprehensive tests covering all supported languages and routing.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.error_detector import ErrorDetector


class TestErrorDetectorIntegration(unittest.TestCase):
    """Integration tests for main ErrorDetector router."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ErrorDetector()
    
    def test_python_syntax_error_routing(self):
        """Test Python syntax error routing."""
        code = "def hello()\n    print('Hello World')"
        result = self.detector.detect(code, "python")
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "python_syntax_error")
        self.assertEqual(result["language"], "python")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "PY_SYN_001")
    
    def test_javascript_logical_error_routing(self):
        """Test JavaScript logical error routing."""
        code = "if (x = 5) {\n    console.log('x is 5');\n}"
        result = self.detector.detect(code, "javascript")
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "assignment_in_condition")
        self.assertEqual(result["language"], "javascript")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JS_LOG_001")
    
    def test_cpp_runtime_error_routing(self):
        """Test C++ runtime error routing."""
        code = "int result = 100 / 0;"
        result = self.detector.detect(code, "cpp")
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "division_by_zero_risk")
        self.assertEqual(result["language"], "cpp")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_RT_001")
    
    def test_java_semantic_error_routing(self):
        """Test Java semantic error routing."""
        code = """String str1 = "hello";
String str2 = "hello";
if (str1 == str2) {
    System.out.println("Equal");
}"""
        result = self.detector.detect(code, "java")
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "string_comparison_with_double_equals")
        self.assertEqual(result["language"], "java")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "JAVA_SEM_001")
    
    def test_language_aliases(self):
        """Test language name normalization and aliases."""
        # Test Python aliases
        result_py = self.detector.detect("print('hello')", "py")
        self.assertEqual(result_py["language"], "python")
        
        # Test JavaScript aliases
        result_js = self.detector.detect("console.log('hello')", "js")
        self.assertEqual(result_js["language"], "javascript")
        
        # Test C++ aliases
        result_cpp1 = self.detector.detect("int x = 5;", "c++")
        result_cpp2 = self.detector.detect("int x = 5;", "cc")
        self.assertEqual(result_cpp1["language"], "cpp")
        self.assertEqual(result_cpp2["language"], "cpp")
    
    def test_correct_code_no_errors(self):
        """Test correct code across all languages returns no errors."""
        # Python
        py_code = """def add(a, b):
    return a + b

result = add(5, 10)
print(f"Result: {result}")"""
        result_py = self.detector.detect(py_code, "python")
        self.assertFalse(result_py["has_error"])
        
        # JavaScript
        js_code = """function add(a, b) {
    return a + b;
}

const result = add(5, 10);
console.log(`Result: ${result}`);"""
        result_js = self.detector.detect(js_code, "javascript")
        self.assertFalse(result_js["has_error"])
        
        # C++
        cpp_code = """int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 10);
    return 0;
}"""
        result_cpp = self.detector.detect(cpp_code, "cpp")
        self.assertFalse(result_cpp["has_error"])
        
        # Java
        java_code = """public class Calculator {
    public static int add(int a, int b) {
        return a + b;
    }
    
    public static void main(String[] args) {
        int result = add(5, 10);
        System.out.println("Result: " + result);
    }
}"""
        result_java = self.detector.detect(java_code, "java")
        self.assertFalse(result_java["has_error"])


if __name__ == '__main__':
    unittest.main()
