"""
Test cases for C++ ErrorDetector.
12 comprehensive tests covering syntax, logical, runtime, and semantic errors.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cpp_error_detector import CppErrorDetector


class TestCppErrorDetector(unittest.TestCase):
    """Test cases for C++ error detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = CppErrorDetector()
    
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
        self.assertEqual(result["rule_id"], "CPP_SYN_002")
    
    def test_syntax_error_unmatched_brackets(self):
        """Test syntax error: unmatched brackets."""
        code = "int main() {\n    cout << \"Hello World\" << endl;\n"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["subtype"], "missing_brace")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_SYN_005")
    
    def test_runtime_error_division_by_zero(self):
        """Test runtime error: division by zero."""
        code = "int result = 100 / 0;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "division_by_zero_risk")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_RT_001")
    
    def test_runtime_error_null_pointer_dereference(self):
        """Test runtime error: null pointer dereference."""
        code = """int* ptr = nullptr;
cout << *ptr;"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "null_pointer_dereference_risk")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_RT_004")
    
    def test_runtime_error_memory_leak(self):
        """Test runtime error: memory leak risk."""
        code = "int* arr = new int[10];"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "memory_leak_risk")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "CPP_RT_008")
    
    def test_runtime_error_unsafe_string_function(self):
        """Test runtime error: unsafe C string function."""
        code = """char dest[50];
strcpy(dest, source);"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "runtime_error")
        self.assertEqual(result["subtype"], "unsafe_c_string_function")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_RT_018")
    
    def test_logical_error_assignment_in_condition(self):
        """Test logical error: assignment in condition."""
        code = "if (x = 5) {\n    cout << \"x is 5\" << endl;\n}"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "assignment_in_condition")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_LOG_001")
    
    def test_logical_error_infinite_loop(self):
        """Test logical error: infinite loop."""
        code = """while (true) {
    cout << \"Running forever\" << endl;
}"""
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "infinite_loop")
        self.assertEqual(result["severity"], "high")
        self.assertEqual(result["rule_id"], "CPP_LOG_003")
    
    def test_semantic_error_using_namespace_std(self):
        """Test semantic error: using namespace std."""
        code = "using namespace std;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "using_namespace_std_warning")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "CPP_SEM_001")
    
    def test_semantic_error_magic_number(self):
        """Test semantic error: magic number."""
        code = "int size = 1000;"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "logical_error")
        self.assertEqual(result["subtype"], "use_of_magic_number")
        self.assertEqual(result["severity"], "low")
        self.assertEqual(result["rule_id"], "CPP_LOG_010")
    
    def test_semantic_error_compare_floats_directly(self):
        """Test semantic error: comparing floats directly."""
        code = "float a = 0.1 + 0.2;\nif (a == 0.3) { /* ... */ }"
        result = self.detector.detect(code)
        
        self.assertTrue(result["has_error"])
        self.assertEqual(result["error_type"], "semantic_error")
        self.assertEqual(result["subtype"], "compare_floats_directly")
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["rule_id"], "CPP_SEM_005")
    
    def test_correct_code_no_error(self):
        """Test correct code with no errors."""
        code = """#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

int main() {
    const int SIZE = 10;
    vector<int> numbers(SIZE);
    
    for (int i = 0; i < SIZE; ++i) {
        numbers[i] = i * 2;
    }
    
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    
    cout << "Sum: " << sum << endl;
    return 0;
}"""
        result = self.detector.detect(code)
        
        self.assertFalse(result["has_error"])
        self.assertEqual(result["error_type"], None)
        self.assertEqual(result["subtype"], None)
        self.assertEqual(result["line"], None)
        self.assertEqual(result["message"], "No obvious error detected")
        self.assertEqual(result["language"], "cpp")


if __name__ == '__main__':
    unittest.main()
