"""
Error Detection Module for PolyMentor AI Coding Assistant

A rule-based error detection system that analyzes code in multiple languages
and detects syntax errors, logical errors, runtime errors, semantic errors, and warnings.
"""

from .common_rules import normalize_language
from .python_error_detector import PythonErrorDetector
from .javascript_error_detector import JavaScriptErrorDetector
from .cpp_error_detector import CppErrorDetector
from .java_error_detector import JavaErrorDetector


class ErrorDetector:
    """
    Main router for language-specific error detectors.
    
    Supports: Python, JavaScript, Java, C++
    Detects: Syntax errors, Logical errors, Runtime errors, Semantic errors, Warnings
    """
    
    def __init__(self):
        """Initialize language-specific detectors."""
        self.python_detector = PythonErrorDetector()
        self.javascript_detector = JavaScriptErrorDetector()
        self.cpp_detector = CppErrorDetector()
        self.java_detector = JavaErrorDetector()
    
    def detect(self, code: str, language: str) -> dict:
        """
        Main method to detect errors in code.
        
        Args:
            code: Source code as string
            language: Programming language (python, py, javascript, js, c++, cpp, cc, java)
            
        Returns:
            Dict containing error information with format:
            {
                "has_error": bool,
                "error_type": "syntax_error" or "logical_error" or "runtime_error" or "semantic_error" or "warning" or None,
                "subtype": str or None,
                "line": int or None,
                "message": str,
                "language": str,
                "severity": "low" or "medium" or "high" or None,
                "rule_id": str or None
            }
        """
        # Normalize language name
        normalized_lang = normalize_language(language)
        
        # Route to appropriate language detector
        if normalized_lang == "python":
            return self.python_detector.detect(code)
        elif normalized_lang == "javascript":
            return self.javascript_detector.detect(code)
        elif normalized_lang == "cpp":
            return self.cpp_detector.detect(code)
        elif normalized_lang == "java":
            return self.java_detector.detect(code)
        else:
            return {
                "has_error": False,
                "error_type": None,
                "subtype": None,
                "line": None,
                "message": f"Unsupported language: {language}",
                "language": language,
                "severity": None,
                "rule_id": None
            }
