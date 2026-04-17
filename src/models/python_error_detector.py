"""
Python-specific error detection rules.
Implements 29 rules for syntax, logical, runtime, and semantic errors.
"""

import ast
import re
from typing import Dict, Optional, List
from .common_rules import (
    make_result, make_no_error_result, get_lines, find_unmatched_brackets,
    find_missing_quote, contains_break_statement, contains_zero_check,
    contains_null_check, contains_return_statement, detect_assignment_in_condition,
    detect_off_by_one_patterns, detect_infinite_loop_patterns,
    detect_mixed_tabs_spaces, detect_empty_condition_block,
    get_indent_level, is_comment_line
)


class PythonErrorDetector:
    """Python-specific error detector implementing 29 rules."""
    
    def detect(self, code: str) -> Dict:
        """
        Detect errors in Python code using rule-based approach.
        Returns first high-confidence error found.
        """
        lines = get_lines(code)
        
        # Priority order: Syntax -> Runtime -> Logical -> Semantic -> Warning
        
        # SYNTAX ERRORS
        result = self._check_syntax_errors(code, lines)
        if result and result["has_error"]:
            return result
        
        # RUNTIME ERRORS
        result = self._check_runtime_errors(code, lines)
        if result and result["has_error"]:
            return result
        
        # LOGICAL ERRORS
        result = self._check_logical_errors(code, lines)
        if result and result["has_error"]:
            return result
        
        # SEMANTIC ERRORS
        result = self._check_semantic_errors(code, lines)
        if result and result["has_error"]:
            return result
        
        # WARNINGS
        result = self._check_warnings(code, lines)
        if result and result["has_error"]:
            return result
        
        return make_no_error_result("python")
    
    def _check_syntax_errors(self, code: str, lines: List[str]) -> Optional[Dict]:
        """Check Python syntax errors (Rules 1-7)."""
        
        # Rule 1: Python syntax error from ast.parse
        try:
            ast.parse(code)
        except SyntaxError as e:
            return make_result(
                True, "syntax_error", "python_syntax_error", e.lineno,
                str(e), "python", "high", "PY_SYN_001"
            )
        
        # Rule 2: Missing colon after control statements
        colon_keywords = {'if', 'elif', 'else', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with'}
        for i, line in enumerate(lines, 1):
            if is_comment_line(line, "python"):
                continue
                
            stripped = line.strip()
            words = stripped.split()
            if words and words[0] in colon_keywords and ':' not in line:
                return make_result(
                    True, "syntax_error", "missing_colon", i,
                    f"Missing colon after '{words[0]}' statement",
                    "python", "high", "PY_SYN_002"
                )
        
        # Rule 3: Indentation error heuristic
        prev_indent = 0
        for i, line in enumerate(lines, 1):
            if not line.strip() or is_comment_line(line, "python"):
                continue
                
            current_indent = get_indent_level(line)
            # Check for inconsistent indentation (mix of tabs and spaces)
            if '\t' in line[:current_indent] and ' ' in line[:current_indent]:
                return make_result(
                    True, "syntax_error", "indentation_error", i,
                    "Mixed tabs and spaces in indentation",
                    "python", "high", "PY_SYN_003"
                )
            
            # Check for sudden indentation changes without reason
            if current_indent > prev_indent + 4:  # More than one level jump
                return make_result(
                    True, "syntax_error", "indentation_error", i,
                    "Unexpected indentation increase",
                    "python", "medium", "PY_SYN_004"
                )
            prev_indent = current_indent
        
        # Rule 4: Unmatched brackets
        result = find_unmatched_brackets(code, "python")
        if result:
            return result
        
        # Rule 5: Missing parenthesis in print or call
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('print ') and '(' not in stripped:
                return make_result(
                    True, "syntax_error", "missing_parenthesis", i,
                    "Missing parentheses in print statement",
                    "python", "high", "PY_SYN_005"
                )
        
        # Rule 6: Missing quote
        result = find_missing_quote(code, "python")
        if result:
            return result
        
        # Rule 7: Invalid assignment in condition
        for i, line in enumerate(lines, 1):
            if detect_assignment_in_condition(line):
                return make_result(
                    True, "syntax_error", "invalid_assignment_in_condition", i,
                    "Invalid assignment in condition statement",
                    "python", "high", "PY_SYN_007"
                )
        
        return None
    
    def _check_runtime_errors(self, code: str, lines: List[str]) -> Optional[Dict]:
        """Check Python runtime error risks (Rules 17-22)."""
        
        # Rule 17: Division by zero risk
        for i, line in enumerate(lines, 1):
            div_match = re.search(r'/\s*([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if div_match:
                variable = div_match.group(1)
                if not contains_zero_check(lines, variable, i):
                    return make_result(
                        True, "runtime_error", "division_by_zero_risk", i,
                        f"Possible division by zero: variable '{variable}' not checked for zero",
                        "python", "medium", "PY_RT_001"
                    )
        
        # Rule 18: Index out of bounds risk
        for i, line in enumerate(lines, 1):
            if re.search(r'\b\w+\[.*\+\s*\w+.*\]', line):
                return make_result(
                    True, "runtime_error", "index_out_of_bounds_risk", i,
                    "Possible index out of bounds error with array access",
                    "python", "medium", "PY_RT_002"
                )
        
        # Rule 19: Key error risk for dict access
        for i, line in enumerate(lines, 1):
            if re.search(r'\b\w+\[[\'"].*[\'"]\]', line):
                # Check if key exists check is present
                dict_var = re.search(r'(\w+)\[', line)
                if dict_var:
                    var_name = dict_var.group(1)
                    if not any(f'{var_name}.get(' in lines[j] or f'in {var_name}' in lines[j] 
                              for j in range(max(0, i-3), min(i+3, len(lines)))):
                        return make_result(
                            True, "runtime_error", "key_error_risk", i,
                            f"Possible KeyError: direct dictionary access without existence check",
                            "python", "medium", "PY_RT_003"
                        )
        
        # Rule 20: None-like usage risk
        for i, line in enumerate(lines, 1):
            if re.search(r'\b\w+\s*\.\s*\w+\s*\(', line):
                var_match = re.search(r'(\w+)\s*\.', line)
                if var_match:
                    var_name = var_match.group(1)
                    if not contains_null_check(lines, var_name, i):
                        return make_result(
                            True, "runtime_error", "null_like_usage_risk", i,
                            f"Possible NoneType error: variable '{var_name}' may be None",
                            "python", "medium", "PY_RT_004"
                        )
        
        # Rule 21: Uninitialized variable risk
        for i, line in enumerate(lines, 1):
            if re.search(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=.*\+\s*[a-zA-Z_][a-zA-Z0-9_]*', line):
                # Look for variable being used before assignment
                var_match = re.search(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
                if var_match:
                    var_name = var_match.group(1)
                    # Check if variable was used before this line
                    for j in range(0, i-1):
                        if var_name in lines[j] and f'{var_name} =' not in lines[j]:
                            return make_result(
                                True, "runtime_error", "uninitialized_variable_risk", i,
                                f"Variable '{var_name}' may be used before initialization",
                                "python", "medium", "PY_RT_005"
                            )
        
        # Rule 22: Recursion without base case
        func_name = None
        has_base_case = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Detect function definition
            if stripped.startswith('def '):
                func_name = stripped.split('(')[0].replace('def ', '')
                has_base_case = False
                continue
            
            # Check for recursive call
            if func_name and f'{func_name}(' in line:
                # Look for base case in previous lines
                for j in range(max(0, i-5), i):
                    if any(keyword in lines[j].lower() for keyword in ['if', 'return', 'base']):
                        has_base_case = True
                        break
                
                if not has_base_case:
                    return make_result(
                        True, "runtime_error", "recursion_without_base_case", i,
                        f"Recursive function '{func_name}' may lack base case",
                        "python", "medium", "PY_RT_006"
                    )
        
        return None
    
    def _check_logical_errors(self, code: str, lines: List[str]) -> Optional[Dict]:
        """Check Python logical errors (Rules 8-16)."""
        
        # Rule 8: Assignment in condition
        for i, line in enumerate(lines, 1):
            if detect_assignment_in_condition(line):
                return make_result(
                    True, "logical_error", "assignment_in_condition", i,
                    "Possible assignment instead of comparison in condition",
                    "python", "high", "PY_LOG_001"
                )
        
        # Rule 9: Off-by-one errors
        off_by_one = detect_off_by_one_patterns(lines)
        if off_by_one:
            line_num, message = off_by_one
            return make_result(
                True, "logical_error", "off_by_one", line_num,
                message, "python", "medium", "PY_LOG_002"
            )
        
        # Rule 10: Infinite loop
        infinite_loop = detect_infinite_loop_patterns(lines, "python")
        if infinite_loop:
            line_num, message = infinite_loop
            return make_result(
                True, "logical_error", "infinite_loop", line_num,
                message, "python", "high", "PY_LOG_003"
            )
        
        # Rule 11: Unreachable branch
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if 'if True:' in stripped or 'if False:' in stripped:
                return make_result(
                    True, "logical_error", "unreachable_branch", i,
                    "Condition is always True/False - unreachable code",
                    "python", "medium", "PY_LOG_004"
                )
        
        # Rule 12: Wrong comparison operator
        for i, line in enumerate(lines, 1):
            if re.search(r'\bis\s+(0|1|[0-9]+\.?[0-9]*)\b', line):
                return make_result(
                    True, "logical_error", "wrong_comparison_operator", i,
                    "Using 'is' with numbers - use '==' instead",
                    "python", "medium", "PY_LOG_005"
                )
            
            if re.search(r'\bis\s+[\'"][^\'"]*[\'"]\b', line):
                return make_result(
                    True, "logical_error", "wrong_comparison_operator", i,
                    "Using 'is' with strings - use '==' instead",
                    "python", "medium", "PY_LOG_006"
                )
        
        # Rule 13: Mutation during iteration
        for i, line in enumerate(lines, 1):
            if re.search(r'for\s+\w+\s+in\s+(\w+):', line):
                list_name = re.search(r'for\s+\w+\s+in\s+(\w+):', line).group(1)
                # Look for list modification in next few lines
                for j in range(i, min(i+10, len(lines))):
                    if re.search(f'{list_name}\\.(append|remove|pop|insert|clear)', lines[j]):
                        return make_result(
                            True, "logical_error", "mutation_during_iteration", j+1,
                            f"Modifying list '{list_name}' during iteration",
                            "python", "medium", "PY_LOG_007"
                        )
        
        # Rule 14: Wrong accumulator update
        for i, line in enumerate(lines, 1):
            if re.search(r'^\s*\w+\s*=\s*\w+\s*$', line):
                # Check if inside loop
                for j in range(max(0, i-5), i):
                    if 'for ' in lines[j] or 'while ' in lines[j]:
                        return make_result(
                            True, "logical_error", "wrong_accumulator_update", i,
                            "Wrong accumulator update - did you mean '+='?",
                            "python", "medium", "PY_LOG_008"
                        )
        
        # Rule 15: Loop variable not updated
        for i, line in enumerate(lines, 1):
            if re.search(r'while\s+(\w+):', line):
                var_name = re.search(r'while\s+(\w+):', line).group(1)
                # Check if variable is updated in loop body
                has_update = False
                for j in range(i, min(i+10, len(lines))):
                    if re.search(f'{var_name}\\s*[-+*/]=', lines[j]) or f'{var_name} = ' in lines[j]:
                        has_update = True
                        break
                
                if not has_update:
                    return make_result(
                        True, "logical_error", "loop_variable_not_updated", i,
                        f"Loop variable '{var_name}' is never updated",
                        "python", "medium", "PY_LOG_009"
                    )
        
        # Rule 16: Always true/false comparison
        for i, line in enumerate(lines, 1):
            if re.search(r'==\s*(True|False|None)', line):
                return make_result(
                    True, "logical_error", "always_true_or_false_comparison", i,
                    "Comparison with literal may always be true/false",
                    "python", "low", "PY_LOG_010"
                )
        
        return None
    
    def _check_semantic_errors(self, code: str, lines: List[str]) -> Optional[Dict]:
        """Check Python semantic errors (Rules 23-29)."""
        
        # Rule 23: Unused variable warning
        assigned_vars = {}
        used_vars = set()
        
        for i, line in enumerate(lines, 1):
            # Track variable assignments
            assign_match = re.search(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
            if assign_match:
                var_name = assign_match.group(1)
                if var_name not in assigned_vars:
                    assigned_vars[var_name] = i
            
            # Track variable usage
            for var_name in assigned_vars:
                if var_name in line and f'{var_name} =' not in line:
                    used_vars.add(var_name)
        
        # Find unused variables
        for var_name, line_num in assigned_vars.items():
            if var_name not in used_vars and var_name not in ['_', 'result']:
                return make_result(
                    True, "semantic_error", "unused_variable", line_num,
                    f"Variable '{var_name}' is assigned but never used",
                    "python", "low", "PY_SEM_001"
                )
        
        # Rule 24: Shadowing builtin warning
        builtins = {'list', 'dict', 'str', 'int', 'sum', 'max', 'min', 'input', 'len', 'range', 'print'}
        for i, line in enumerate(lines, 1):
            assign_match = re.search(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
            if assign_match:
                var_name = assign_match.group(1)
                if var_name in builtins:
                    return make_result(
                        True, "semantic_error", "shadowing_builtin", i,
                        f"Variable '{var_name}' shadows built-in function",
                        "python", "medium", "PY_SEM_002"
                    )
        
        # Rule 25: Mutable default argument warning
        for i, line in enumerate(lines, 1):
            if re.search(r'def\s+\w+\([^)]*=\s*[\[\{]', line):
                return make_result(
                    True, "semantic_error", "mutable_default_argument", i,
                    "Mutable default argument detected - use None instead",
                    "python", "medium", "PY_SEM_003"
                )
        
        # Rule 26: Comparing string and number warning
        for i, line in enumerate(lines, 1):
            if re.search(r'[\'"][^\'"]*[\'"]\s*(==|!=|<|>|<=|>=)\s*\d+', line) or \
               re.search(r'\d+\s*(==|!=|<|>|<=|>=)\s*[\'"][^\'"]*[\'"]', line):
                return make_result(
                    True, "semantic_error", "comparing_string_and_number", i,
                    "Comparing string and number may cause unexpected behavior",
                    "python", "medium", "PY_SEM_004"
                )
        
        # Rule 27: Bare except warning
        for i, line in enumerate(lines, 1):
            if re.search(r'except\s*:', line):
                return make_result(
                    True, "semantic_error", "bare_except", i,
                    "Bare except clause - specify exception type",
                    "python", "medium", "PY_SEM_005"
                )
        
        # Rule 28: Broad except Exception warning
        for i, line in enumerate(lines, 1):
            if re.search(r'except\s+Exception\s*:', line):
                return make_result(
                    True, "semantic_error", "broad_except", i,
                    "Broad except Exception - catch specific exceptions",
                    "python", "low", "PY_SEM_006"
                )
        
        # Rule 29: Duplicate condition warning
        conditions = {}
        for i, line in enumerate(lines, 1):
            if_match = re.search(r'if\s+(.+?):', line)
            if if_match:
                condition = if_match.group(1).strip()
                if condition in conditions:
                    return make_result(
                        True, "semantic_error", "duplicate_condition", i,
                        f"Duplicate condition: '{condition}' already checked on line {conditions[condition]}",
                        "python", "low", "PY_SEM_007"
                    )
                conditions[condition] = i
        
        return None
    
    def _check_warnings(self, code: str, lines: List[str]) -> Optional[Dict]:
        """Check Python warnings (additional warning rules)."""
        
        # Mixed tabs and spaces warning
        mixed_indent = detect_mixed_tabs_spaces(lines)
        if mixed_indent:
            line_num, message = mixed_indent
            return make_result(
                True, "warning", "mixed_tabs_spaces", line_num,
                message, "python", "low", "PY_WARN_001"
            )
        
        # Empty condition block warning
        empty_block = detect_empty_condition_block(lines)
        if empty_block:
            line_num, message = empty_block
            return make_result(
                True, "warning", "empty_condition_block", line_num,
                message, "python", "low", "PY_WARN_002"
            )
        
        return None
