"""
Common helper functions for error detection across multiple languages.
Shared rules and utilities used by all language-specific detectors.
"""

import re
from typing import Dict, Optional, List, Tuple


def normalize_language(language: str) -> str:
    """Normalize language names and aliases to standard form."""
    language = language.lower().strip()
    
    # Python aliases
    if language in ['python', 'py']:
        return 'python'
    
    # JavaScript aliases
    elif language in ['javascript', 'js']:
        return 'javascript'
    
    # C++ aliases
    elif language in ['c++', 'cpp', 'cc']:
        return 'cpp'
    
    # Java aliases
    elif language in ['java']:
        return 'java'
    
    else:
        return language


def make_result(has_error: bool, error_type: Optional[str], subtype: Optional[str], 
               line: Optional[int], message: str, language: str, 
               severity: Optional[str] = None, rule_id: Optional[str] = None) -> Dict:
    """Create standardized result dictionary."""
    return {
        "has_error": has_error,
        "error_type": error_type,
        "subtype": subtype,
        "line": line,
        "message": message,
        "language": language,
        "severity": severity,
        "rule_id": rule_id
    }


def make_no_error_result(language: str) -> Dict:
    """Create standardized no-error result."""
    return {
        "has_error": False,
        "error_type": None,
        "subtype": None,
        "line": None,
        "message": "No obvious error detected",
        "language": language,
        "severity": None,
        "rule_id": None
    }


def get_lines(code: str) -> List[str]:
    """Split code into lines and return list."""
    return code.split('\n') if code else []


def find_unmatched_brackets(code: str, language: str) -> Optional[Dict]:
    """
    Detect unmatched parentheses, curly braces, and square brackets.
    Returns first unmatched bracket error found.
    """
    stack = []
    bracket_pairs = {')': '(', '}': '{', ']': '['}
    opening_brackets = set(bracket_pairs.values())
    
    lines = get_lines(code)
    
    for line_num, line in enumerate(lines, 1):
        for char_num, char in enumerate(line):
            if char in opening_brackets:
                stack.append((char, line_num, char_num))
            elif char in bracket_pairs:
                if not stack or stack[-1][0] != bracket_pairs[char]:
                    return make_result(
                        True, "syntax_error", "unmatched_brackets", line_num,
                        f"Unexpected closing '{char}' at position {char_num}",
                        language, "high", "COMMON_SYN_001"
                    )
                stack.pop()
    
    if stack:
        unclosed_bracket, line_num, char_num = stack[-1]
        return make_result(
            True, "syntax_error", "unmatched_brackets", line_num,
            f"Unclosed '{unclosed_bracket}' at position {char_num}",
            language, "high", "COMMON_SYN_002"
        )
    
    return None


def find_missing_quote(code: str, language: str) -> Optional[Dict]:
    """Detect missing closing quotes in strings."""
    lines = get_lines(code)
    
    for line_num, line in enumerate(lines, 1):
        # Skip comment lines
        stripped = line.strip()
        if stripped.startswith(('#', '//', '/*', '*')):
            continue
            
        # Count quotes - simple heuristic
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        
        # Check for odd number of quotes (unmatched)
        if single_quotes % 2 == 1 and double_quotes % 2 == 0:
            return make_result(
                True, "syntax_error", "missing_quote", line_num,
                "Unclosed single quote detected",
                language, "high", "COMMON_SYN_003"
            )
        elif double_quotes % 2 == 1 and single_quotes % 2 == 0:
            return make_result(
                True, "syntax_error", "missing_quote", line_num,
                "Unclosed double quote detected",
                language, "high", "COMMON_SYN_004"
            )
    
    return None


def find_line_number_by_pattern(lines: List[str], pattern: str) -> Optional[int]:
    """Find line number containing specific pattern."""
    for i, line in enumerate(lines, 1):
        if pattern in line:
            return i
    return None


def contains_break_statement(lines: List[str], start_line: int, end_line: int) -> bool:
    """Check if break statement exists in given line range."""
    for i in range(start_line - 1, min(end_line, len(lines))):
        if 'break' in lines[i]:
            return True
    return False


def contains_zero_check(lines: List[str], variable: str, line_num: int) -> bool:
    """Check if variable is checked for zero in surrounding lines."""
    start_check = max(0, line_num - 4)
    end_check = min(len(lines), line_num + 3)
    
    for i in range(start_check, end_check):
        if f'{variable} == 0' in lines[i] or f'{variable} != 0' in lines[i]:
            return True
    return False


def contains_null_check(lines: List[str], variable: str, line_num: int) -> bool:
    """Check if variable is checked for null/None in surrounding lines."""
    start_check = max(0, line_num - 4)
    end_check = min(len(lines), line_num + 3)
    
    for i in range(start_check, end_check):
        line = lines[i].lower()
        if any(check in line for check in [
            f'{variable.lower()} is none', f'{variable.lower()} == none',
            f'{variable.lower()} is null', f'{variable.lower()} == null',
            f'{variable.lower()} is not none', f'{variable.lower()} != none',
            f'{variable.lower()} is not null', f'{variable.lower()} != null'
        ]):
            return True
    return False


def contains_return_statement(lines: List[str], start_line: int, end_line: int) -> bool:
    """Check if return statement exists in given line range."""
    for i in range(start_line - 1, min(end_line, len(lines))):
        if 'return' in lines[i]:
            return True
    return False


def contains_delete_after_new(lines: List[str], var_name: str, line_num: int) -> bool:
    """Check if delete statement exists after new for given variable."""
    for i in range(line_num, min(line_num + 10, len(lines))):
        if f'delete {var_name}' in lines[i] or f'delete[] {var_name}' in lines[i]:
            return True
    return False


def detect_constant_condition(condition: str) -> bool:
    """Detect if condition is always true or false."""
    condition = condition.strip().lower()
    
    # Check for literal constants
    always_true = ['true', '1', 'non-zero', 'not false']
    always_false = ['false', '0', 'none', 'null']
    
    if condition in always_true:
        return True
    if condition in always_false:
        return True
    
    # Check for comparisons with literals
    if re.search(r'(==|!=)\s*(true|false|0|1|none|null)', condition):
        return True
    
    return False


def detect_assignment_in_condition(line: str) -> bool:
    """Detect assignment operator in condition (not comparison)."""
    # Remove comparison operators first
    line_no_comp = re.sub(r'==|===|!=|!==|<=|>=|<|>', '', line)
    
    # Check for single assignment
    if '=' in line_no_comp:
        # Make sure it's not in a function call or other context
        if re.search(r'(if|while|for)\s*\(.*\b[a-zA-Z_][a-zA-Z0-9_]*\s*=', line):
            return True
        if re.search(r'(if|while|for)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=', line):
            return True
    
    return False


def detect_off_by_one_patterns(lines: List[str]) -> Optional[Tuple[int, str]]:
    """Detect common off-by-one error patterns."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Pattern: i <= len(arr) or i < len(arr) + 1
        if re.search(r'\b\w+\s*<=\s*len\s*\(', stripped):
            return (i, "Using <= with array length may cause out-of-bounds access")
        
        # Pattern: arr[i+1] after range(len(arr))
        if '[i+1]' in stripped or '[i + 1]' in stripped:
            # Look for range(len(arr)) in previous lines
            for j in range(max(0, i-3), i):
                if 'range(len(' in lines[j]:
                    return (i, "Accessing arr[i+1] may cause out-of-bounds error")
    
    return None


def detect_infinite_loop_patterns(lines: List[str], language: str) -> Optional[Tuple[int, str]]:
    """Detect infinite loop patterns without break."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for infinite loop patterns
        infinite_patterns = {
            'python': ['while True', 'while(1)', 'while (1)'],
            'javascript': ['while(true)', 'for(;;)', 'while (true)'],
            'cpp': ['while(true)', 'for(;;)', 'while (true)'],
            'java': ['while(true)', 'for(;;)', 'while (true)']
        }
        
        if language in infinite_patterns:
            for pattern in infinite_patterns[language]:
                if pattern in stripped:
                    # Check if break exists in next few lines
                    if not contains_break_statement(lines, i, i + 10):
                        return (i, f"Infinite loop detected: {pattern} without break statement")
    
    return None


def detect_mixed_tabs_spaces(lines: List[str]) -> Optional[Tuple[int, str]]:
    """Detect suspicious mixed tabs and spaces usage."""
    for i, line in enumerate(lines, 1):
        if line and not line.strip():
            continue
            
        # Check if line starts with both tabs and spaces
        has_tab = line.startswith('\t')
        has_space = line.startswith(' ')
        
        # More sophisticated check for mixed usage
        leading_chars = len(line) - len(line.lstrip(' \t'))
        leading_content = line[:leading_chars]
        
        if '\t' in leading_content and ' ' in leading_content:
            return (i, "Mixed tabs and spaces in indentation")
    
    return None


def detect_empty_condition_block(lines: List[str]) -> Optional[Tuple[int, str]]:
    """Detect empty condition blocks that may indicate incomplete code."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Look for if/while/for followed by empty block
        if any(keyword in stripped for keyword in ['if', 'elif', 'while', 'for']):
            # Check next few lines for empty block
            if i < len(lines):
                next_line = lines[i].strip()
                if next_line == '{' or (next_line.endswith(':') and i + 1 < len(lines) and lines[i + 1].strip() in ['pass', ';', '{}']):
                    return (i + 1, "Empty condition block detected")
    
    return None


def get_indent_level(line: str) -> int:
    """Get the indentation level of a line."""
    return len(line) - len(line.lstrip(' \t'))


def is_comment_line(line: str, language: str) -> bool:
    """Check if line is a comment."""
    stripped = line.strip()
    
    comment_patterns = {
        'python': ['#'],
        'javascript': ['//', '/*', '*'],
        'cpp': ['//', '/*', '*'],
        'java': ['//', '/*', '*']
    }
    
    if language in comment_patterns:
        for pattern in comment_patterns[language]:
            if stripped.startswith(pattern):
                return True
    
    return False
