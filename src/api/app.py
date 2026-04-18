"""
src/api/app.py
--------------
PolyMentor FastAPI application.

Works in two modes:
  - PRE-TRAINING mode: rule-based responses (no model needed) — works right now
  - POST-TRAINING mode: full ML pipeline (after best_mentor_model.pt is created)

The mode is selected automatically based on whether the model file exists.
"""

import ast
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PolyMentor API",
    description=(
        "AI-powered coding mentor. Submit code in any supported language "
        "and receive error explanations, hints, and quality feedback."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rule-based engine (works with zero ML, zero training)
# ---------------------------------------------------------------------------

ERROR_TO_CONCEPT = {
    "syntax_error": "Python Syntax Rules",
    "logical_error": "Program Logic & Control Flow",
    "type_error": "Data Types & Type Casting",
    "off_by_one": "Loop Indexing & Array Boundaries",
    "infinite_loop": "Loop Termination Conditions",
    "null_reference": "Null Safety & Object Initialization",
    "division_by_zero": "Input Validation & Edge Cases",
    "bad_practice": "Clean Code & Readability",
    "structural_issue": "Function Design & Decomposition",
}

EXPLANATIONS = {
    "syntax_error": (
        "Your code has a syntax error — the language cannot parse its structure. "
        "Common causes: missing colon after if/for/def, using = instead of == "
        "inside a condition, unmatched brackets or quotes, wrong indentation."
    ),
    "type_error": (
        "You are using a value of the wrong type. For example, trying to add "
        "a string and a number, or calling a method that doesn't exist on that type."
    ),
    "off_by_one": (
        "Your loop starts or ends one step too early or too late. "
        "This is extremely common with range() and array indexing."
    ),
    "infinite_loop": (
        "Your loop's stopping condition is never reached. The variable that "
        "controls the loop is not changing in the right direction."
    ),
    "division_by_zero": (
        "Your code divides a number by zero, which is undefined and causes "
        "a ZeroDivisionError at runtime."
    ),
    "logical_error": (
        "Your code runs without crashing but produces the wrong result. "
        "The algorithm logic does not match the intended behaviour."
    ),
    "bad_practice": (
        "Your code works but uses patterns that make it hard to read or maintain. "
        "Consider more descriptive names and avoiding magic numbers."
    ),
    "null_reference": (
        "You are trying to use an object that is None/null. "
        "Always check that an object is initialised before calling methods on it."
    ),
    "structural_issue": (
        "The overall structure of your code could be improved. "
        "Consider breaking it into smaller, well-named functions."
    ),
}

HINTS = {
    "syntax_error": [
        "Step 1: Read the error message — it usually tells you the exact line number.",
        "Step 2: Check that line for missing colons, wrong operators (= vs ==), or unmatched brackets.",
        "Step 3: Compare your syntax against a working example of the same construct.",
    ],
    "type_error": [
        "Step 1: Print type(variable) for each value involved in the operation.",
        "Step 2: Identify which type the operation expects vs what you have.",
        "Step 3: Convert the value to the correct type (e.g. int(), str(), float()).",
    ],
    "off_by_one": [
        "Step 1: Trace through your loop manually with 3 items on paper.",
        "Step 2: Check: does your index start at 0 or 1? Where does it end?",
        "Step 3: Adjust your range by ±1 and re-trace. Does the output match now?",
    ],
    "infinite_loop": [
        "Step 1: Find your loop. What is the condition that should stop it?",
        "Step 2: Inside the loop, is that condition ever actually reached?",
        "Step 3: Add a print inside the loop to see if the stopping variable changes.",
    ],
    "division_by_zero": [
        "Step 1: Find where division happens in your code.",
        "Step 2: Can the denominator ever be 0 with the inputs you're using?",
        "Step 3: Add a guard: if denominator != 0: before dividing.",
    ],
    "logical_error": [
        "Step 1: Add print() statements before and after the suspicious section.",
        "Step 2: Write down the expected value vs what you actually got.",
        "Step 3: Trace each step of your logic on paper for a simple test case.",
    ],
    "bad_practice": [
        "Step 1: Rename single-letter variables to descriptive names.",
        "Step 2: Replace magic numbers with named constants.",
        "Step 3: Break functions longer than 20 lines into smaller helpers.",
    ],
    "null_reference": [
        "Step 1: Find where the None value is assigned or returned.",
        "Step 2: Add a check: if obj is not None: before using it.",
        "Step 3: Make sure the function that produces the value always returns something.",
    ],
    "structural_issue": [
        "Step 1: Identify blocks of code that do one specific thing.",
        "Step 2: Extract each block into its own named function.",
        "Step 3: Each function should ideally do one thing and be < 20 lines.",
    ],
}

DEFAULT_HINTS = [
    "Step 1: Re-read the code slowly, line by line.",
    "Step 2: Try running just the broken section in isolation.",
    "Step 3: Search the exact error message online.",
]


def detect_errors_rule_based(code: str, language: str) -> list:
    """Detect errors using syntax checking and pattern matching. No ML needed."""
    errors = []

    if language == "python":
        # Syntax check via Python's own parser
        try:
            ast.parse(code)
        except SyntaxError:
            errors.append("syntax_error")
            return errors  # Can't check further if syntax is broken

        # Pattern: = inside if condition (common beginner mistake)
        if re.search(r"\bif\b.*[^=!<>]=(?!=)", code):
            errors.append("syntax_error")

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Division without zero check
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    errors.append("division_by_zero")

                # while True with no break
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        has_break = any(
                            isinstance(n, ast.Break) for n in ast.walk(node)
                        )
                        if not has_break:
                            errors.append("infinite_loop")

                # range(len(...) + 1) — classic off-by-one
                if isinstance(node, ast.Call):
                    if getattr(getattr(node, "func", None), "id", "") == "range":
                        if node.args and isinstance(node.args[0], ast.BinOp):
                            if isinstance(node.args[0].op, ast.Add):
                                errors.append("off_by_one")

        except Exception:
            pass

    elif language == "javascript":
        # JS syntax pattern checks
        if re.search(r"while\s*\(.*\)\s*;", code):
            errors.append("infinite_loop")
        if re.search(r"==[^=]", code) is None and re.search(r"===", code) is None:
            if "=" in code and "if" in code:
                errors.append("syntax_error")

    # Universal: deeply nested code
    max_indent = (
        max(
            (len(line) - len(line.lstrip()))
            for line in code.split("\n")
            if line.strip()
        )
        if code.strip()
        else 0
    )
    if max_indent >= 16:
        errors.append("structural_issue")

    if not errors:
        errors.append("bad_practice")

    return list(dict.fromkeys(errors))  # deduplicate, preserve order


def score_code(code: str) -> tuple[int, list]:
    """Score code quality 0-100 and return improvement suggestions."""
    score = 100
    suggestions = []
    lines = code.strip().split("\n")

    long_lines = [l for l in lines if len(l) > 100]
    if long_lines:
        score -= len(long_lines) * 3
        suggestions.append(
            f"Shorten {len(long_lines)} line(s) that exceed 100 characters."
        )

    deep = sum(1 for l in lines if l.startswith("    " * 4))
    if deep:
        score -= deep * 5
        suggestions.append(
            "Reduce deep nesting — consider extracting inner blocks into functions."
        )

    magic = len(re.findall(r"\b(?<!\.)(?!0\b)\d{2,}\b", code))
    if magic:
        score -= magic * 2
        suggestions.append(f"Replace {magic} magic number(s) with named constants.")

    bad_vars = len(re.findall(r"\b(?![ijknxy])[a-wz]\b\s*=", code))
    if bad_vars:
        score -= bad_vars * 3
        suggestions.append("Use descriptive variable names instead of single letters.")

    if len(lines) > 15 and '"""' not in code and "'''" not in code and "#" not in code:
        score -= 10
        suggestions.append("Add comments or docstrings to explain what the code does.")

    if not suggestions:
        suggestions.append("Good job! Code is clean and readable.")

    return max(0, min(100, score)), suggestions


# ---------------------------------------------------------------------------
# ML pipeline (loaded only if model exists)
# ---------------------------------------------------------------------------

_pipeline = None
MODEL_PATH = os.environ.get(
    "POLYMENTOR_MODEL_PATH", "models_saved/best_mentor_model.pt"
)


@app.on_event("startup")
async def startup():
    global _pipeline
    if Path(MODEL_PATH).exists():
        try:
            # Only import heavy ML deps when model actually exists
            from src.inference.pipeline import PolyMentorPipeline

            _pipeline = PolyMentorPipeline.from_pretrained(MODEL_PATH)
            print(f"✅ ML pipeline loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  ML pipeline failed to load: {e}")
            print("   Falling back to rule-based mode.")
            _pipeline = None
    else:
        print("ℹ️  No trained model found — running in rule-based mode.")
        print(f"   Train a model with: bash scripts/train.sh")
        print(f"   Then restart the API.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze.")
    language: str = Field("python", description="python | javascript | cpp | java")
    level: str = Field("beginner", description="beginner | intermediate | advanced")
    num_hints: int = Field(3, ge=1, le=5)


class AnalyzeResponse(BaseModel):
    status: str
    mode: str  # "ml" or "rule_based"
    error_type: Optional[str]
    error_types: List[str]
    explanation: str
    hint: str
    hints: List[str]
    concept_taught: str
    quality_score: int
    suggestions: List[str]
    language: str
    level: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "mode": "ml" if _pipeline is not None else "rule_based",
        "model_loaded": _pipeline is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze a code snippet. Works in both rule-based and ML mode."""
    start = time.perf_counter()

    code = request.code.strip()
    language = request.language.lower()
    level = request.level.lower()

    if not code:
        return AnalyzeResponse(
            status="clean",
            mode="rule_based",
            error_type=None,
            error_types=[],
            explanation="No code provided.",
            hint="",
            hints=[],
            concept_taught="",
            quality_score=0,
            suggestions=["Paste your code to get feedback."],
            language=language,
            level=level,
            elapsed_ms=0.0,
        )

    # --- Use ML pipeline if available ---
    if _pipeline is not None:
        try:
            result = _pipeline.analyze(code=code, language=language, level=level)
            elapsed = (time.perf_counter() - start) * 1000
            return AnalyzeResponse(
                status=result.status,
                mode="ml",
                error_type=result.error_type,
                error_types=result.error_types,
                explanation=result.explanation,
                hint=result.hint,
                hints=result.hints,
                concept_taught=result.concept_taught,
                quality_score=result.quality_score,
                suggestions=getattr(result, "suggestions", []),
                language=language,
                level=level,
                elapsed_ms=round(elapsed, 2),
            )
        except Exception as e:
            print(f"ML pipeline error: {e}, falling back to rule-based")

    # --- Rule-based fallback ---
    error_types = detect_errors_rule_based(code, language)
    primary = error_types[0] if error_types else "bad_practice"

    explanation = EXPLANATIONS.get(primary, "An issue was detected in your code.")
    all_hints = HINTS.get(primary, DEFAULT_HINTS)

    # Adjust hints by level
    if level == "advanced":
        hints = all_hints[-1:]
    elif level == "intermediate":
        hints = all_hints[1:]
    else:
        hints = all_hints

    hints = hints[: request.num_hints]
    concept = ERROR_TO_CONCEPT.get(primary, "General Programming")
    quality, suggestions = score_code(code)
    elapsed = (time.perf_counter() - start) * 1000

    is_clean = error_types == ["bad_practice"] and quality >= 80

    return AnalyzeResponse(
        status="clean" if is_clean else "error_found",
        mode="rule_based",
        error_type=None if is_clean else primary,
        error_types=[] if is_clean else error_types,
        explanation=(
            "No errors detected! Code looks clean." if is_clean else explanation
        ),
        hint=hints[0] if hints else "",
        hints=hints,
        concept_taught="" if is_clean else concept,
        quality_score=quality,
        suggestions=suggestions,
        language=language,
        level=level,
        elapsed_ms=round(elapsed, 2),
    )


@app.get("/")
async def root():
    return {
        "message": "PolyMentor API is running!",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze",
        "mode": "ml" if _pipeline is not None else "rule_based",
    }
