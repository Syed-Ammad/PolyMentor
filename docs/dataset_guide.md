# PolyMentor — Dataset Guide

> How data is sourced, collected, labeled, and processed into training-ready form.

---

## Overview

PolyMentor requires three distinct types of training signal:

1. **Error detection signal** — Code snippets labeled with one or more error types.
2. **Explanation signal** — Error + plain-English explanation pairs that teach the *why*.
3. **Hint signal** — Problem + progressive hint sequences that guide without giving away answers.

No single public dataset covers all three. PolyMentor combines five sources, normalises them through the data pipeline, and stores the result in a unified JSON schema.

---

## Data Sources

### CodeNet (IBM)

- **Location:** `data/raw/code_datasets/`
- **URL:** https://github.com/IBM/Project_CodeNet
- **Content:** 14 million code samples across 55 languages, each annotated with execution outcome (accepted, wrong answer, runtime error, time limit exceeded, etc.)
- **Used for:** Error detection pre-training. Execution outcomes map to broad error categories and provide large-scale signal for the CodeBERT classifier.
- **Notes:** Only C++, Python, JavaScript, and Java samples are retained. Samples without an execution outcome label are discarded.

---

### Stack Overflow Dump

- **Location:** `data/raw/error_samples/`
- **URL:** https://archive.org/details/stackexchange
- **Content:** Questions tagged with error-related tags, paired with their accepted answers.
- **Used for:** Explanation model training. The question body provides the buggy code and error context; the accepted answer provides the explanation text.
- **Notes:** Requires significant cleaning — code blocks must be extracted, non-code answers filtered, and duplicate questions deduplicated. `cleaner.py` handles this automatically.

---

### LeetCode / HackerRank Problem Sets

- **Location:** `data/raw/programming_questions/`
- **Content:** Competitive programming problems paired with buggy student submissions.
- **Used for:** Hint system training. The problem statement defines the intended behaviour; the buggy submission is the starting point; the correct solution anchors the hint sequence.
- **Notes:** Hint sequences are constructed semi-automatically by diffing the buggy and correct solutions and ordering the differences by conceptual proximity.

---

### ManyBugs / IntroClass

- **Location:** `data/raw/error_samples/`
- **URL:** https://repairbenchmarks.cs.umass.edu/
- **Content:** Real-world, labeled C bug datasets. ManyBugs covers larger programs; IntroClass covers beginner-level assignments.
- **Used for:** Error classification. High-quality labels make these datasets particularly valuable for training the multi-label classifier on rare error types.
- **Notes:** C samples are partially reusable for C++ training via the shared AST structure but are not used for Python or JavaScript classification.

---

### Custom Collected

- **Location:** `data/raw/error_samples/` and `data/raw/programming_questions/`
- **Content:** Beginner code submissions collected internally, paired with teacher-written explanations.
- **Used for:** Both explanation training and hint training. Teacher-written explanations are the highest-quality signal in the dataset because they are explicitly pedagogical rather than technically corrective.
- **Notes:** This dataset is small but high-value. It is weighted more heavily during fine-tuning of the explanation model.

---

## Schema

All sources are normalised into a single record schema before splitting:

```json
{
  "id": "so_12345678",
  "language": "python",
  "code": "for i in range(10):\n    if i = 5:\n        break",
  "error_types": ["syntax_error", "assignment_in_condition"],
  "error_locations": [2],
  "explanation": "The = operator assigns a value rather than comparing one. Use == to test equality.",
  "concept": "comparison_operators",
  "difficulty": "beginner",
  "hints": [
    "Think about what operator you use to check if two values are equal.",
    "Python uses = for assignment and == for comparison — which do you need here?",
    "Replace = with == inside the if condition."
  ],
  "quality_score": null,
  "source": "stackoverflow"
}
```

Fields that cannot be derived from a given source are set to `null` and excluded from the corresponding training objective.

---

## Label System

### Error Types (`data/labels/error_types.json`)

The error taxonomy has two levels: category and specific type. The classifier outputs multi-label predictions at the specific-type level.

| Category | Specific Types |
|---|---|
| `syntax_error` | `missing_colon`, `unmatched_bracket`, `wrong_indentation`, `assignment_in_condition`, `missing_return` |
| `logical_error` | `off_by_one`, `wrong_condition`, `infinite_loop`, `inverted_logic`, `wrong_variable` |
| `type_error` | `wrong_type_passed`, `implicit_conversion_bug`, `null_dereference` |
| `runtime_pattern` | `division_by_zero`, `out_of_bounds`, `stack_overflow` |
| `bad_practice` | `magic_number`, `deeply_nested`, `unused_variable`, `god_function` |
| `structural_issue` | `misused_recursion`, `poor_decomposition`, `duplicated_logic` |

### Difficulty Levels (`data/labels/difficulty_levels.json`)

| Level | Description |
|---|---|
| `beginner` | No assumed prior knowledge. Explanations define all terms. Hints are very granular. |
| `intermediate` | Assumes familiarity with basic syntax and control flow. Explanations focus on reasoning. |
| `advanced` | Assumes strong language proficiency. Explanations focus on design and efficiency trade-offs. |

Difficulty is assigned per record based on the complexity of the concept involved and the estimated experience level of the submitter.

---

## Data Pipeline

The full pipeline is automated. Run it with:

```bash
bash scripts/preprocess.sh
```

Internally this executes the following stages in order:

**1. Collection** (`src/data_pipeline/collector.py`)
Downloads or reads raw source files. Filters to supported languages. Deduplicates by code hash.

**2. Cleaning** (`src/data_pipeline/cleaner.py`)
Strips HTML entities and markdown from Stack Overflow content. Extracts code blocks. Normalises indentation to spaces. Removes samples shorter than 3 lines or longer than 512 tokens.

**3. Tokenisation** (`src/data_pipeline/tokenizer.py`)
Applies the CodeBERT tokeniser with per-language settings from `configs/language_config.yaml`. Produces token IDs and attention masks. Truncates to the configured max sequence length.

**4. Dataset Building** (`src/data_pipeline/dataset_builder.py`)
Stratified split by language, difficulty, and error category to ensure balanced representation in each partition:

| Split | Size | File |
|---|---|---|
| Train | 80% | `data/processed/train.json` |
| Validation | 10% | `data/processed/val.json` |
| Test | 10% | `data/processed/test.json` |

The test split is held out and never used during model selection or hyperparameter tuning.

---

## Adding New Data

To add a new dataset:

1. Place raw files under `data/raw/` in the appropriate sub-directory.
2. Write a collection adapter in `collector.py` that reads the new format and emits records in the standard schema.
3. Tag records with `"source": "<your_dataset_name>"` for traceability.
4. Re-run `bash scripts/preprocess.sh` to regenerate the processed splits.

Ensure your adapter produces `null` for any schema fields the new source cannot supply — the pipeline handles missing fields gracefully for each training objective.
