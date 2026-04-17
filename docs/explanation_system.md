# PolyMentor — Explanation & Hint System

> How PolyMentor generates human-like explanations and progressive hints from detected errors.

---

## Philosophy

Most code analysis tools tell you *what* broke and *where*. PolyMentor's explanation system is built around a different question: *what does the learner need to understand to not make this mistake again?*

This means the system doesn't just describe the error — it maps it to the underlying concept, adjusts explanation depth to the learner's level, and generates hints that build understanding rather than hand over the answer.

---

## Architecture Overview

The explanation pipeline has four cooperating components:

```
Detected Error
      │
      ▼
error_classifier.py  ──►  concept mapping + difficulty scoring
      │
      ▼
explanation_generator.py  ──►  human-language explanation
      │
      ▼
hint_system.py  ──►  ordered step-by-step hints
      │
      ▼
feedback_scorer.py  ──►  quality score + improvement suggestions
```

All four live in `src/reasoning_engine/`. The inference layer (`src/inference/explain.py`) calls them in sequence and assembles the final response.

---

## Component Details

### 1. Error Classifier (`error_classifier.py`)

This module bridges the gap between a raw model prediction and a teachable concept.

**Input:** A multi-label error type vector from the detection model (e.g. `["syntax_error/assignment_in_condition"]`).

**Processing:**
- Looks up each error type in the concept graph defined in `data/labels/error_types.json`.
- Traverses the graph upward: specific error → concept → parent concept → domain.
- Produces a ranked list of concepts the learner needs to understand, ordered from most immediate to most foundational.

**Example traversal:**
```
assignment_in_condition
  └─► comparison_operators (== vs =)
        └─► control_flow_basics
              └─► fundamental_syntax
```

**Output:** Ranked concept list + primary concept label used in the response.

---

### 2. Explanation Generator (`explanation_generator.py` + `src/models/explanation_model.py`)

**Model:** A fine-tuned sequence-to-sequence model. The current implementation supports CodeT5 and LLaMA-based backbones, selected via `configs/model_config.yaml`.

**Training data:** Error + explanation pairs sourced from Stack Overflow accepted answers and the custom teacher-written dataset (see [`dataset_guide.md`](dataset_guide.md)). Teacher-written examples are upweighted because they are explicitly pedagogical.

**Input to model:**
```
[LANG] python [LEVEL] beginner [ERROR] syntax_error/assignment_in_condition
[CODE] if i = 5:
[CONCEPT] comparison_operators
```

**Output:** A plain-English explanation that:
- States what the error is in accessible terms.
- Explains *why* the erroneous code doesn't work the way the learner likely expected.
- Names the concept and briefly defines it.
- Does not give away the corrected code (that is the hint system's job).

**Explanation depth by level:**

| Level | Behaviour |
|---|---|
| `beginner` | Defines all terms. Avoids jargon. Uses analogies. Assumes no prior knowledge. |
| `intermediate` | Names concepts directly. Focuses on the reasoning error. Brief definition if concept is niche. |
| `advanced` | Concise. Focuses on design implication or performance trade-off where relevant. |

---

### 3. Hint System (`hint_system.py` + `src/models/hint_generator.py`)

The hint system generates a *sequence* of hints rather than a single one. Each hint in the sequence reveals slightly more information, guiding the learner toward the solution through their own reasoning.

**Design principle:** A learner who asks for a hint should be able to make progress without being given the answer. The final hint in the sequence brings them to the doorstep of the solution but does not open the door.

**Hint generation process:**

1. The hint generator receives the detected error type, the relevant concept, the learner level, and the original code snippet.
2. It generates an ordered list of 3–5 hints using the fine-tuned hint model.
3. Hints are ordered from most abstract to most specific.
4. The inference layer returns only the first hint by default. Subsequent hints are released on demand (in tutor mode) or all at once (in API mode if `full_hints=True`).

**Example hint sequence for `assignment_in_condition`:**
```
Hint 1: Think about what you are trying to do inside the if condition — 
        are you trying to set a value or check a value?

Hint 2: Python uses two different operators for these two things. 
        What are they, and which one belongs in an if condition?

Hint 3: The = operator always assigns. To test equality, you need the == operator.
        Look at your if condition again.
```

**Hint model:** Fine-tuned on problem + buggy-solution + correct-solution triples from LeetCode and HackerRank, with hints constructed by diffing the two solutions and ordering the diffs by conceptual proximity to the detected error.

---

### 4. Feedback Scorer (`feedback_scorer.py`)

Produces a 0–100 quality score and a list of concrete improvement suggestions, independent of whether an error was detected.

**Score components:**

| Dimension | Weight | What is measured |
|---|---|---|
| Correctness | 40% | Error-free code scores full marks here; each error type reduces the score |
| Readability | 30% | Variable naming, line length, comment presence, consistent indentation |
| Complexity | 20% | Cyclomatic complexity, nesting depth, function length |
| Clean code | 10% | Magic numbers, unused variables, duplicated logic |

**Suggestions** are generated as a short list of actionable items derived from the lowest-scoring dimensions. They are phrased as imperatives: "Extract this nested block into a named function", "Replace the magic number 42 with a named constant".

---

## Tutor Mode (`src/inference/tutor_mode.py`)

Tutor mode wraps the explanation pipeline in a stateful, multi-turn session. It tracks:

- Which hints have already been shown.
- Whether the learner acknowledged the explanation.
- The learner's error history across the session (used for adaptive difficulty — described below).

Tutor mode is invoked via `bash scripts/run_tutor.sh` for interactive CLI use, or via the `TutorSession` class in code.

---

## Adaptive Difficulty (Planned)

The current system uses the `level` parameter supplied at call time. The planned adaptive difficulty system will estimate the learner's level dynamically from their error history:

- Persistent errors of the same type lower the estimated level for that concept → simpler explanations and more granular hints.
- Correct code after a hint raises the estimated level → less scaffolding next time.
- The user's estimated level per concept is stored in a session profile and updated after each interaction.

This is tracked under the roadmap item "Adaptive difficulty scoring per user session".

---

## Planned: Code-to-Concept Graph

A structured graph connecting error patterns to programming concepts is planned as a first-class data asset. It will replace the current flat lookup in `error_classifier.py` with a traversable graph, enabling:

- Multi-hop concept explanations ("this error relates to loop indexing, which relates to off-by-one errors, which relates to iteration fundamentals").
- Concept-level progress tracking across sessions.
- Smarter hint ordering based on which ancestor concepts the learner has already mastered.

---

## Planned: Feedback Learning Loop

Model quality will improve over time using signals from user interactions:

| Signal | How it is used |
|---|---|
| Hint accepted (learner solved it after this hint) | Positive training signal for that hint |
| Explanation rated helpful | Positive training signal for that explanation |
| Explanation skipped or rated unhelpful | Negative signal; triggers reranking |
| Learner made the same error again later | Signals explanation was insufficient |

This closes the loop between model outputs and real learning outcomes, something static datasets cannot provide.
