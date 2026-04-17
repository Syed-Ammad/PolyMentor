# PolyMentor — Polycode Integration Plan

> How PolyMentor plugs into the Polycode platform as its AI learning layer.

---

## Context

[Polycode](https://github.com/your-org/polycode) is an interactive coding education platform. PolyMentor is designed to be its AI intelligence layer — the system that watches a learner's code as they write it and provides real-time feedback, explanations, and hints without a human teacher in the loop.

This document describes the integration architecture, the API contract between the two systems, and the phased rollout plan.

---

## Integration Architecture

PolyMentor exposes a single HTTP endpoint. Polycode calls it. Everything else is internal to PolyMentor.

```
┌─────────────────────────────────────┐
│           Polycode Platform          │
│                                     │
│  ┌────────────┐   ┌──────────────┐  │
│  │ Code Editor│   │  Learner     │  │
│  │ (frontend) │   │  Dashboard   │  │
│  └─────┬──────┘   └──────▲───────┘  │
│        │   submit code   │ feedback │
│        ▼                 │          │
│  ┌─────────────────────────────┐    │
│  │   Polycode Backend Service  │    │
│  └─────────────┬───────────────┘    │
└────────────────│────────────────────┘
                 │  POST /analyze
                 ▼
┌────────────────────────────────────┐
│         PolyMentor Service         │
│   (FastAPI + inference pipeline)   │
└────────────────────────────────────┘
```

The integration is intentionally shallow: Polycode treats PolyMentor as an external service with a stable JSON API. PolyMentor has no knowledge of Polycode's internal data model. This keeps the two systems independently deployable and upgradeable.

---

## API Contract

### Endpoint

```
POST /analyze
Content-Type: application/json
```

### Request

```json
{
  "code": "for i in range(10):\n    if i = 5:\n        break",
  "language": "python",
  "level": "beginner",
  "session_id": "user_abc_session_42",
  "context": {
    "problem_id": "lc_001",
    "attempt_number": 3
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `code` | string | Yes | The raw source code to analyse |
| `language` | string | Yes | One of: `python`, `javascript`, `cpp`, `java` |
| `level` | string | No | `beginner`, `intermediate`, or `advanced`. Defaults to `beginner`. |
| `session_id` | string | No | Opaque identifier for session continuity. Used by tutor mode for state tracking. |
| `context` | object | No | Optional metadata (problem ID, attempt number) passed through for logging. |

### Response

```json
{
  "status": "error_found",
  "error_type": "syntax_error/assignment_in_condition",
  "error_location": 2,
  "explanation": "You used the = operator inside an if condition. In Python, = assigns a value to a variable — it doesn't check whether two values are equal. Inside an if condition, you almost always want to compare, not assign.",
  "hint": "Think about what you are trying to do inside the if condition — are you trying to set a value or check a value?",
  "concept_taught": "Comparison Operators: == vs =",
  "quality_score": 72,
  "suggestions": [
    "Use == to test equality inside conditions.",
    "Consider naming loop variables more descriptively than 'i' when the intent is non-trivial."
  ],
  "session_state": "hint_1_of_3"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `error_found`, `clean`, or `error` (internal failure) |
| `error_type` | string | Taxonomy path, e.g. `syntax_error/assignment_in_condition` |
| `error_location` | int | Line number (1-indexed) |
| `explanation` | string | Human-readable explanation of the error |
| `hint` | string | First hint in the sequence |
| `concept_taught` | string | The concept this error maps to |
| `quality_score` | int | 0–100 composite score |
| `suggestions` | array | Actionable improvement recommendations |
| `session_state` | string | Opaque token for Polycode to pass back in the next request to retrieve subsequent hints |

---

## Hint Pagination

Subsequent hints are requested by passing the `session_state` token back:

```json
{
  "code": "...",
  "language": "python",
  "level": "beginner",
  "session_id": "user_abc_session_42",
  "session_state": "hint_1_of_3",
  "action": "next_hint"
}
```

PolyMentor returns the next hint in the sequence. When the final hint has been returned, `session_state` is set to `"hints_exhausted"`.

---

## Authentication

All requests from Polycode to PolyMentor are authenticated with a shared service-to-service API key passed in the `X-API-Key` header. Key rotation is handled by the infrastructure team and does not require code changes.

---

## Deployment

PolyMentor runs as a containerised service in the same AWS region as the Polycode backend to minimise latency.

```
Polycode Backend  ──(private VPC)──►  PolyMentor Service
                                       (ECS Fargate task)
                                       GPU instance for model inference
                                       Auto-scales on request volume
```

Model checkpoints are loaded from S3 at container startup. The container does not bundle model weights — this keeps image size small and allows hot model upgrades without rebuilding the image.

---

## PolyGuard + PolyMentor Unified Layer

PolyGuard (security analysis) and PolyMentor (learning guidance) share the same CodeBERT backbone and AST infrastructure. In the unified intelligence layer, a single code submission is routed through both systems:

```
Code Submission
      │
      ├──► PolyGuard  ──► security issues, vulnerability flags, secure fix suggestions
      │
      └──► PolyMentor ──► error explanations, concept teaching, progressive hints
```

Polycode's backend merges the two responses and presents them in context: security issues appear in a "Security" panel, learning feedback appears in a "Mentor" panel. The learner sees a unified view without knowing the underlying systems are separate.

The shared infrastructure means both models can be updated together in a single deployment when the backbone is retrained.

---

## Rollout Plan

### Phase 1 — Basic Integration (Current)

- `/analyze` endpoint live with error detection and explanation.
- Polycode calls it on manual "Check my code" button press.
- No session state or hint pagination yet.
- Latency target: < 2 seconds p95.

### Phase 2 — Tutor Mode

- Session state and hint pagination enabled.
- Polycode surfaces a "Get a hint" button that calls `next_hint`.
- Learner dashboard shows concepts taught per session.

### Phase 3 — Real-Time Feedback

- PolyMentor called on a debounced keystroke (500ms idle) rather than manual button press.
- Requires latency target of < 500ms p95 — may require model quantisation or a lighter detection model for the first pass.
- Full explanation model called only when the learner pauses for > 2 seconds after an error is flagged.

### Phase 4 — Adaptive Difficulty

- Learner skill profile stored server-side and passed in each request.
- PolyMentor adjusts explanation depth and hint granularity based on the learner's history with each concept.
- Polycode dashboard shows per-concept mastery progress.

### Phase 5 — Feedback Learning Loop

- User signals (hint accepted, explanation rated, same error repeated) collected by Polycode and sent to PolyMentor's training pipeline.
- Models retrained on a weekly cadence using accumulated signals.
- A/B testing framework built into the Polycode integration to evaluate model improvements before full rollout.

---

## Monitoring

| Metric | Owner | Alert threshold |
|---|---|---|
| `/analyze` p95 latency | PolyMentor | > 3 seconds |
| Error rate (5xx) | PolyMentor | > 1% over 5 minutes |
| Explanation model confidence | PolyMentor | Mean < 0.6 over 1 hour |
| Hint acceptance rate | Polycode | < 20% (signals hints unhelpful) |
| Learner error recurrence rate | Polycode | > 60% same error in same session |

Alerts route to the QuantumLogics on-call channel. Hint acceptance rate and error recurrence rate are the primary indicators of learning effectiveness — they matter more than raw latency for evaluating whether the system is working.
