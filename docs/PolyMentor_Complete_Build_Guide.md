# 🧠 PolyMentor — Complete Build Guide from Scratch

> **Current Status:** Repo created with only `README.md`
> **Goal:** Build a fully working AI-powered coding mentor — error detection, concept teaching, and progressive hints across C++, Python, JavaScript, and Java.

---

## 📋 Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Phase 1 — Repo & Environment Setup](#2-phase-1--repo--environment-setup)
3. [Phase 2 — Project Structure](#3-phase-2--project-structure)
4. [Phase 3 — Configuration Files](#4-phase-3--configuration-files)
5. [Phase 4 — Dataset Collection & Labeling](#5-phase-4--dataset-collection--labeling)
6. [Phase 5 — Data Pipeline](#6-phase-5--data-pipeline)
7. [Phase 6 — Feature Extraction](#7-phase-6--feature-extraction)
8. [Phase 7 — Error Detection Model](#8-phase-7--error-detection-model)
9. [Phase 8 — Reasoning Engine](#9-phase-8--reasoning-engine)
10. [Phase 9 — Explanation & Hint Models](#10-phase-9--explanation--hint-models)
11. [Phase 10 — Training Pipeline](#11-phase-10--training-pipeline)
12. [Phase 11 — Evaluation](#12-phase-11--evaluation)
13. [Phase 12 — Inference Pipeline (Public API)](#13-phase-12--inference-pipeline-public-api)
14. [Phase 13 — Interactive Tutor Mode](#14-phase-13--interactive-tutor-mode)
15. [Phase 14 — FastAPI Backend](#15-phase-14--fastapi-backend)
16. [Phase 15 — Testing](#16-phase-15--testing)
17. [Phase 16 — Docker & Deployment](#17-phase-16--docker--deployment)
18. [Full Build Checklist](#18-full-build-checklist)

---

## 1. Project Overview & Architecture

PolyMentor works in a 4-stage pipeline:

```
Source Code Input
      │
      ▼
AST Parsing + CodeBERT Tokenization
      │
      ▼
Error Detection Model (Multi-label CodeBERT classifier)
      │
      ▼
Reasoning Engine (Error → Concept mapping + Difficulty scoring)
      │
      ├──► Explanation Generator (Seq2Seq — WHY did it fail?)
      ├──► Hint System (Step-by-step — HOW to fix it?)
      └──► Quality Scorer (Code quality score 0–100)
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.x |
| Code Understanding | CodeBERT (`microsoft/codebert-base`) |
| Explanation Generation | CodeT5 fine-tuned |
| AST Parsing | Tree-sitter |
| Backend API | FastAPI + Uvicorn |
| Deployment | Docker + AWS |

---

## 2. Phase 1 — Repo & Environment Setup

### 2.1 — System Requirements

Make sure your machine has:

- Python **3.10+**
- pip **23+**
- Git
- Node.js 18+ (optional, for tooling)
- NVIDIA GPU + CUDA (recommended for training; CPU works for inference)
- 16 GB RAM recommended

### 2.2 — Clone Your Repo

```bash
git clone https://github.com/your-username/PolyMentor.git
cd PolyMentor
```

### 2.3 — Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 2.4 — Create `requirements.txt`

Create this file in the root of your repo:

```txt
# Core ML
torch>=2.0.0
transformers>=4.38.0
datasets>=2.18.0
accelerate>=0.27.0
evaluate>=0.4.0
scikit-learn>=1.4.0
numpy>=1.26.0
scipy>=1.12.0

# Code Parsing
tree-sitter==0.21.3
tree-sitter-python==0.21.0
tree-sitter-javascript==0.21.0
tree-sitter-java==0.21.0
tree-sitter-cpp==0.21.0

# API & Serving
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0

# Utilities
pyyaml>=6.0.1
tqdm>=4.66.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
ipykernel>=6.29.0

# Code Quality
flake8>=7.0.0
black>=24.0.0
pytest>=8.0.0
pytest-cov>=4.1.0
```

### 2.5 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.6 — Create `setup.py`

This lets Python resolve `src/` imports everywhere in the project:

```python
from setuptools import setup, find_packages

setup(
    name="polymentor",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

Install in editable mode:

```bash
pip install -e .
```

### 2.7 — Create `.gitignore`

```gitignore
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/

# Data & Models (too large for git)
data/raw/
data/processed/
models_saved/
experiments/logs/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Env
.env
*.local.yaml
```

---

## 3. Phase 2 — Project Structure

Run these commands to scaffold the entire folder structure:

```bash
# Config
mkdir -p configs

# Data
mkdir -p data/raw/code_datasets
mkdir -p data/raw/error_samples
mkdir -p data/raw/programming_questions
mkdir -p data/processed
mkdir -p data/labels

# Notebooks
mkdir -p notebooks

# Source code
mkdir -p src/data_pipeline
mkdir -p src/features
mkdir -p src/models
mkdir -p src/reasoning_engine
mkdir -p src/training
mkdir -p src/evaluation
mkdir -p src/inference
mkdir -p src/api
mkdir -p src/utils

# Experiments
mkdir -p experiments/exp_01_tfidf_baseline
mkdir -p experiments/exp_02_codebert_model
mkdir -p experiments/exp_03_explanation_finetune
mkdir -p experiments/logs

# Saved models
mkdir -p models_saved

# Tests
mkdir -p tests

# Scripts
mkdir -p scripts

# Docs
mkdir -p docs

# Create __init__.py in all src subdirs
touch src/__init__.py
touch src/data_pipeline/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/reasoning_engine/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/inference/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

Your structure should now look like this:

```
PolyMentor/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels/
├── notebooks/
├── src/
│   ├── data_pipeline/
│   ├── features/
│   ├── models/
│   ├── reasoning_engine/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── api/
│   └── utils/
├── experiments/
├── models_saved/
├── tests/
├── scripts/
├── docs/
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 4. Phase 3 — Configuration Files

### 4.1 — `configs/model_config.yaml`

```yaml
model:
  backbone: "microsoft/codebert-base"
  num_labels: 9
  max_seq_length: 512
  dropout: 0.1
  hidden_size: 768

explanation_model:
  backbone: "Salesforce/codet5-base"
  max_input_length: 512
  max_output_length: 256
```

### 4.2 — `configs/training_config.yaml`

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-5
  weight_decay: 0.01
  epochs: 10
  warmup_ratio: 0.1
  fp16: true
  seed: 42
  output_dir: "models_saved/"
  logging_steps: 50
  eval_steps: 200
  save_steps: 500
  load_best_model_at_end: true
  metric_for_best_model: "f1_micro"
```

### 4.3 — `configs/language_config.yaml`

```yaml
supported_languages:
  python:
    extension: ".py"
    tree_sitter_grammar: "tree-sitter-python"
    tokenizer: "microsoft/codebert-base"
  javascript:
    extension: ".js"
    tree_sitter_grammar: "tree-sitter-javascript"
    tokenizer: "microsoft/codebert-base"
  java:
    extension: ".java"
    tree_sitter_grammar: "tree-sitter-java"
    tokenizer: "microsoft/codebert-base"
  cpp:
    extension: ".cpp"
    tree_sitter_grammar: "tree-sitter-cpp"
    tokenizer: "microsoft/codebert-base"
```

### 4.4 — `src/utils/config_loader.py`

```python
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = "configs") -> dict:
    """Load all configs from the configs/ directory."""
    config_dir = Path(config_dir)
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        name = yaml_file.stem.replace("_config", "")
        configs[name] = load_config(str(yaml_file))
    return configs
```

### 4.5 — `src/utils/logger.py`

```python
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

---

## 5. Phase 4 — Dataset Collection & Labeling

### 5.1 — Define Error Taxonomy

Create `data/labels/error_types.json`:

```json
{
  "syntax_error": 0,
  "logical_error": 1,
  "type_error": 2,
  "off_by_one": 3,
  "infinite_loop": 4,
  "null_reference": 5,
  "division_by_zero": 6,
  "bad_practice": 7,
  "structural_issue": 8
}
```

Create `data/labels/difficulty_levels.json`:

```json
{
  "beginner": 0,
  "intermediate": 1,
  "advanced": 2
}
```

### 5.2 — Define the Unified Data Schema

Every sample in your dataset must follow this schema:

```json
{
  "id": "py_001",
  "code": "for i in range(10):\n    if i = 5:\n        break",
  "language": "python",
  "error_types": ["syntax_error"],
  "difficulty": "beginner",
  "explanation": "You used = which assigns a value. To compare two values, use == instead.",
  "hint_steps": [
    "Step 1: Look at your if condition. What operator are you using?",
    "Step 2: In Python, = assigns a value to a variable. Which operator checks if two values are equal?",
    "Step 3: Replace = with == inside your if statement."
  ],
  "concept_taught": "Comparison Operators: == vs =",
  "quality_score": 72
}
```

### 5.3 — Download Public Datasets

**CodeNet (IBM):**
```bash
# Download a subset (full dataset is ~8GB)
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
tar -xzf Project_CodeNet.tar.gz -C data/raw/code_datasets/
```

**Stack Overflow Dump:**

Download the SO dump from [archive.org/details/stackexchange](https://archive.org/details/stackexchange). Filter for posts tagged `python`, `javascript`, `java`, `c++` with accepted answers.

**ManyBugs:**
```bash
git clone https://github.com/squaresLab/ManyBugs data/raw/error_samples/manybugs
```

### 5.4 — Create Placeholder Processed Files

While you collect and clean data, create placeholder files:

```bash
echo "[]" > data/processed/train.json
echo "[]" > data/processed/val.json
echo "[]" > data/processed/test.json
```

---

## 6. Phase 5 — Data Pipeline

### 6.1 — `src/data_pipeline/collector.py`

```python
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCollector:
    """Loads and normalizes raw code samples into the unified schema."""

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load_json_file(self, filepath: str) -> list:
        with open(filepath, "r") as f:
            return json.load(f)

    def load_all_samples(self) -> list:
        """Load all .json files from data/raw/ recursively."""
        samples = []
        for json_file in self.raw_dir.rglob("*.json"):
            try:
                data = self.load_json_file(json_file)
                if isinstance(data, list):
                    samples.extend(data)
                    logger.info(f"Loaded {len(data)} samples from {json_file}")
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        return samples
```

### 6.2 — `src/data_pipeline/cleaner.py`

```python
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ["id", "code", "language", "error_types", "difficulty",
                   "explanation", "hint_steps", "concept_taught"]

SUPPORTED_LANGUAGES = {"python", "javascript", "java", "cpp"}


class DataCleaner:
    """Validates and cleans raw data samples."""

    def clean(self, samples: list) -> list:
        cleaned = []
        seen_ids = set()

        for sample in samples:
            # Check required fields
            if not all(k in sample for k in REQUIRED_FIELDS):
                continue

            # Deduplicate by ID
            if sample["id"] in seen_ids:
                continue
            seen_ids.add(sample["id"])

            # Validate language
            if sample["language"] not in SUPPORTED_LANGUAGES:
                continue

            # Clean code: strip trailing whitespace per line
            sample["code"] = "\n".join(
                line.rstrip() for line in sample["code"].split("\n")
            )

            # Truncate very long snippets
            if len(sample["code"]) > 4000:
                continue

            cleaned.append(sample)

        logger.info(f"Cleaned: {len(cleaned)}/{len(samples)} samples kept")
        return cleaned
```

### 6.3 — `src/data_pipeline/tokenizer.py`

```python
from transformers import AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeTokenizer:
    """Wraps HuggingFace tokenizer for code snippets."""

    def __init__(self, model_name: str = "microsoft/codebert-base", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info(f"Loaded tokenizer: {model_name}")

    def tokenize(self, code: str, language: str = "") -> dict:
        """Tokenize a code snippet. Language prefix helps the model."""
        prefix = f"<{language}> " if language else ""
        return self.tokenizer(
            prefix + code,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    def batch_tokenize(self, codes: list, languages: list = None) -> dict:
        """Tokenize a batch of code snippets."""
        if languages is None:
            languages = [""] * len(codes)
        texts = [f"<{lang}> {code}" for lang, code in zip(languages, codes)]
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
```

### 6.4 — `src/data_pipeline/dataset_builder.py`

```python
import json
import random
from pathlib import Path
from src.data_pipeline.collector import DataCollector
from src.data_pipeline.cleaner import DataCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """Builds train/val/test splits and writes them to data/processed/."""

    def __init__(self, raw_dir: str = "data/raw", output_dir: str = "data/processed",
                 train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
        self.raw_dir = raw_dir
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def build(self):
        """Full pipeline: collect → clean → split → save."""
        collector = DataCollector(self.raw_dir)
        cleaner = DataCleaner()

        samples = collector.load_all_samples()
        samples = cleaner.clean(samples)

        random.seed(self.seed)
        random.shuffle(samples)

        n = len(samples)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train:n_train + n_val],
            "test": samples[n_train + n_val:]
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, "w") as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"Saved {len(split_data)} samples to {output_path}")

        return splits


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
```

### 6.5 — `scripts/preprocess.sh`

```bash
#!/bin/bash
set -e
echo "🔄 Running data preprocessing..."
python -m src.data_pipeline.dataset_builder
echo "✅ Preprocessing complete. Check data/processed/"
```

```bash
chmod +x scripts/preprocess.sh
```

---

## 7. Phase 6 — Feature Extraction

### 7.1 — Build Tree-sitter Language Grammars

First, clone the grammar repos:

```bash
mkdir -p vendor
git clone https://github.com/tree-sitter/tree-sitter-python vendor/tree-sitter-python
git clone https://github.com/tree-sitter/tree-sitter-javascript vendor/tree-sitter-javascript
git clone https://github.com/tree-sitter/tree-sitter-java vendor/tree-sitter-java
git clone https://github.com/tree-sitter/tree-sitter-cpp vendor/tree-sitter-cpp
```

### 7.2 — `src/features/ast_parser.py`

```python
from tree_sitter import Language, Parser
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

GRAMMAR_DIR = Path("vendor")
LIB_PATH = Path("build/languages.so")


def build_language_library():
    """Compile Tree-sitter grammars into a shared library."""
    LIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Language.build_library(
        str(LIB_PATH),
        [
            str(GRAMMAR_DIR / "tree-sitter-python"),
            str(GRAMMAR_DIR / "tree-sitter-javascript"),
            str(GRAMMAR_DIR / "tree-sitter-java"),
            str(GRAMMAR_DIR / "tree-sitter-cpp"),
        ]
    )
    logger.info(f"Built language library at {LIB_PATH}")


LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "java": "java",
    "cpp": "cpp",
}


class ASTParser:
    """Parses source code into an Abstract Syntax Tree using Tree-sitter."""

    def __init__(self):
        if not LIB_PATH.exists():
            build_language_library()
        self.parsers = {}
        for lang_name, ts_name in LANGUAGE_MAP.items():
            lang = Language(str(LIB_PATH), ts_name)
            parser = Parser()
            parser.set_language(lang)
            self.parsers[lang_name] = parser

    def parse(self, code: str, language: str) -> dict:
        """Parse code and return a dict representation of the AST."""
        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}")

        parser = self.parsers[language]
        tree = parser.parse(bytes(code, "utf8"))
        return self._node_to_dict(tree.root_node)

    def _node_to_dict(self, node, depth: int = 0) -> dict:
        return {
            "type": node.type,
            "start_point": node.start_point,
            "end_point": node.end_point,
            "depth": depth,
            "children": [self._node_to_dict(c, depth + 1) for c in node.children]
        }

    def get_node_types(self, code: str, language: str) -> list:
        """Return a flat list of all AST node types for feature extraction."""
        ast = self.parse(code, language)
        return self._flatten_types(ast)

    def _flatten_types(self, node: dict) -> list:
        types = [node["type"]]
        for child in node.get("children", []):
            types.extend(self._flatten_types(child))
        return types
```

### 7.3 — `src/features/code_embeddings.py`

```python
import torch
from transformers import AutoModel, AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeEmbedder:
    """Generates dense vector embeddings for code using CodeBERT."""

    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"CodeEmbedder loaded on {self.device}")

    def embed(self, code: str, language: str = "") -> torch.Tensor:
        """Get the CLS token embedding for a single code snippet."""
        prefix = f"<{language}> " if language else ""
        inputs = self.tokenizer(
            prefix + code,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        # CLS token = index 0 of last hidden state
        return output.last_hidden_state[:, 0, :].squeeze(0)

    def embed_batch(self, codes: list, languages: list = None) -> torch.Tensor:
        """Get embeddings for a batch of code snippets."""
        if languages is None:
            languages = [""] * len(codes)
        texts = [f"<{lang}> {code}" for lang, code in zip(languages, codes)]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        return output.last_hidden_state[:, 0, :]
```

---

## 8. Phase 7 — Error Detection Model

### 8.1 — `src/models/error_detector.py`

```python
import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorDetector(nn.Module):
    """
    Multi-label error classifier built on CodeBERT.
    A single code snippet can have multiple simultaneous error types.
    """

    def __init__(self, num_labels: int = 9, model_name: str = "microsoft/codebert-base",
                 dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

        logger.info(f"ErrorDetector initialized: {num_labels} labels, backbone={model_name}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits (NOT sigmoid). Apply sigmoid for probabilities.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # CLS token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions (0 or 1) per label."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).int()
```

### 8.2 — `src/models/model_factory.py`

```python
import torch
from pathlib import Path
from src.models.error_detector import ErrorDetector
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Handles loading and saving of all PolyMentor models."""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config = load_config(config_path)

    def build_error_detector(self) -> ErrorDetector:
        cfg = self.config["model"]
        return ErrorDetector(
            num_labels=cfg["num_labels"],
            model_name=cfg["backbone"],
            dropout=cfg["dropout"]
        )

    def load_error_detector(self, checkpoint_path: str) -> ErrorDetector:
        model = self.build_error_detector()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded ErrorDetector from {checkpoint_path}")
        return model

    def save_model(self, model: torch.nn.Module, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
```

---

## 9. Phase 8 — Reasoning Engine

### 9.1 — `src/reasoning_engine/error_classifier.py`

```python
import json
from pathlib import Path

# Maps error label → programming concept to teach
ERROR_TO_CONCEPT = {
    "syntax_error": "Python/Language Syntax Rules",
    "logical_error": "Program Logic & Control Flow",
    "type_error": "Data Types & Type Casting",
    "off_by_one": "Loop Indexing & Array Boundaries",
    "infinite_loop": "Loop Termination Conditions",
    "null_reference": "Null Safety & Object Initialization",
    "division_by_zero": "Input Validation & Edge Cases",
    "bad_practice": "Clean Code & Readability Principles",
    "structural_issue": "Function Design & Code Decomposition",
}


class ErrorClassifier:
    """Maps raw error labels to human-readable types and concepts."""

    def __init__(self, error_types_path: str = "data/labels/error_types.json"):
        with open(error_types_path, "r") as f:
            self.error_types = json.load(f)
        # Reverse map: index → name
        self.idx_to_label = {v: k for k, v in self.error_types.items()}

    def decode(self, binary_vector: list) -> list:
        """Convert a binary prediction vector to a list of error label strings."""
        return [
            self.idx_to_label[i]
            for i, val in enumerate(binary_vector)
            if val == 1
        ]

    def get_concepts(self, error_labels: list) -> list:
        """Map error labels to the concepts they teach."""
        return [ERROR_TO_CONCEPT.get(label, "General Programming") for label in error_labels]

    def get_primary_error(self, error_labels: list) -> str:
        """Return the most important error to address first."""
        priority = [
            "syntax_error", "type_error", "null_reference",
            "division_by_zero", "off_by_one", "infinite_loop",
            "logical_error", "structural_issue", "bad_practice"
        ]
        for p in priority:
            if p in error_labels:
                return p
        return error_labels[0] if error_labels else "unknown"
```

### 9.2 — `src/reasoning_engine/feedback_scorer.py`

```python
import ast
import re


class FeedbackScorer:
    """
    Scores code quality on a 0–100 scale.
    Checks readability, complexity, and clean code principles.
    """

    def score(self, code: str, language: str = "python") -> int:
        score = 100

        lines = code.strip().split("\n")

        # Penalize very long lines
        long_lines = sum(1 for l in lines if len(l) > 100)
        score -= long_lines * 3

        # Penalize deeply nested code (4+ levels of indentation)
        deep_nesting = sum(1 for l in lines if l.startswith("    " * 4))
        score -= deep_nesting * 5

        # Penalize magic numbers
        magic_numbers = len(re.findall(r'\b(?<!\.)\d{2,}\b', code))
        score -= magic_numbers * 2

        # Penalize very short variable names (single char, excluding i/j/k/n/x/y)
        bad_vars = len(re.findall(r'\b(?![ijknxy])[a-wz]\b\s*=', code))
        score -= bad_vars * 3

        # Penalize no comments/docstrings in longer functions
        if len(lines) > 15 and '"""' not in code and "#" not in code:
            score -= 10

        return max(0, min(100, score))
```

### 9.3 — `src/reasoning_engine/hint_system.py`

```python
# Maps error type → progressive hint steps
HINT_TEMPLATES = {
    "syntax_error": [
        "Step 1: Read the error message carefully — it usually tells you the exact line.",
        "Step 2: Look at the line mentioned. Check for missing colons, brackets, or quotes.",
        "Step 3: Compare your syntax against a working example of the same construct.",
    ],
    "off_by_one": [
        "Step 1: Trace through your loop manually with a small example (e.g., 3 items).",
        "Step 2: Check: does your index start at 0 or 1? Where does it end?",
        "Step 3: Try changing your range by ±1 and re-trace. Does the output match now?",
    ],
    "logical_error": [
        "Step 1: Add print statements before and after the suspicious section to see values.",
        "Step 2: Write down the expected value vs. what you actually got.",
        "Step 3: Trace each step of your logic on paper for a simple test case.",
    ],
    "infinite_loop": [
        "Step 1: Find your loop. What condition makes it stop?",
        "Step 2: Is that condition ever actually reached during execution?",
        "Step 3: Add a print inside the loop to see if the stopping variable changes each iteration.",
    ],
    "type_error": [
        "Step 1: Print the type of the variable causing the error using type().",
        "Step 2: Check what type the function or operation expects as input.",
        "Step 3: Convert the variable to the correct type before using it.",
    ],
    "division_by_zero": [
        "Step 1: Find where division happens in your code.",
        "Step 2: Can the denominator ever be 0? What inputs would cause that?",
        "Step 3: Add a check: if denominator != 0: before dividing.",
    ],
}

DEFAULT_HINTS = [
    "Step 1: Re-read the code slowly, line by line.",
    "Step 2: Try running just the broken section in isolation.",
    "Step 3: Search for the error message online — you're likely not the first to see it.",
]


class HintSystem:
    """Generates progressive, step-by-step hints for a detected error."""

    def get_hints(self, error_label: str, level: str = "beginner") -> list:
        hints = HINT_TEMPLATES.get(error_label, DEFAULT_HINTS)

        # For advanced users, return fewer leading hints
        if level == "advanced":
            return hints[-1:]
        elif level == "intermediate":
            return hints[1:]
        return hints  # beginner gets all steps

    def get_first_hint(self, error_label: str, level: str = "beginner") -> str:
        hints = self.get_hints(error_label, level)
        return hints[0] if hints else DEFAULT_HINTS[0]
```

### 9.4 — `src/reasoning_engine/explanation_generator.py`

```python
# Rule-based explanation generator (used before the fine-tuned model is ready)
EXPLANATIONS = {
    "syntax_error": (
        "Your code has a syntax error — the language cannot understand its structure. "
        "This usually means a missing colon, bracket, quote, or wrong indentation."
    ),
    "logical_error": (
        "Your code runs without crashing, but produces the wrong result. "
        "The logic of your algorithm doesn't match what you intended it to do."
    ),
    "off_by_one": (
        "You have an off-by-one error. Your loop starts or ends one step too early or too late, "
        "which is very common when working with arrays or ranges."
    ),
    "infinite_loop": (
        "Your loop never terminates because its stopping condition is never reached. "
        "The variable that should end the loop isn't changing correctly."
    ),
    "type_error": (
        "You passed the wrong data type to a function or operation. "
        "For example, trying to add a string and a number without converting one of them."
    ),
    "division_by_zero": (
        "Your code attempts to divide a number by zero, which is mathematically undefined "
        "and causes a runtime crash."
    ),
    "bad_practice": (
        "Your code works, but uses patterns that make it hard to read, maintain, or extend. "
        "Consider naming variables descriptively and avoiding magic numbers."
    ),
}

DEFAULT_EXPLANATION = (
    "An error was detected in your code. Review the flagged section carefully "
    "and compare it against the expected behavior."
)


class ExplanationGenerator:
    """Provides human-readable explanations for detected errors."""

    def explain(self, error_label: str) -> str:
        return EXPLANATIONS.get(error_label, DEFAULT_EXPLANATION)

    def explain_all(self, error_labels: list) -> list:
        return [self.explain(label) for label in error_labels]
```

---

## 10. Phase 9 — Explanation & Hint Models (Fine-tuned)

This phase adds a **neural explanation generator** on top of CodeT5 for richer, code-aware explanations.

### 10.1 — `src/models/explanation_model.py`

```python
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplanationModel:
    """
    Fine-tuned CodeT5 model that generates natural language explanations
    for code errors. Input: code snippet + error type. Output: explanation string.
    """

    def __init__(self, model_name: str = "Salesforce/codet5-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        logger.info(f"ExplanationModel loaded: {model_name} on {self.device}")

    def generate(self, code: str, error_label: str, max_length: int = 256) -> str:
        """Generate a plain-English explanation for a code error."""
        prompt = f"explain error: [{error_label}]\ncode:\n{code}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def load_fine_tuned(self, checkpoint_path: str):
        """Load fine-tuned weights from a local checkpoint."""
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info(f"Loaded fine-tuned explanation model from {checkpoint_path}")
```

### 10.2 — Fine-tuning Script (`src/training/finetune_explanation.py`)

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

# Load your annotated explanation pairs
with open("data/processed/train.json") as f:
    train_data = json.load(f)

model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def preprocess(sample):
    error_label = sample["error_types"][0] if sample["error_types"] else "unknown"
    prompt = f"explain error: [{error_label}]\ncode:\n{sample['code']}"
    target = sample["explanation"]

    model_inputs = tokenizer(prompt, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset = Dataset.from_list(train_data)
tokenized = dataset.map(preprocess)

args = Seq2SeqTrainingArguments(
    output_dir="models_saved/explanation_model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    predict_with_generate=True,
    save_strategy="epoch",
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
model.save_pretrained("models_saved/explanation_model/final")
tokenizer.save_pretrained("models_saved/explanation_model/final")
```

---

## 11. Phase 10 — Training Pipeline

### 11.1 — `src/training/loss_functions.py`

```python
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Helps when error types are imbalanced (e.g., syntax errors are much more common).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()
```

### 11.2 — `src/training/metrics.py`

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Compute multi-label classification metrics."""
    return {
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, predictions, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, predictions, average="micro", zero_division=0),
    }
```

### 11.3 — `src/training/train.py`

```python
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.models.model_factory import ModelFactory
from src.data_pipeline.tokenizer import CodeTokenizer
from src.training.loss_functions import FocalLoss
from src.training.metrics import compute_metrics
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeErrorDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: CodeTokenizer, error_types: dict):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.error_types = error_types
        self.num_labels = len(error_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = self.tokenizer.tokenize(sample["code"], sample["language"])
        label_vector = torch.zeros(self.num_labels)
        for error in sample.get("error_types", []):
            if error in self.error_types:
                label_vector[self.error_types[error]] = 1.0

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": label_vector,
        }


def train():
    config = load_config("configs/training_config.yaml")
    model_config = load_config("configs/model_config.yaml")

    with open("data/labels/error_types.json") as f:
        error_types = json.load(f)

    factory = ModelFactory()
    model = factory.build_error_detector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = CodeTokenizer(model_config["model"]["backbone"])

    train_dataset = CodeErrorDataset("data/processed/train.json", tokenizer, error_types)
    val_dataset = CodeErrorDataset("data/processed/val.json", tokenizer, error_types)

    tc = config["training"]
    train_loader = DataLoader(train_dataset, batch_size=tc["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=tc["batch_size"])

    optimizer = AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    total_steps = len(train_loader) * tc["epochs"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = FocalLoss()

    best_f1 = 0.0

    for epoch in range(tc["epochs"]):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch["labels"].numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        metrics = compute_metrics(all_preds, all_labels)

        logger.info(f"Epoch {epoch+1}/{tc['epochs']} | "
                    f"Loss: {total_loss/len(train_loader):.4f} | "
                    f"F1: {metrics['f1_micro']:.4f}")

        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            factory.save_model(model, "models_saved/best_mentor_model.pt")
            logger.info(f"✅ New best model saved (F1: {best_f1:.4f})")


if __name__ == "__main__":
    train()
```

### 11.4 — `scripts/train.sh`

```bash
#!/bin/bash
set -e
echo "🚀 Starting PolyMentor training..."
python -m src.training.train
echo "✅ Training complete. Best model saved to models_saved/best_mentor_model.pt"
```

```bash
chmod +x scripts/train.sh
```

---

## 12. Phase 11 — Evaluation

### 12.1 — `src/evaluation/evaluate.py`

```python
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.model_factory import ModelFactory
from src.data_pipeline.tokenizer import CodeTokenizer
from src.training.train import CodeErrorDataset
from src.training.metrics import compute_metrics
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate():
    model_config = load_config("configs/model_config.yaml")

    with open("data/labels/error_types.json") as f:
        error_types = json.load(f)

    factory = ModelFactory()
    model = factory.load_error_detector("models_saved/best_mentor_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = CodeTokenizer(model_config["model"]["backbone"])
    test_dataset = CodeErrorDataset("data/processed/test.json", tokenizer, error_types)
    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    logger.info("=" * 50)
    logger.info("📊 EVALUATION RESULTS")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)

    return metrics


if __name__ == "__main__":
    evaluate()
```

### 12.2 — `scripts/evaluate.sh`

```bash
#!/bin/bash
set -e
echo "📊 Evaluating PolyMentor..."
python -m src.evaluation.evaluate
```

```bash
chmod +x scripts/evaluate.sh
```

---

## 13. Phase 12 — Inference Pipeline (Public API)

This is the single public entrypoint. All components converge here.

### 13.1 — `src/inference/pipeline.py`

```python
import torch
from dataclasses import dataclass
from src.models.model_factory import ModelFactory
from src.data_pipeline.tokenizer import CodeTokenizer
from src.reasoning_engine.error_classifier import ErrorClassifier
from src.reasoning_engine.hint_system import HintSystem
from src.reasoning_engine.feedback_scorer import FeedbackScorer
from src.reasoning_engine.explanation_generator import ExplanationGenerator
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MentorResult:
    error_types: list
    primary_error: str
    explanation: str
    hints: list
    concept_taught: str
    quality_score: int


class PolyMentorPipeline:
    """
    The single public entrypoint for PolyMentor.
    Analyzes code and returns structured mentor feedback.
    """

    def __init__(self, model_path: str, config_dir: str = "configs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = load_config(f"{config_dir}/model_config.yaml")

        # Load error detector
        factory = ModelFactory(f"{config_dir}/model_config.yaml")
        self.detector = factory.load_error_detector(model_path)
        self.detector.to(self.device)
        self.detector.eval()

        # Load supporting components
        self.tokenizer = CodeTokenizer(model_config["model"]["backbone"])
        self.error_classifier = ErrorClassifier()
        self.hint_system = HintSystem()
        self.scorer = FeedbackScorer()
        self.explainer = ExplanationGenerator()

        logger.info("✅ PolyMentorPipeline ready")

    @classmethod
    def from_pretrained(cls, model_path: str) -> "PolyMentorPipeline":
        return cls(model_path=model_path)

    def analyze(self, code: str, language: str = "python", level: str = "beginner") -> MentorResult:
        """
        Analyze a code snippet and return full mentor feedback.

        Args:
            code: Source code string
            language: Programming language (python, javascript, java, cpp)
            level: User skill level (beginner, intermediate, advanced)

        Returns:
            MentorResult with error types, explanation, hints, concept, and quality score
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(code, language)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        # Detect errors
        pred_vector = self.detector.predict(input_ids, attention_mask)
        binary = pred_vector.squeeze(0).tolist()
        error_labels = self.error_classifier.decode(binary)

        if not error_labels:
            return MentorResult(
                error_types=[],
                primary_error="none",
                explanation="No errors detected! Your code looks clean.",
                hints=[],
                concept_taught="N/A",
                quality_score=self.scorer.score(code, language)
            )

        primary = self.error_classifier.get_primary_error(error_labels)
        explanation = self.explainer.explain(primary)
        hints = self.hint_system.get_hints(primary, level)
        concepts = self.error_classifier.get_concepts(error_labels)
        quality = self.scorer.score(code, language)

        return MentorResult(
            error_types=error_labels,
            primary_error=primary,
            explanation=explanation,
            hints=hints,
            concept_taught=concepts[0] if concepts else "General Programming",
            quality_score=quality
        )
```

### 13.2 — Quick Test

```python
from src.inference.pipeline import PolyMentorPipeline

mentor = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")

result = mentor.analyze("""
for i in range(10):
    print(i)
    if i = 5:
        break
""", language="python", level="beginner")

print("Error types:    ", result.error_types)
print("Primary error:  ", result.primary_error)
print("Explanation:    ", result.explanation)
print("Hint:           ", result.hints[0])
print("Concept taught: ", result.concept_taught)
print("Quality score:  ", result.quality_score)
```

---

## 14. Phase 13 — Interactive Tutor Mode

### 14.1 — `src/inference/tutor_mode.py`

```python
from src.inference.pipeline import PolyMentorPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TutorSession:
    """
    Interactive tutoring session.
    The tutor works through errors one at a time,
    giving progressive hints if the user asks.
    """

    def __init__(self, pipeline: PolyMentorPipeline, language: str = "python",
                 level: str = "beginner"):
        self.pipeline = pipeline
        self.language = language
        self.level = level

    def start(self):
        print("\n" + "=" * 60)
        print("🧠 PolyMentor — Interactive Tutor Mode")
        print("Type your code, then press Enter twice to submit.")
        print("Type 'hint' for the next hint, 'quit' to exit.")
        print("=" * 60 + "\n")

        hints_used = []
        current_hints = []
        hint_index = 0

        while True:
            print("📝 Paste your code (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            code = "\n".join(lines)

            if code.lower() == "quit":
                print("👋 Session ended. Keep coding!")
                break

            result = self.pipeline.analyze(code, self.language, self.level)
            current_hints = result.hints
            hint_index = 0

            if not result.error_types:
                print("\n✅ No errors found! Code quality score:", result.quality_score)
                continue

            print(f"\n🔍 Detected: {', '.join(result.error_types)}")
            print(f"📚 Concept: {result.concept_taught}")
            print(f"💬 Explanation: {result.explanation}")
            print(f"\n💡 First hint: {current_hints[0] if current_hints else 'No hints available.'}")
            print(f"📊 Code quality: {result.quality_score}/100\n")

            while True:
                action = input("Type 'hint' for next hint, 'new' for new code, 'quit' to exit: ").strip().lower()
                if action == "hint":
                    hint_index += 1
                    if hint_index < len(current_hints):
                        print(f"\n💡 {current_hints[hint_index]}\n")
                    else:
                        print("\n✅ No more hints. Try fixing the error and resubmit!\n")
                elif action == "new":
                    break
                elif action == "quit":
                    return


def run_tutor():
    pipeline = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")
    language = input("Language (python/javascript/java/cpp): ").strip() or "python"
    level = input("Level (beginner/intermediate/advanced): ").strip() or "beginner"
    session = TutorSession(pipeline, language, level)
    session.start()


if __name__ == "__main__":
    run_tutor()
```

### 14.2 — `scripts/run_tutor.sh`

```bash
#!/bin/bash
set -e
echo "🧠 Launching PolyMentor Tutor Mode..."
python -m src.inference.tutor_mode
```

```bash
chmod +x scripts/run_tutor.sh
```

---

## 15. Phase 14 — FastAPI Backend

### 15.1 — `src/api/app.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.pipeline import PolyMentorPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="PolyMentor API",
    description="AI-powered coding mentor — error detection, explanation, and hints.",
    version="0.1.0"
)

# Load the model once at startup
pipeline = None


@app.on_event("startup")
def load_model():
    global pipeline
    pipeline = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")
    logger.info("PolyMentor pipeline loaded and ready.")


class AnalyzeRequest(BaseModel):
    code: str
    language: str = "python"
    level: str = "beginner"


class AnalyzeResponse(BaseModel):
    error_types: list
    primary_error: str
    explanation: str
    hints: list
    concept_taught: str
    quality_score: int


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        result = pipeline.analyze(
            code=request.code,
            language=request.language,
            level=request.level
        )
        return AnalyzeResponse(
            error_types=result.error_types,
            primary_error=result.primary_error,
            explanation=result.explanation,
            hints=result.hints,
            concept_taught=result.concept_taught,
            quality_score=result.quality_score
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
```

### 15.2 — Run the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the Swagger UI.

---

## 16. Phase 15 — Testing

### 16.1 — `tests/test_data_pipeline.py`

```python
import pytest
from src.data_pipeline.cleaner import DataCleaner

def test_cleaner_removes_missing_fields():
    cleaner = DataCleaner()
    samples = [{"id": "1", "code": "x = 1"}]  # Missing required fields
    result = cleaner.clean(samples)
    assert len(result) == 0

def test_cleaner_keeps_valid_sample():
    cleaner = DataCleaner()
    sample = {
        "id": "1", "code": "x = 1", "language": "python",
        "error_types": ["syntax_error"], "difficulty": "beginner",
        "explanation": "Test", "hint_steps": ["Step 1"],
        "concept_taught": "Variables"
    }
    result = cleaner.clean([sample])
    assert len(result) == 1
```

### 16.2 — `tests/test_model.py`

```python
import torch
import pytest
from src.models.error_detector import ErrorDetector

def test_error_detector_output_shape():
    model = ErrorDetector(num_labels=9)
    model.eval()
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 9)

def test_error_detector_predict_binary():
    model = ErrorDetector(num_labels=9)
    input_ids = torch.randint(0, 1000, (1, 512))
    attention_mask = torch.ones(1, 512, dtype=torch.long)
    preds = model.predict(input_ids, attention_mask)
    assert set(preds.unique().tolist()).issubset({0, 1})
```

### 16.3 — `tests/test_inference.py`

```python
from src.reasoning_engine.error_classifier import ErrorClassifier
from src.reasoning_engine.hint_system import HintSystem
from src.reasoning_engine.feedback_scorer import FeedbackScorer

def test_error_classifier_decode():
    clf = ErrorClassifier()
    binary = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = clf.decode(binary)
    assert "syntax_error" in labels

def test_hint_system_returns_steps():
    hs = HintSystem()
    hints = hs.get_hints("syntax_error", "beginner")
    assert len(hints) > 0
    assert "Step" in hints[0]

def test_feedback_scorer_range():
    scorer = FeedbackScorer()
    code = "x = 1\nprint(x)"
    score = scorer.score(code)
    assert 0 <= score <= 100
```

### 16.4 — Run Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Linting
flake8 src/ --max-line-length=100
black src/ --check
```

---

## 17. Phase 16 — Docker & Deployment

### 17.1 — `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 17.2 — `docker-compose.yml`

```yaml
version: "3.9"

services:
  polymentor:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models_saved:/app/models_saved
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

### 17.3 — Build & Run

```bash
# Build the image
docker build -t polymentor:latest .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 17.4 — Deploy to AWS

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name polymentor --region us-east-1

# 2. Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag and push image
docker tag polymentor:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/polymentor:latest
docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/polymentor:latest

# 4. Deploy via ECS or EC2 (configure task definition with image URI above)
```

---

## 18. Full Build Checklist

| Phase | Task | Status |
|---|---|---|
| 1 | Python venv, pip, `requirements.txt`, `setup.py`, `.gitignore` | ☐ |
| 2 | Full folder structure + `__init__.py` files | ☐ |
| 3 | YAML configs + `config_loader.py` + `logger.py` | ☐ |
| 4 | Error taxonomy JSON files + raw dataset download | ☐ |
| 5 | `collector.py`, `cleaner.py`, `tokenizer.py`, `dataset_builder.py` | ☐ |
| 5 | Run `bash scripts/preprocess.sh` | ☐ |
| 6 | Clone Tree-sitter grammars + `ast_parser.py` + `code_embeddings.py` | ☐ |
| 7 | `error_detector.py` + `model_factory.py` | ☐ |
| 8 | `error_classifier.py` + `feedback_scorer.py` + `hint_system.py` + `explanation_generator.py` | ☐ |
| 9 | `explanation_model.py` + fine-tuning script | ☐ |
| 10 | `loss_functions.py` + `metrics.py` + `train.py` | ☐ |
| 10 | Run `bash scripts/train.sh` | ☐ |
| 11 | `evaluate.py` + run `bash scripts/evaluate.sh` | ☐ |
| 12 | `pipeline.py` + quick inference test | ☐ |
| 13 | `tutor_mode.py` + run `bash scripts/run_tutor.sh` | ☐ |
| 14 | `src/api/app.py` + `uvicorn` local test | ☐ |
| 15 | All tests pass: `pytest tests/ -v` | ☐ |
| 16 | `Dockerfile` + `docker-compose.yml` + AWS ECR push | ☐ |

---

<div align="center">

Built with 🎓 by the QuantumLogics team.

_PolyMentor — Don't just fix code. Understand it._

</div>
