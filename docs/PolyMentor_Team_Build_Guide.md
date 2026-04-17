# 🧠 PolyMentor — Full Team ML Build Guide
### Zero Budget · 5 Engineers · Production-Ready AI Mentor

> **Current Status:** Repo running, data pipeline working, API has one import error to fix.
> **Goal:** Train and deploy a real ML model for free using cloud GPUs.

---

## 🚨 Fix This RIGHT NOW (Before Anything Else)

The API is crashing because `pipeline.py` imports `setup_logger` but `logger.py` only has `get_logger`.

Open `src/inference/pipeline.py` and find:
```python
from src.utils.logger import setup_logger
```
Replace with:
```python
from src.utils.logger import get_logger
```

Then find every place `setup_logger(...)` is called in that file and change it to `get_logger(...)`.

After saving, run:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` — if you see the Swagger UI, the API is fixed. ✅

Also install the missing numpy:
```bash
pip install numpy
```

---

## 👥 Team Structure — Who Does What

With 5 ML engineers, split work in parallel so nobody is blocked.

| Engineer | Role | Owns |
|---|---|---|
| **Engineer 1** | ML Lead | Model architecture, training loop, evaluation |
| **Engineer 2** | Data Engineer | Dataset collection, cleaning, augmentation |
| **Engineer 3** | NLP Engineer | Explanation model, hint generation, fine-tuning |
| **Engineer 4** | Backend Engineer | FastAPI, inference pipeline, testing |
| **Engineer 5** | DevOps/MLOps | Free GPU setup, experiment tracking, deployment |

---

## 📋 Master Roadmap

```
Week 1  → Fix bugs + expand dataset + set up free GPU environments
Week 2  → Train baseline error detector (TF-IDF + Logistic Regression)
Week 3  → Fine-tune CodeBERT error detector
Week 4  → Fine-tune CodeT5 explanation model
Week 5  → Connect all components into working pipeline
Week 6  → Evaluate, fix weak spots, add more data
Week 7  → Deploy to Hugging Face Spaces (free)
Week 8  → Polish, document, demo
```

---

## Phase 1 — Fix All Remaining Bugs

### 1.1 — Fix `pipeline.py` import (described above)

### 1.2 — Fix numpy warning
```bash
pip install numpy
```

### 1.3 — Freeze your working dependencies
After all fixes, run:
```bash
pip freeze > requirements.txt
```
Commit this so all 5 engineers have identical environments.

### 1.4 — Verify the API works end to end
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000/docs`, click `/analyze`, click "Try it out", paste:
```json
{
  "code": "for i in range(10):\n    if i = 5:\n        break",
  "language": "python",
  "level": "beginner"
}
```
Click Execute. You should get a response with error types, explanation, hints, and a quality score.

### 1.5 — Push everything to GitHub and share with the team
```bash
git add .
git commit -m "fix: import errors resolved, dataset pipeline working"
git push origin main
```

---

## Phase 2 — Expand the Dataset (Engineer 2's Job)

**You need at least 500 samples to train anything meaningful.**
**Target: 2,000+ samples for a good model.**

### 2.1 — Dataset Sources (All Free)

#### Source A — CodeNet (IBM) — Best Source
This is 14 million code samples with execution results. You don't need all of it.

1. Go to: https://developer.ibm.com/exchanges/data/all/project-codenet/
2. Register for a free IBM account
3. Download just the Python subset (~500MB instead of 8GB):
```bash
# Download only Python subset
wget "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz"
tar -xzf Project_CodeNet_Python800.tar.gz -C data/raw/code_datasets/
```

#### Source B — Stack Overflow Dataset (Free on Kaggle)
1. Go to: https://www.kaggle.com/datasets/stackoverflow/stackoverflow
2. Sign in with Google (free)
3. Download `Questions.csv` and `Answers.csv`
4. Filter for Python/JS/Java tags with accepted answers — these are your explanation pairs

```python
# scripts/process_stackoverflow.py
import pandas as pd
import json

questions = pd.read_csv("data/raw/stackoverflow/Questions.csv", encoding="latin-1")
answers = pd.read_csv("data/raw/stackoverflow/Answers.csv", encoding="latin-1")

# Keep only accepted answers
merged = questions.merge(answers, left_on="AcceptedAnswerId", right_on="Id")

# Filter by tag
python_qs = merged[merged["Tags_x"].str.contains("python", na=False)]

samples = []
for _, row in python_qs.head(1000).iterrows():
    samples.append({
        "id": f"so_{row['Id_x']}",
        "code": row["Body_x"][:500],  # truncate
        "language": "python",
        "error_types": ["logical_error"],  # label manually or via rules
        "difficulty": "intermediate",
        "explanation": row["Body_y"][:300],
        "hint_steps": ["Step 1: Read the accepted answer carefully."],
        "concept_taught": "General Debugging",
        "quality_score": 70
    })

with open("data/raw/stackoverflow_samples.json", "w") as f:
    json.dump(samples, f, indent=2)
```

#### Source C — GitHub Code Search (Free API)
Use GitHub's free API to fetch buggy code samples from issues and PRs:

```python
# scripts/github_collector.py
import requests
import json
import time

GITHUB_TOKEN = "your_free_github_token"  # free at github.com/settings/tokens
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

queries = [
    "SyntaxError python language:python",
    "IndexError python language:python",
    "TypeError python language:python",
]

samples = []
for query in queries:
    url = f"https://api.github.com/search/code?q={query}&per_page=30"
    resp = requests.get(url, headers=headers).json()
    for item in resp.get("items", []):
        # Fetch file content
        content_url = item["git_url"]
        content_resp = requests.get(content_url, headers=headers).json()
        # ... parse and add to samples
    time.sleep(2)  # respect rate limits
```

#### Source D — HuggingFace Datasets (Best — No Download Needed)
These load directly in Python, no manual download:

```python
# scripts/load_hf_datasets.py
from datasets import load_dataset
import json

# CodeSearchNet — 2M code samples, free
dataset = load_dataset("code_search_net", "python", split="train[:2000]")

samples = []
for i, item in enumerate(dataset):
    samples.append({
        "id": f"csn_{i}",
        "code": item["func_code_string"],
        "language": "python",
        "error_types": [],  # these are clean samples — useful for negative examples
        "difficulty": "intermediate",
        "explanation": item["func_documentation_string"],
        "hint_steps": [],
        "concept_taught": "Clean Code",
        "quality_score": 85
    })

with open("data/raw/codesearchnet_samples.json", "w") as f:
    json.dump(samples, f, indent=2)
```

```bash
pip install datasets
python scripts/load_hf_datasets.py
```

#### Source E — Manually Write 50 High Quality Samples
Split among the team — each engineer writes 10 samples covering their specialty.
These high-quality, manually verified samples are the most valuable training data you have.
Use the schema from `data/raw/starter_samples.json` as the template.

### 2.2 — Auto-Labeling Script
Once you have raw code, use this to auto-label error types using Python's built-in compiler:

```python
# scripts/auto_label.py
import ast
import json
import py_compile
import tempfile
import os


def detect_errors(code: str, language: str = "python") -> list:
    """Auto-detect errors in Python code using the AST."""
    errors = []

    if language != "python":
        return ["unknown"]

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError:
        errors.append("syntax_error")
        return errors  # Can't check further if syntax is broken

    # Check for common patterns
    tree = ast.parse(code)
    for node in ast.walk(tree):
        # Division without zero check
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            errors.append("division_by_zero")

        # Infinite while True
        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                errors.append("infinite_loop")

    if not errors:
        errors.append("bad_practice")  # Default for clean-looking code

    return list(set(errors))


# Load your raw samples and auto-label them
with open("data/raw/raw_unlabeled.json") as f:
    raw = json.load(f)

labeled = []
for sample in raw:
    if not sample.get("error_types"):
        sample["error_types"] = detect_errors(sample["code"], sample.get("language", "python"))
    labeled.append(sample)

with open("data/raw/auto_labeled.json", "w") as f:
    json.dump(labeled, f, indent=2)

print(f"Labeled {len(labeled)} samples")
```

### 2.3 — Rebuild Dataset After Adding Sources
Every time you add new raw data:
```bash
python -m src.data_pipeline.dataset_builder
```

Target splits:
- `train.json` — 1,600+ samples
- `val.json` — 200+ samples
- `test.json` — 200+ samples

---

## Phase 3 — Free GPU Setup (Engineer 5's Job)

**You have 3 excellent free GPU options. Set up ALL THREE so the team can always have one available.**

### Option A — Google Colab (Best for beginners, free T4 GPU)

1. Go to https://colab.research.google.com
2. New notebook → Runtime → Change runtime type → T4 GPU
3. Mount your Google Drive to save model checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone your repo
!git clone https://github.com/your-username/PolyMentor.git
%cd PolyMentor

# Install deps
!pip install -r requirements.txt
!pip install -e .

# Copy data from Drive
!cp -r /content/drive/MyDrive/PolyMentor/data ./data
!cp -r /content/drive/MyDrive/PolyMentor/configs ./configs
```

**Colab free tier gives you:** T4 GPU (16GB VRAM), ~12 hours per session.
**Limitation:** Sessions disconnect after ~12 hours. Save checkpoints frequently.

### Option B — Kaggle Notebooks (30 hours/week free GPU, better than Colab)

1. Go to https://www.kaggle.com → Sign in with Google
2. Create → New Notebook
3. Settings (right panel) → Accelerator → GPU T4 x2
4. You get **30 free GPU hours per week** — more reliable than Colab

```python
# In Kaggle notebook
!git clone https://github.com/your-username/PolyMentor.git
%cd PolyMentor
!pip install -r requirements.txt
!pip install -e .
```

Upload your dataset as a Kaggle Dataset (private), then add it to the notebook.

### Option C — Hugging Face Spaces ZeroGPU (Free A100 GPU)

ZeroGPU gives you free access to A100 GPUs (much more powerful):

1. Go to https://huggingface.co/spaces
2. Create a new Space → SDK: Gradio → Hardware: ZeroGPU (Free)
3. Push your training code as a Space

```python
# app.py for HF Space — triggers training
import gradio as gr
import subprocess

def train_model():
    result = subprocess.run(
        ["python", "-m", "src.training.train"],
        capture_output=True, text=True
    )
    return result.stdout

gr.Interface(fn=train_model, inputs=[], outputs="text").launch()
```

### Option D — Lightning AI (Free Tier, Best for ML)

1. Go to https://lightning.ai
2. Sign up free → New Studio
3. You get a free GPU studio with persistent storage
4. Clone your repo and train directly — no time limits on CPU, GPU credits available

### Saving Checkpoints to Google Drive (Critical)

Add this to your training script so you never lose progress:

```python
# At the end of src/training/train.py, after saving best model:
import shutil
import os

drive_path = "/content/drive/MyDrive/PolyMentor/models_saved/"
if os.path.exists("/content/drive"):  # We are on Colab
    os.makedirs(drive_path, exist_ok=True)
    shutil.copy("models_saved/best_mentor_model.pt", drive_path)
    print("✅ Checkpoint saved to Google Drive")
```

---

## Phase 4 — Experiment 1: Baseline Model (Engineer 1, Week 2)

Before touching CodeBERT, build a simple baseline. This is fast, runs on CPU, and proves your pipeline works.

### 4.1 — `experiments/exp_01_tfidf_baseline/baseline.py`

```python
"""
Baseline: TF-IDF features + Logistic Regression for multi-label error detection.
No GPU needed. Trains in seconds. Good for validating the pipeline.
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import pickle
from pathlib import Path


def load_data(path):
    with open(path) as f:
        return json.load(f)


def run_baseline():
    train = load_data("data/processed/train.json")
    val = load_data("data/processed/val.json")
    test = load_data("data/processed/test.json")

    # Features: raw code text
    X_train = [s["code"] for s in train]
    X_val = [s["code"] for s in val]
    X_test = [s["code"] for s in test]

    # Labels: multi-label binarized
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([s["error_types"] for s in train])
    y_val = mlb.transform([s["error_types"] for s in val])
    y_test = mlb.transform([s["error_types"] for s in test])

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), analyzer="char_wb")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # Train multi-label classifier
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    val_preds = clf.predict(X_val_tfidf)
    test_preds = clf.predict(X_test_tfidf)

    print("=" * 50)
    print("BASELINE RESULTS")
    print(f"Val  F1 (micro): {f1_score(y_val, val_preds, average='micro', zero_division=0):.4f}")
    print(f"Test F1 (micro): {f1_score(y_test, test_preds, average='micro', zero_division=0):.4f}")
    print("=" * 50)

    # Save baseline
    Path("models_saved").mkdir(exist_ok=True)
    with open("models_saved/baseline_model.pkl", "wb") as f:
        pickle.dump({"clf": clf, "tfidf": tfidf, "mlb": mlb}, f)
    print("Baseline model saved to models_saved/baseline_model.pkl")


if __name__ == "__main__":
    run_baseline()
```

Run it:
```bash
python experiments/exp_01_tfidf_baseline/baseline.py
```

This runs in seconds on CPU and gives you your first benchmark F1 score. The CodeBERT model (Phase 5) needs to beat this.

---

## Phase 5 — Experiment 2: CodeBERT Fine-Tuning (Engineer 1 + 5, Week 3)

This is the main model. Run this on Colab/Kaggle GPU.

### 5.1 — Create the Colab Training Notebook

Save as `notebooks/train_codebert.ipynb` and run on Colab with T4 GPU:

**Cell 1 — Setup:**
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!git clone https://github.com/your-username/PolyMentor.git
%cd PolyMentor

# Install
!pip install -r requirements.txt -q
!pip install -e . -q

# Copy data
import shutil
shutil.copytree('/content/drive/MyDrive/PolyMentor/data', './data')
shutil.copytree('/content/drive/MyDrive/PolyMentor/configs', './configs')

print("Setup complete!")
```

**Cell 2 — Verify GPU:**
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Cell 3 — Train:**
```python
!python -m src.training.train
```

**Cell 4 — Save to Drive:**
```python
import shutil
shutil.copy(
    'models_saved/best_mentor_model.pt',
    '/content/drive/MyDrive/PolyMentor/models_saved/best_mentor_model.pt'
)
print("✅ Model saved to Drive!")
```

### 5.2 — Training Tips for Free GPU

**Save checkpoints every epoch** (add to `train.py`):
```python
# Inside the epoch loop, after computing val metrics:
checkpoint_path = f"models_saved/checkpoint_epoch_{epoch+1}.pt"
factory.save_model(model, checkpoint_path)

# Also save to Drive if on Colab
if os.path.exists("/content/drive"):
    shutil.copy(checkpoint_path, f"/content/drive/MyDrive/PolyMentor/{checkpoint_path}")
```

**Reduce batch size if you get OOM (Out of Memory) errors:**
In `configs/training_config.yaml`:
```yaml
training:
  batch_size: 8          # reduce from 16 if OOM
  gradient_accumulation_steps: 4   # this simulates batch_size=32
```

**Use mixed precision to fit larger batches:**
```python
# In train.py, use torch's automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Phase 6 — Experiment 3: Explanation Model (Engineer 3, Week 4)

### 6.1 — Fine-tune CodeT5 for Explanation Generation

Run this on Kaggle (30h/week) — it needs about 2-3 hours of GPU time.

```python
# notebooks/train_explanation_model.ipynb

from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json

# Load data
with open("data/processed/train.json") as f:
    train_data = json.load(f)

# Filter: only samples that have real explanations (not placeholder)
train_data = [s for s in train_data if len(s.get("explanation", "")) > 30]

MODEL = "Salesforce/codet5-small"  # Use 'small' — faster, less VRAM, still good

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)


def preprocess(sample):
    error_label = sample["error_types"][0] if sample["error_types"] else "unknown"
    # Input: error type + code
    prompt = f"explain: [{error_label}] {sample['code'][:400]}"
    # Target: explanation
    target = sample["explanation"]

    enc = tokenizer(prompt, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=128, truncation=True, padding="max_length")

    enc["labels"] = labels["input_ids"]
    return enc


dataset = Dataset.from_list(train_data)
tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

args = Seq2SeqTrainingArguments(
    output_dir="models_saved/explanation_model",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    predict_with_generate=True,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    report_to="none",  # no wandb
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

trainer.train()

# Save final model
model.save_pretrained("models_saved/explanation_model/final")
tokenizer.save_pretrained("models_saved/explanation_model/final")
print("✅ Explanation model saved!")
```

### 6.2 — Update Pipeline to Use Fine-tuned Explanation Model

After training, update `src/inference/pipeline.py` to load the fine-tuned model:

```python
# In PolyMentorPipeline.__init__, replace the rule-based explainer with:
from src.models.explanation_model import ExplanationModel

self.neural_explainer = ExplanationModel.load_fine_tuned(
    "models_saved/explanation_model/final"
)

# In analyze(), replace:
explanation = self.explainer.explain(primary)
# With:
explanation = self.neural_explainer.generate(code, primary)
```

---

## Phase 7 — Experiment Tracking (Engineer 5)

Track all experiments for free using MLflow locally:

### 7.1 — Install and Set Up MLflow
```bash
pip install mlflow
```

### 7.2 — Add Tracking to `train.py`

```python
import mlflow

with mlflow.start_run(run_name=f"codebert_epoch_{epoch+1}"):
    mlflow.log_param("batch_size", tc["batch_size"])
    mlflow.log_param("learning_rate", tc["learning_rate"])
    mlflow.log_param("epochs", tc["epochs"])
    mlflow.log_metric("val_f1_micro", metrics["f1_micro"], step=epoch)
    mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
```

### 7.3 — View Results Dashboard
```bash
mlflow ui
```
Open `http://localhost:5000` — a full experiment dashboard, completely free.

---

## Phase 8 — Evaluation Strategy (Engineer 1)

### 8.1 — What Numbers to Target

| Metric | Baseline (TF-IDF) | Target (CodeBERT) | Good |
|---|---|---|---|
| F1 Micro | ~0.40 | >0.70 | >0.80 |
| F1 Macro | ~0.30 | >0.60 | >0.75 |
| Precision | ~0.45 | >0.72 | >0.82 |
| Recall | ~0.38 | >0.68 | >0.78 |

### 8.2 — Per-Error Analysis

Add this to `src/evaluation/error_analysis.py`:

```python
import json
import numpy as np
from sklearn.metrics import classification_report

def analyze_per_error(predictions, labels, error_types_path="data/labels/error_types.json"):
    with open(error_types_path) as f:
        error_types = json.load(f)

    label_names = list(error_types.keys())
    report = classification_report(
        labels, predictions,
        target_names=label_names,
        zero_division=0
    )
    print(report)
    return report
```

This shows you which error types the model is weak on so you know where to add more training data.

### 8.3 — Error Analysis Workflow

When an error type has F1 < 0.5:
1. Count how many samples of that type are in training data
2. If fewer than 50 → add more samples of that type
3. Retrain and re-evaluate

---

## Phase 9 — Free Deployment (Engineer 4 + 5, Week 7)

### Option A — Hugging Face Spaces (BEST — Free, Public URL, No Credit Card)

**Step 1:** Create a free HF account at https://huggingface.co

**Step 2:** Upload your trained model to HF Hub:
```bash
pip install huggingface_hub

# Login (creates a free token)
huggingface-cli login

# Upload model weights
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="models_saved/best_mentor_model.pt",
    path_in_repo="best_mentor_model.pt",
    repo_id="your-username/polymentor",
    repo_type="model"
)
```

**Step 3:** Create a Gradio demo Space (`app.py`):
```python
import gradio as gr
from huggingface_hub import hf_hub_download
from src.inference.pipeline import PolyMentorPipeline
import os

# Download model from HF Hub
model_path = hf_hub_download(
    repo_id="your-username/polymentor",
    filename="best_mentor_model.pt"
)

pipeline = PolyMentorPipeline.from_pretrained(model_path)


def analyze_code(code, language, level):
    if not code.strip():
        return "Please paste some code.", "", "", "", 0

    result = pipeline.analyze(code, language, level)

    if not result.error_types:
        return "✅ No errors detected!", "", "", "", result.quality_score

    hints_text = "\n".join(result.hints)
    return (
        f"🔍 {', '.join(result.error_types)}",
        result.explanation,
        hints_text,
        f"📚 {result.concept_taught}",
        result.quality_score
    )


with gr.Blocks(title="PolyMentor — AI Coding Mentor") as demo:
    gr.Markdown("# 🧠 PolyMentor\nPaste your code and get AI-powered feedback.")

    with gr.Row():
        with gr.Column():
            code_input = gr.Code(label="Your Code", language="python", lines=15)
            language = gr.Dropdown(
                ["python", "javascript", "java", "cpp"],
                value="python", label="Language"
            )
            level = gr.Dropdown(
                ["beginner", "intermediate", "advanced"],
                value="beginner", label="Your Level"
            )
            submit_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column():
            error_output = gr.Textbox(label="Detected Errors")
            explanation_output = gr.Textbox(label="Explanation", lines=4)
            hints_output = gr.Textbox(label="Step-by-Step Hints", lines=6)
            concept_output = gr.Textbox(label="Concept Taught")
            score_output = gr.Number(label="Code Quality Score / 100")

    submit_btn.click(
        analyze_code,
        inputs=[code_input, language, level],
        outputs=[error_output, explanation_output, hints_output, concept_output, score_output]
    )

demo.launch()
```

**Step 4:** Push to HF Spaces:
```bash
# Create a new Space repo
huggingface-cli repo create polymentor --type space --space_sdk gradio

# Push your code
git remote add hf https://huggingface.co/spaces/your-username/polymentor
git push hf main
```

Your app will be live at: `https://huggingface.co/spaces/your-username/polymentor` 🎉

### Option B — Render.com (Free REST API hosting)

1. Push your repo to GitHub
2. Go to https://render.com → New → Web Service
3. Connect GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt && pip install -e .`
   - **Start Command:** `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
5. Add environment variable: `MODEL_PATH=models_saved/best_mentor_model.pt`

Free tier gives you a live `https://polymentor.onrender.com` URL. Auto-deploys on every git push.

---

## Phase 10 — Team Git Workflow

### 10.1 — Branch Strategy

```
main          ← stable, always working
dev           ← integration branch
feature/data-pipeline    ← Engineer 2
feature/codebert-model   ← Engineer 1
feature/explanation-model ← Engineer 3
feature/api-backend       ← Engineer 4
feature/deployment        ← Engineer 5
```

### 10.2 — Daily Workflow

```bash
# Start of day — sync with team
git checkout dev
git pull origin dev

# Create your feature branch
git checkout -b feature/your-task

# Work, then commit
git add .
git commit -m "feat: add auto-labeling script for Python syntax errors"

# Push and open PR to dev
git push origin feature/your-task
# Open PR on GitHub → request review from 1 teammate
```

### 10.3 — PR Rules

- Every PR must be reviewed by at least 1 other engineer
- All tests must pass before merging: `pytest tests/ -v`
- Never push directly to `main`
- Merge to `dev` first, test together, then merge `dev` → `main` weekly

---

## Phase 11 — Full Test Suite (Engineer 4)

### 11.1 — `tests/test_data_pipeline.py`
(Already written in the build guide — make sure it passes)

### 11.2 — `tests/test_model.py`
(Already written — make sure it passes)

### 11.3 — `tests/test_inference.py`
(Already written — make sure it passes)

### 11.4 — `tests/test_api.py`

```python
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_returns_result():
    response = client.post("/analyze", json={
        "code": "for i in range(10):\n    if i = 5:\n        break",
        "language": "python",
        "level": "beginner"
    })
    assert response.status_code == 200
    data = response.json()
    assert "error_types" in data
    assert "explanation" in data
    assert "hints" in data
    assert "quality_score" in data


def test_analyze_empty_code():
    response = client.post("/analyze", json={
        "code": "",
        "language": "python",
        "level": "beginner"
    })
    assert response.status_code == 200


def test_analyze_clean_code():
    response = client.post("/analyze", json={
        "code": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
        "language": "python",
        "level": "beginner"
    })
    assert response.status_code == 200
```

Run the full suite:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📊 Progress Tracker

Copy this into your team Notion / Google Doc and update daily:

| Task | Owner | Status | Notes |
|---|---|---|---|
| Fix `setup_logger` import error | Eng 4 | ☐ | |
| Install numpy | All | ☐ | `pip install numpy` |
| Push fixed code to GitHub | Eng 4 | ☐ | |
| All 5 engineers clone repo | All | ☐ | |
| Collect 500+ samples | Eng 2 | ☐ | Use HF datasets first |
| Collect 2000+ samples | Eng 2 | ☐ | Add SO + CodeNet |
| Set up Colab training notebook | Eng 5 | ☐ | |
| Set up Kaggle training notebook | Eng 5 | ☐ | |
| Run baseline TF-IDF model | Eng 1 | ☐ | Target: F1 > 0.40 |
| Fine-tune CodeBERT | Eng 1 | ☐ | Target: F1 > 0.70 |
| Fine-tune CodeT5 explanation | Eng 3 | ☐ | |
| Connect explanation model to pipeline | Eng 3 | ☐ | |
| All API tests passing | Eng 4 | ☐ | |
| MLflow tracking set up | Eng 5 | ☐ | |
| Deploy to HF Spaces | Eng 5 | ☐ | |
| Deploy REST API to Render | Eng 4 | ☐ | |
| Public demo URL live | All | ☐ | 🎉 |

---

## 🛠️ Quick Reference — Commands You'll Use Every Day

```bash
# Activate virtual environment
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows Git Bash

# Rebuild dataset
python -m src.data_pipeline.dataset_builder

# Run baseline model
python experiments/exp_01_tfidf_baseline/baseline.py

# Run full training (local CPU — slow but works for testing)
python -m src.training.train

# Start API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
pytest tests/ -v

# Check code quality
flake8 src/ --max-line-length=100

# View experiment dashboard
mlflow ui

# Sync with team
git pull origin dev
```

---

## ⚠️ Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'setup_logger'` | Wrong function name | Change to `get_logger` in pipeline.py |
| `ModuleNotFoundError: No module named 'torch'` | torch not in venv | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: No module named 'numpy'` | numpy not installed | `pip install numpy` |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` to 8 in training_config.yaml |
| `val.json has 0 samples` | Too few total samples | Add more data, need 20+ total for splits to work |
| `RuntimeError: Expected all tensors on same device` | CPU/GPU mismatch | Add `.to(device)` to all tensors before model call |
| Colab session disconnects | Free tier limit | Save checkpoint to Drive every epoch |

---

<div align="center">

**PolyMentor Team — Zero Budget, Full Model** 🎓

Built with determination by the QuantumLogics team.

_Every great ML project started with one working API call._

</div>
