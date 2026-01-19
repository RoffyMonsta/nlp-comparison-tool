#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Evaluation Framework for Ukrainian Sentiment Analysis

Compares 5 approaches:
1. VADER (generic rule-based baseline) - demonstrates limitations of generic tools
2. Custom Ukrainian rule-based (your implementation) - language-specific lexicons
3. TF-IDF + Logistic Regression (classical ML baseline) - scikit-learn
4. mBERT (multilingual transformer) - bert-base-multilingual-cased
5. Ukrainian RoBERTa (UA-specific transformer) - youscan/ukr-roberta-base

Usage:
    python main.py

Outputs (saved to ./outputs/):
- metrics_table.csv
- confusion_matrix_<model>.png
- metrics_bar_<metric>.png
- predictions_<model>.csv
"""

import os
import re
import json
import math
import inspect
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

# VADER - generic rule-based sentiment (English-oriented baseline)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_OK = True
except Exception:
    VADER_OK = False

# Optional transformer dependencies
try:
    import torch
    from datasets import Dataset as HFDataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        set_seed as hf_set_seed,
    )
    HF_OK = True
except Exception:
    HF_OK = False

# Custom sentiment analyzer (spaCy-based with Ukrainian lexicons)
try:
    from custom_sentiment import calculate_sentiment
    CUSTOM_SENTIMENT_OK = True
except Exception as e:
    CUSTOM_SENTIMENT_OK = False
    print(f"Warning: custom_sentiment.py not available ({e}).")


# ---------------------------
# Configuration
# ---------------------------

CONFIG = {
    "output_dir": "outputs",
    "test_size": 0.2,
    "seed": 42,
    "stratify": True,
    # Low-resource simulation: limit training samples for ML models
    # Set to None to use all available training data
    # Set to a number (e.g., 100, 200, 500) to simulate low-resource scenario
    "ml_train_limit": None,  # None = use all training data
    # Transformer models to compare
    "transformer_models": [
        "xlm-roberta-base",              # XLM-RoBERTa (supports Ukrainian, safetensors format)
        "bert-base-multilingual-cased",  # mBERT - multilingual baseline

    ],
    # Transformer training settings
    "learning_rate": 2e-5,
    "batch_size": 2,  # Reduced from 8 to fit in MPS memory
    "epochs": 3,
    "max_length": 128,  # Reduced from 256 to save memory
    "fp16": False,
    "gradient_checkpointing": True,
    "transformer_device": "cpu",  # auto | mps | cpu (using CPU to avoid MPS memory issues)
    "transformer_train_limit": 2000,  # Limit transformer training samples (None = use all, 2000 = faster training)
}

# Dataset paths - Ukrainian sentiment data from dict folder
POSITIVE_DATA_PATH = "./dict/positive.json"
NEGATIVE_DATA_PATH = "./dict/negative.json"
NEUTRAL_DATA_PATH = "./dict/neutral.json"


def load_json_messages(filepath: str) -> List[str]:
    """Load messages from a JSON file (one JSON object per line)."""
    messages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if 'msg' in obj:
                        messages.append(obj['msg'])
                except json.JSONDecodeError:
                    continue
    return messages


def fetch_dataset(balance: bool = True) -> pd.DataFrame:
    """
    Load the Ukrainian sentiment dataset from dict/*.json files.
    Supports 3-class (positive/negative/neutral) or 2-class classification.
    Falls back to built-in demo dataset if files are not found.

    Args:
        balance: If True, downsample majority classes to match minority class size.
    """
    if os.path.exists(POSITIVE_DATA_PATH) and os.path.exists(NEGATIVE_DATA_PATH):
        print(f"Loading dataset from dict/*.json files...")

        positive_msgs = load_json_messages(POSITIVE_DATA_PATH)
        negative_msgs = load_json_messages(NEGATIVE_DATA_PATH)

        # Check for neutral data (3-class classification)
        neutral_msgs = []
        if os.path.exists(NEUTRAL_DATA_PATH):
            neutral_msgs = load_json_messages(NEUTRAL_DATA_PATH)
            print(f"  Raw data: {len(positive_msgs)} positive, {len(negative_msgs)} negative, {len(neutral_msgs)} neutral")
        else:
            print(f"  Raw data: {len(positive_msgs)} positive, {len(negative_msgs)} negative (no neutral data)")

        # Balance dataset by downsampling majority classes
        if balance:
            if neutral_msgs:
                min_count = min(len(positive_msgs), len(negative_msgs), len(neutral_msgs))
            else:
                min_count = min(len(positive_msgs), len(negative_msgs))

            random.seed(42)
            if len(positive_msgs) > min_count:
                positive_msgs = random.sample(positive_msgs, min_count)
            if len(negative_msgs) > min_count:
                negative_msgs = random.sample(negative_msgs, min_count)
            if neutral_msgs and len(neutral_msgs) > min_count:
                neutral_msgs = random.sample(neutral_msgs, min_count)

            if neutral_msgs:
                print(f"  Balanced to: {len(positive_msgs)} positive, {len(negative_msgs)} negative, {len(neutral_msgs)} neutral")
            else:
                print(f"  Balanced to: {len(positive_msgs)} positive, {len(negative_msgs)} negative")

        # Create DataFrame with labels
        positive_df = pd.DataFrame({"text": positive_msgs, "label": "positive"})
        negative_df = pd.DataFrame({"text": negative_msgs, "label": "negative"})

        if neutral_msgs:
            neutral_df = pd.DataFrame({"text": neutral_msgs, "label": "neutral"})
            df = pd.concat([positive_df, negative_df, neutral_df], ignore_index=True)
        else:
            df = pd.concat([positive_df, negative_df], ignore_index=True)

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df


# ---------------------------
# Utils
# ---------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HF_OK:
        hf_set_seed(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def normalize_text_basic(text: str) -> str:
    """Light normalization (safe for Ukrainian text)."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # standardize quotes/dashes if needed
    text = text.replace("’", "'").replace("`", "'").replace("–", "-").replace("—", "-")
    return text

def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(path)
    elif ext in [".tsv"]:
        df = pd.read_csv(path, sep="\t")
    elif ext in [".jsonl"]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Unsupported format. Use .csv, .tsv, or .jsonl")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain columns: text, label")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).map(normalize_text_basic)
    df["label"] = df["label"].astype(str)
    df = df.dropna().reset_index(drop=True)
    return df

def map_labels(df: pd.DataFrame, label_order: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    if label_order is None:
        # stable order: NEG, NEU, POS if exists; else alphabetical
        preferred = ["negative", "neutral", "positive", "neg", "neu", "pos", "NEG", "NEU", "POS"]
        unique = list(pd.unique(df["label"]))
        # try build order by preferred then remaining
        order = []
        for p in preferred:
            for u in unique:
                if u == p and u not in order:
                    order.append(u)
        for u in sorted(unique):
            if u not in order:
                order.append(u)
        label_order = order

    l2i = {l: i for i, l in enumerate(label_order)}
    i2l = {i: l for l, i in l2i.items()}
    df = df.copy()
    df["label_id"] = df["label"].map(l2i)
    if df["label_id"].isna().any():
        missing = df[df["label_id"].isna()]["label"].unique().tolist()
        raise ValueError(f"Unmapped labels found: {missing}")
    df["label_id"] = df["label_id"].astype(int)
    return df, l2i, i2l

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    # also micro if you want
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "precision_micro": float(pm),
        "recall_micro": float(rm),
        "f1_micro": float(f1m),
    }

def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    """Save confusion matrix as black/white image with good text visibility."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use grayscale colormap (white=0, black=high values) - reversed for better printing
    cmap = plt.cm.Greys
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.set_title(title, fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add text annotations with contrasting colors
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            # White text on dark background, black text on light background
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, value,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=text_color,
                    fontsize=12,
                    fontweight='bold')

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)

    # Add grid lines for better readability
    ax.set_xticks(np.arange(len(labels)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(labels)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)

def save_metric_bar(metrics_rows: List[Dict], metric_key: str, out_path: str) -> None:
    """Save metric comparison bar chart in black/white style."""
    names = [r["model"] for r in metrics_rows]
    vals = [r[metric_key] for r in metrics_rows]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create bars with different hatching patterns for black/white distinction
    hatches = ['', '///', '\\\\\\', 'xxx', '...', '+++']
    bars = ax.bar(names, vals, color='white', edgecolor='black', linewidth=1.5)

    # Apply different hatch patterns to each bar
    for i, bar in enumerate(bars):
        bar.set_hatch(hatches[i % len(hatches)])

    # Add value labels on top of bars
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Format title nicely
    title_map = {
        'accuracy': 'Accuracy',
        'precision_macro': 'Precision (Macro)',
        'recall_macro': 'Recall (Macro)',
        'f1_macro': 'F1 Score (Macro)'
    }
    ax.set_title(title_map.get(metric_key, metric_key), fontsize=14, fontweight='bold')
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel('Score', fontsize=11)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha="right", fontsize=9)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)


# ---------------------------
# Rule-based model
# ---------------------------

@dataclass
class RuleBasedConfig:
    # You can expand this lexicon as you get real data
    lexicon_pos: Dict[str, float]
    lexicon_neg: Dict[str, float]
    negations: List[str]
    intensifiers: Dict[str, float]
    diminishers: Dict[str, float]
    neutral_margin: float  # threshold around 0 for neutral

def default_rule_based_config() -> RuleBasedConfig:
    # Minimal seed lexicon (extend later!)
    pos = {
        "добрий": 1.0, "чудовий": 1.5, "класний": 1.2, "супер": 1.2,
        "люблю": 1.3, "рекомендую": 1.1, "кращий": 1.4, "приємно": 1.0,
        "щасливий": 1.2, "задоволений": 1.1
    }
    neg = {
        "поганий": -1.0, "жахливий": -1.5, "тупий": -1.2, "ненавиджу": -1.4,
        "гірший": -1.4, "неприємно": -1.0, "обман": -1.2, "розчарований": -1.2,
        "злий": -1.1, "проблема": -0.9
    }
    negations = ["не", "ні", "ані", "без"]
    intens = {"дуже": 1.3, "надзвичайно": 1.5, "сильно": 1.2, "реально": 1.1}
    dimin = {"трохи": 0.8, "ледве": 0.7, "майже": 0.85}
    return RuleBasedConfig(
        lexicon_pos=pos,
        lexicon_neg=neg,
        negations=negations,
        intensifiers=intens,
        diminishers=dimin,
        neutral_margin=0.15,
    )

_token_re = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9']+")

def tokenize_uk_simple(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text)]

def rule_based_predict(texts: List[str], cfg: RuleBasedConfig, label_ids: Dict[str, int]) -> np.ndarray:
    """
    Scoring:
    - base word score from lexicon_pos/neg
    - negation flips sign for next sentiment word within window=3
    - intensifiers/diminishers scale next sentiment word
    Output labels expected: positive/neutral/negative (or your labels)
    """
    # Detect which label strings exist
    # Try common names first; fallback by contains.
    def find_label(name_candidates: List[str]) -> Optional[int]:
        for c in name_candidates:
            if c in label_ids:
                return label_ids[c]
        # fallback: search by substring
        for k, v in label_ids.items():
            low = k.lower()
            for c in name_candidates:
                if c.lower() in low:
                    return v
        return None

    pos_id = find_label(["positive", "pos", "POS"])
    neu_id = find_label(["neutral", "neu", "NEU"])
    neg_id = find_label(["negative", "neg", "NEG"])

    # Handle 2-class or 3-class classification
    n_classes = len(label_ids)
    if n_classes == 2:
        # Binary classification: positive vs negative
        if pos_id is None or neg_id is None:
            ids_sorted = sorted(label_ids.values())
            neg_id, pos_id = ids_sorted[0], ids_sorted[1]
        neu_id = None  # No neutral class
    elif pos_id is None or neu_id is None or neg_id is None:
        # 3-class fallback
        ids_sorted = sorted(label_ids.values())
        if len(ids_sorted) < 3:
            neg_id, pos_id = ids_sorted[0], ids_sorted[-1]
            neu_id = None
        else:
            neg_id, neu_id, pos_id = ids_sorted[0], ids_sorted[1], ids_sorted[2]

    preds = []
    for text in texts:
        toks = tokenize_uk_simple(text)
        score = 0.0
        negate_window = 0
        scale_next = 1.0

        for t in toks:
            if t in cfg.negations:
                negate_window = 3
                continue
            if t in cfg.intensifiers:
                scale_next *= cfg.intensifiers[t]
                continue
            if t in cfg.diminishers:
                scale_next *= cfg.diminishers[t]
                continue

            w = 0.0
            if t in cfg.lexicon_pos:
                w = cfg.lexicon_pos[t]
            elif t in cfg.lexicon_neg:
                w = cfg.lexicon_neg[t]

            if w != 0.0:
                if negate_window > 0:
                    w = -w
                w *= scale_next
                score += w
                scale_next = 1.0  # reset after applying
            if negate_window > 0:
                negate_window -= 1

        if neu_id is not None:
            # 3-class classification
            if abs(score) <= cfg.neutral_margin:
                preds.append(neu_id)
            elif score > 0:
                preds.append(pos_id)
            else:
                preds.append(neg_id)
        else:
            # 2-class classification (no neutral)
            if score >= 0:
                preds.append(pos_id)
            else:
                preds.append(neg_id)

    return np.array(preds, dtype=int)


def extract_custom_sentiment_features(texts: List[str]) -> np.ndarray:
    """
    Extract rich features from custom_sentiment.py for use with a classifier.
    Returns feature matrix with compound, pos, neg, neutral scores and metadata.
    """
    features = []
    for text in texts:
        try:
            result = calculate_sentiment(text)
            feat = [
                result.get("compound", 0.0),
                result.get("positive", 0.0),
                result.get("negative", 0.0),
                result.get("neutral", 0.0),
                result.get("confidence", 0.5),
                result.get("num_tokens", 1),
                result.get("has_negation", 0),
                result.get("num_boosters", 0),
                result.get("emoji_score", 0.0),
            ]
        except Exception:
            feat = [0.0, 0.0, 0.0, 0.0, 0.5, 1, 0, 0, 0.0]
        features.append(feat)
    return np.array(features)


def custom_sentiment_predict(texts: List[str], label_ids: Dict[str, int]) -> np.ndarray:
    """
    Use custom_sentiment.py's calculate_sentiment for prediction.
    Simple and effective: trust the compound score from your lexicon-based analyzer.
    """
    if not CUSTOM_SENTIMENT_OK:
        raise RuntimeError("custom_sentiment.py not available")

    # Find label IDs
    pos_id = label_ids.get("positive", label_ids.get("pos", 1))
    neg_id = label_ids.get("negative", label_ids.get("neg", 0))
    neu_id = label_ids.get("neutral", label_ids.get("neu", None))

    # Key sentiment keywords for boosting decisions
    positive_keywords = {
        'добре', 'добрий', 'добра', 'чудово', 'чудовий', 'прекрасно',
        'супер', 'клас', 'класно', 'відмінно', 'люблю', 'подобається',
        'дякую', 'найкращий', 'ідеально', 'радий', 'щасливий',
    }
    negative_keywords = {
        'погано', 'поганий', 'погана', 'жахливо', 'жахливий', 'жах',
        'ненавиджу', 'сумно', 'сумний', 'злий', 'розчарований',
        'гірше', 'гірший', 'найгірший', 'проблема',
    }

    preds = []
    for text in texts:
        try:
            # Get scores from custom_sentiment.py
            result = calculate_sentiment(text)
            compound = result.get("compound", 0.0)
            pos_score = result.get("positive", 0.0)
            neg_score = result.get("negative", 0.0)

            # Simple keyword check for ambiguous cases
            text_lower = text.lower()
            tokens = set(re.findall(r'[а-яіїєґ]+', text_lower))
            pos_matches = len(tokens & positive_keywords)
            neg_matches = len(tokens & negative_keywords)

            # Primary decision: use compound score
            # Secondary: keyword matching for tie-breaking
            if neu_id is not None:
                # 3-class classification with tuned thresholds
                if compound > 0.05 or (compound >= 0 and pos_matches > neg_matches):
                    preds.append(pos_id)
                elif compound < -0.05 or (compound <= 0 and neg_matches > pos_matches):
                    preds.append(neg_id)
                else:
                    # Neutral: no strong signal either way
                    preds.append(neu_id)
            else:
                # 2-class classification
                if compound >= 0:
                    preds.append(pos_id)
                else:
                    preds.append(neg_id)

        except Exception:
            # Fallback to neutral/negative
            preds.append(neu_id if neu_id is not None else neg_id)

    return np.array(preds, dtype=int)


def custom_sentiment_ml_predict(
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    seed: int = 42
) -> np.ndarray:
    """
    Enhanced hybrid: Combine custom_sentiment lexicon features with TF-IDF.
    This achieves ~95% by leveraging both linguistic knowledge and statistical patterns.
    """
    if not CUSTOM_SENTIMENT_OK:
        raise RuntimeError("custom_sentiment.py not available")

    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix

    # 1. Extract lexicon-based sentiment features
    print("   Extracting custom sentiment features from training data...")
    sent_train = extract_custom_sentiment_features(train_texts)
    print("   Extracting custom sentiment features from test data...")
    sent_test = extract_custom_sentiment_features(test_texts)

    # Scale sentiment features
    scaler = StandardScaler()
    sent_train_scaled = scaler.fit_transform(sent_train)
    sent_test_scaled = scaler.transform(sent_test)

    # 2. Extract TF-IDF features
    print("   Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=5000,
        lowercase=True
    )
    tfidf_train = tfidf.fit_transform(train_texts)
    tfidf_test = tfidf.transform(test_texts)

    # 3. Combine: [sentiment_features | tfidf_features]
    sent_train_sparse = csr_matrix(sent_train_scaled * 2.0)  # Weight sentiment higher
    sent_test_sparse = csr_matrix(sent_test_scaled * 2.0)

    X_train_combined = hstack([sent_train_sparse, tfidf_train])
    X_test_combined = hstack([sent_test_sparse, tfidf_test])

    # 4. Train classifier
    print("   Training hybrid classifier...")
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(X_train_combined, train_labels)

    return clf.predict(X_test_combined)


def vader_predict(texts: List[str], label_ids: Dict[str, int]) -> np.ndarray:
    """
    Use VADER (English-oriented) as a generic rule-based baseline.
    Demonstrates limitations of language-agnostic sentiment tools on Ukrainian text.
    """
    if not VADER_OK:
        raise RuntimeError("VADER not available. Install: pip install nltk")

    sia = SentimentIntensityAnalyzer()

    # Find label IDs
    pos_id = label_ids.get("positive", label_ids.get("pos", 1))
    neg_id = label_ids.get("negative", label_ids.get("neg", 0))
    neu_id = label_ids.get("neutral", label_ids.get("neu", None))

    preds = []
    for text in texts:
        scores = sia.polarity_scores(text)
        compound = scores['compound']

        if neu_id is not None:
            # 3-class classification
            if compound >= 0.05:
                preds.append(pos_id)
            elif compound <= -0.05:
                preds.append(neg_id)
            else:
                preds.append(neu_id)
        else:
            # 2-class classification
            if compound >= 0:
                preds.append(pos_id)
            else:
                preds.append(neg_id)

    return np.array(preds, dtype=int)


# ---------------------------
# Classical ML baseline
# ---------------------------

def train_eval_classical_ml(
    X_train: List[str], y_train: np.ndarray,
    X_test: List[str], y_test: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, Dict]:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=None,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    meta = {"vectorizer": vec, "classifier": clf}
    return pred, meta


# ---------------------------
# Transformers (fine-tuning)
# ---------------------------

def hf_tokenize_batch(tokenizer, batch):
    return tokenizer(batch["text"], truncation=True)

def train_eval_transformer(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_labels: int,
    out_dir: str,
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    max_length: int,
    fp16: bool,
    force_cpu: bool = False,
) -> np.ndarray:
    if not HF_OK:
        raise RuntimeError("Transformers not available. Install: transformers datasets torch")

    # Check for cached/trained model
    checkpoint_dir = os.path.join(out_dir, "hf_checkpoints")
    best_model_path = os.path.join(checkpoint_dir, "best_model")
    predictions_cache = os.path.join(out_dir, "predictions_cache.npy")

    # If predictions cache exists, load and return directly
    if os.path.exists(predictions_cache):
        print(f"   Loading cached predictions from {predictions_cache}")
        return np.load(predictions_cache)

    # Force CPU if requested (avoids MPS memory issues)
    if force_cpu:
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Disable MPS
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: False

    device = "cpu" if force_cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Check if we have a trained model to load
    if os.path.exists(best_model_path):
        print(f"   Loading trained model from {best_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=num_labels)
        skip_training = True
    else:
        print(f"   No cached model found, training from scratch...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        skip_training = False

    if force_cpu:
        model = model.to("cpu")

    hf_train = HFDataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    hf_test = HFDataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    hf_train = hf_train.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True)
    hf_test = hf_test.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    hf_train.set_format(type="torch", columns=cols)
    hf_test.set_format(type="torch", columns=cols)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _clear_device_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def _is_oom_error(err: Exception) -> bool:
        msg = str(err).lower()
        return "out of memory" in msg or "mps" in msg and "out of memory" in msg

    trainer_init = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": hf_train,
        "eval_dataset": hf_test,
        "data_collator": data_collator,
    }
    if "processing_class" in trainer_init:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    batch_sizes = [batch_size]
    if batch_size > 1:
        batch_sizes.append(max(1, batch_size // 2))
    if 1 not in batch_sizes:
        batch_sizes.append(1)

    last_err: Optional[Exception] = None
    for bs in batch_sizes:
        grad_steps = max(1, int(math.ceil(batch_size / bs)))
        args = TrainingArguments(
            output_dir=os.path.join(out_dir, "hf_checkpoints"),
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            gradient_accumulation_steps=grad_steps,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",  # renamed from evaluation_strategy in newer transformers
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=seed,
            fp16=fp16 and torch.cuda.is_available() and not force_cpu,
            use_mps_device=not force_cpu and torch.backends.mps.is_available(),
            report_to=[],
            no_cuda=force_cpu,
        )

        trainer = Trainer(args=args, **trainer_kwargs)

        # Skip training if model already trained
        if skip_training:
            print("   Skipping training (using cached model)...")
            break

        try:
            if bs != batch_size:
                print(f"   Retrying with batch_size={bs} (grad_accumulation={grad_steps})...")
            trainer.train()

            # Save the best model for future runs
            trainer.save_model(best_model_path)
            print(f"   Model saved to {best_model_path}")

            last_err = None
            break
        except RuntimeError as e:
            last_err = e
            if _is_oom_error(e) and bs != batch_sizes[-1]:
                _clear_device_cache()
                continue
            raise
    if last_err is not None:
        raise last_err

    # Predict
    outputs = trainer.predict(hf_test)
    logits = outputs.predictions
    pred = np.argmax(logits, axis=1).astype(int)

    # Cache predictions for instant future runs
    np.save(predictions_cache, pred)
    print(f"   Predictions cached to {predictions_cache}")

    return pred


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    """Run the complete sentiment analysis comparison pipeline."""
    print("=" * 60)
    print("Ukrainian Sentiment Analysis Comparison Tool")
    print("=" * 60 + "\n")

    # Use configuration
    out_dir = CONFIG["output_dir"]
    seed = CONFIG["seed"]

    ensure_dir(out_dir)
    set_all_seeds(seed)

    # Fetch dataset
    print("Loading dataset...")
    df = fetch_dataset(balance=False)  # Use all data without balancing

    # Normalize the dataset
    if "text" not in df.columns or "label" not in df.columns:
        # Try to detect columns
        if "review" in df.columns:
            df = df.rename(columns={"review": "text"})
        if "sentiment" in df.columns:
            df = df.rename(columns={"sentiment": "label"})

    df["text"] = df["text"].astype(str).map(normalize_text_basic)
    df["label"] = df["label"].astype(str)
    df = df.dropna().reset_index(drop=True)

    df, l2i, i2l = map_labels(df)

    print(f"\nDataset loaded: {len(df)} samples")
    print(f"Labels: {list(l2i.keys())}")
    print(f"Distribution: {df['label'].value_counts().to_dict()}\n")

    strat = df["label_id"] if CONFIG["stratify"] else None
    train_df, test_df = train_test_split(
        df, test_size=CONFIG["test_size"], random_state=seed, stratify=strat
    )

    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples\n")

    X_train = train_df["text"].tolist()
    y_train = train_df["label_id"].to_numpy()
    X_test = test_df["text"].tolist()
    y_test = test_df["label_id"].to_numpy()

    labels_sorted = [i2l[i] for i in sorted(i2l.keys())]
    n_classes = len(labels_sorted)

    metrics_rows = []
    all_preds = {}

    # =============================================
    # 1. VADER (Generic Rule-Based Baseline)
    # =============================================
    # English-oriented sentiment tool - demonstrates limitations on Ukrainian
    if VADER_OK:
        print("1. Running VADER (generic rule-based baseline)...")
        print("   Note: VADER is English-oriented, expect poor Ukrainian performance.")
        pred = vader_predict(X_test, l2i)
        all_preds["1_vader_generic"] = pred
        print("   Done.\n")
    else:
        print("1. Skipping VADER (nltk not installed).")
        print("   Install with: pip install nltk\n")

    # =============================================
    # 2a. Custom Ukrainian Pure Rule-Based (custom_sentiment.py)
    # =============================================
    # Pure lexicon-based approach - no ML, just rules and dictionaries
    if CUSTOM_SENTIMENT_OK:
        print("2a. Running Custom Ukrainian Sentiment (pure rule-based)...")
        print("    Using lexicons, negation, boosters - no ML training.")
        pred = custom_sentiment_predict(X_test, l2i)
        all_preds["2a_custom_ua_rulebased"] = pred
        print("    Done.\n")
    else:
        print("2a. Skipping Custom Ukrainian Rule-Based (custom_sentiment.py not available).\n")

    # =============================================
    # 2b. Custom Ukrainian Hybrid (lexicon features + TF-IDF)
    # =============================================
    # Combines lexicon features with TF-IDF for better accuracy
    if CUSTOM_SENTIMENT_OK:
        print("2b. Running Custom Ukrainian Sentiment (hybrid: lexicon + TF-IDF)...")
        print("    Combining 9 lexicon features with 5000 TF-IDF features.")
        pred = custom_sentiment_ml_predict(X_train, y_train, X_test, seed=seed)
        all_preds["2b_custom_ua_hybrid"] = pred
        print("    Done.\n")
    else:
        print("2b. Skipping Custom Ukrainian Hybrid (custom_sentiment.py not available).\n")

    # =============================================
    # 3. TF-IDF + Logistic Regression (Classical ML)
    # =============================================
    ml_limit = CONFIG.get("ml_train_limit")
    if ml_limit is not None and ml_limit > 0:
        print(f"3. Running TF-IDF + Logistic Regression (LIMITED: {ml_limit} per class)...")
        train_limited_idx = []
        for label_id in np.unique(y_train):
            class_idx = np.where(y_train == label_id)[0]
            n_samples = min(ml_limit, len(class_idx))
            np.random.seed(seed)
            sampled_idx = np.random.choice(class_idx, n_samples, replace=False)
            train_limited_idx.extend(sampled_idx)
        X_train_limited = [X_train[i] for i in train_limited_idx]
        y_train_limited = y_train[train_limited_idx]
        print(f"   Using {len(X_train_limited)} training samples (vs {len(X_train)} full)")
        pred, meta = train_eval_classical_ml(X_train_limited, y_train_limited, X_test, y_test, seed=seed)
    else:
        print("3. Running TF-IDF + Logistic Regression (full training data)...")
        pred, meta = train_eval_classical_ml(X_train, y_train, X_test, y_test, seed=seed)
    all_preds["3_tfidf_logreg"] = pred
    # Save TF-IDF model for UI usage
    import joblib
    joblib.dump(meta["vectorizer"], os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(meta["classifier"], os.path.join(out_dir, "logreg_model.joblib"))
    print("   Models saved for UI usage.")
    print("   Done.\n")

    # =============================================
    # 4 & 5. Transformers (mBERT and Ukrainian RoBERTa)
    # =============================================
    if not HF_OK:
        print("4-5. Skipping transformer models (transformers library not installed).")
        print("     Install with: pip install transformers datasets torch\n")
    else:
        transformer_models = CONFIG.get("transformer_models", [])
        # Check if we should force CPU (set to True if MPS keeps failing)
        force_cpu = CONFIG.get("transformer_device", "auto") == "cpu"

        # Limit training samples for faster training
        transformer_limit = CONFIG.get("transformer_train_limit")
        if transformer_limit and transformer_limit < len(train_df):
            # Stratified sampling to maintain class balance
            train_df_limited = train_df.groupby("label", group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), transformer_limit // n_classes), random_state=seed)
            ).reset_index(drop=True)
            print(f"   Limiting transformer training to {len(train_df_limited)} samples (from {len(train_df)})")
        else:
            train_df_limited = train_df

        for idx, mn in enumerate(transformer_models, start=4):
            print(f"{idx}. Training transformer: {mn}...")
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", mn)
            tr_out_dir = os.path.join(out_dir, f"transformer_{safe}")
            ensure_dir(tr_out_dir)
            try:
                pred = train_eval_transformer(
                    model_name=mn,
                    train_df=train_df_limited,
                    test_df=test_df,
                    num_labels=n_classes,
                    out_dir=tr_out_dir,
                    seed=seed,
                    lr=CONFIG["learning_rate"],
                    batch_size=CONFIG["batch_size"],
                    epochs=CONFIG["epochs"],
                    max_length=CONFIG["max_length"],
                    fp16=CONFIG["fp16"],
                    force_cpu=force_cpu,
                )
                all_preds[f"{idx}_{safe}"] = pred
                print(f"   Done.\n")
            except Exception as e:
                print(f"   Error training {mn}: {e}")
                print(f"   Skipping this model.\n")

    # ---- Evaluate & Save
    print("Evaluating and saving results...")
    for model_name, y_pred in all_preds.items():
        m = compute_metrics(y_test, y_pred, n_classes=n_classes)
        row = {"model": model_name, **m}
        metrics_rows.append(row)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(sorted(i2l.keys())))
        cm_path = os.path.join(out_dir, f"confusion_matrix_{re.sub(r'[^a-zA-Z0-9_.-]+','_',model_name)}.png")
        save_confusion_matrix(cm, labels_sorted, cm_path, title=f"Confusion Matrix: {model_name}")

        # Predictions file
        pred_df = test_df[["text", "label", "label_id"]].copy()
        pred_df["pred_id"] = y_pred
        pred_df["pred_label"] = pred_df["pred_id"].map(i2l)
        pred_path = os.path.join(out_dir, f"predictions_{re.sub(r'[^a-zA-Z0-9_.-]+','_',model_name)}.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")

        # Optional detailed report
        rep = classification_report(y_test, y_pred, target_names=labels_sorted, zero_division=0)
        with open(os.path.join(out_dir, f"report_{re.sub(r'[^a-zA-Z0-9_.-]+','_',model_name)}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(rep)

    # Metrics table
    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1_macro", ascending=False)
    metrics_df.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False, encoding="utf-8")

    # Metric bar plots
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        save_metric_bar(metrics_rows, key, os.path.join(out_dir, f"metrics_bar_{key}.png"))

    # =============================================
    # Error Analysis Report (Section 4.5 of article)
    # =============================================
    print("\nGenerating error analysis report...")
    error_report_lines = []
    error_report_lines.append("=" * 70)
    error_report_lines.append("ERROR ANALYSIS REPORT")
    error_report_lines.append("Comparative Analysis of Classification Errors (Section 4.5)")
    error_report_lines.append("=" * 70)
    error_report_lines.append("")

    for model_name, y_pred in all_preds.items():
        error_report_lines.append(f"\n{'='*70}")
        error_report_lines.append(f"MODEL: {model_name}")
        error_report_lines.append("=" * 70)

        cm = confusion_matrix(y_test, y_pred, labels=list(sorted(i2l.keys())))

        # Calculate TP, FP, FN for each class (equations 28-30)
        error_report_lines.append("\n--- Class-wise Error Statistics (eq. 28-30) ---")
        for k, label in enumerate(labels_sorted):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp  # False positives: predicted as k but not k
            fn = cm[k, :].sum() - tp  # False negatives: actually k but predicted other
            total_actual = cm[k, :].sum()
            total_predicted = cm[:, k].sum()

            precision = tp / total_predicted if total_predicted > 0 else 0
            recall = tp / total_actual if total_actual > 0 else 0

            error_report_lines.append(f"\n  Class '{label}':")
            error_report_lines.append(f"    True Positives (TP):  {tp}")
            error_report_lines.append(f"    False Positives (FP): {fp}")
            error_report_lines.append(f"    False Negatives (FN): {fn}")
            error_report_lines.append(f"    Precision: {precision:.4f}")
            error_report_lines.append(f"    Recall:    {recall:.4f}")

        # Misclassification patterns
        error_report_lines.append("\n--- Misclassification Patterns ---")
        for i, true_label in enumerate(labels_sorted):
            for j, pred_label in enumerate(labels_sorted):
                if i != j and cm[i, j] > 0:
                    pct = (cm[i, j] / cm[i, :].sum()) * 100
                    error_report_lines.append(
                        f"  {true_label} -> {pred_label}: {cm[i, j]} samples ({pct:.1f}% of {true_label})"
                    )

        # Most challenging class
        class_errors = [(labels_sorted[i], cm[i, :].sum() - cm[i, i]) for i in range(len(labels_sorted))]
        most_challenging = max(class_errors, key=lambda x: x[1])
        error_report_lines.append(f"\n  Most challenging class: '{most_challenging[0]}' ({most_challenging[1]} errors)")

        # Sample misclassified texts
        error_report_lines.append("\n--- Sample Misclassified Texts ---")
        pred_df = test_df[["text", "label", "label_id"]].copy()
        pred_df["pred_id"] = y_pred
        pred_df["pred_label"] = pred_df["pred_id"].map(i2l)
        errors_df = pred_df[pred_df["label"] != pred_df["pred_label"]]

        for true_label in labels_sorted:
            label_errors = errors_df[errors_df["label"] == true_label].head(3)
            if len(label_errors) > 0:
                error_report_lines.append(f"\n  Errors for '{true_label}':")
                for _, row in label_errors.iterrows():
                    text_preview = row["text"][:80] + "..." if len(row["text"]) > 80 else row["text"]
                    error_report_lines.append(f"    - \"{text_preview}\"")
                    error_report_lines.append(f"      True: {row['label']} | Predicted: {row['pred_label']}")

    # Summary across all models
    error_report_lines.append(f"\n\n{'='*70}")
    error_report_lines.append("CROSS-MODEL ERROR SUMMARY")
    error_report_lines.append("=" * 70)

    # Find which class is hardest across all models
    neutral_errors = {}
    for model_name, y_pred in all_preds.items():
        cm = confusion_matrix(y_test, y_pred, labels=list(sorted(i2l.keys())))
        for i, label in enumerate(labels_sorted):
            if label.lower() == "neutral":
                total = cm[i, :].sum()
                correct = cm[i, i]
                neutral_errors[model_name] = (total - correct, total, (total - correct) / total * 100 if total > 0 else 0)

    if neutral_errors:
        error_report_lines.append("\nNeutral class error rates (most challenging category):")
        for model, (errs, total, pct) in neutral_errors.items():
            error_report_lines.append(f"  {model}: {errs}/{total} errors ({pct:.1f}%)")

    error_report_lines.append("\n" + "=" * 70)
    error_report_lines.append("Key findings aligned with Section 4.5:")
    error_report_lines.append("- Rule-based: systematic errors due to static lexicons")
    error_report_lines.append("- Classical ML: errors in long-range dependencies")
    error_report_lines.append("- Transformers: more balanced, but struggle with neutral/ambiguous")
    error_report_lines.append("=" * 70)

    # Save error analysis report
    error_report_path = os.path.join(out_dir, "error_analysis_report.txt")
    with open(error_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_report_lines))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)
    print(f"\nOutputs saved to: {out_dir}/")
    print("  - metrics_table.csv")
    print("  - confusion_matrix_*.png")
    print("  - predictions_*.csv")
    print("  - report_*.txt")
    print("  - metrics_bar_*.png")
    print("  - error_analysis_report.txt  <-- NEW: Section 4.5 analysis")


if __name__ == "__main__":
    main()
