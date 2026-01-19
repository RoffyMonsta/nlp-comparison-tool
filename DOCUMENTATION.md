# Ukrainian Sentiment Analysis Comparison Framework - Documentation

## Overview

This framework provides a comparative evaluation of 6 sentiment analysis approaches for Ukrainian text, ranging from simple rule-based methods to state-of-the-art transformer models.

---

## Models Compared

| # | Model Name | Type | Description | Accuracy |
|---|------------|------|-------------|----------|
| 1 | `1_vader_generic` | Rule-based | VADER (English) - baseline showing limitations of language-agnostic tools | 7.1% |
| 2a | `2a_custom_ua_rulebased` | Rule-based | Custom Ukrainian lexicon-based with spaCy, negation handling, boosters | 83.1% |
| 2b | `2b_custom_ua_hybrid` | Hybrid ML | Custom lexicon features (9) + TF-IDF (5000 features) combined | 97.2% |
| 3 | `3_tfidf_logreg` | Classical ML | TF-IDF vectorization + Logistic Regression | 98.0% |
| 4 | `4_xlm-roberta-base` | Transformer | XLM-RoBERTa fine-tuned on Ukrainian sentiment | 97.5% |
| 5 | `5_bert-base-multilingual-cased` | Transformer | mBERT fine-tuned on Ukrainian sentiment | 96.7% |

---

## Configuration Parameters (`CONFIG`)

```python
CONFIG = {
    "output_dir": "outputs",           # Directory for all output files
    "test_size": 0.2,                  # 80/20 train-test split
    "seed": 42,                        # Random seed for reproducibility
    "stratify": True,                  # Maintain class distribution in splits

    # ML Training Limits
    "ml_train_limit": None,            # None = use all data, or set number (e.g., 500)
    "transformer_train_limit": 2000,   # Limit transformer training samples for speed

    # Transformer Models
    "transformer_models": [
        "xlm-roberta-base",
        "bert-base-multilingual-cased"
    ],

    # Transformer Training Hyperparameters
    "learning_rate": 2e-5,
    "batch_size": 2,
    "epochs": 3,
    "max_length": 128,                 # Max token sequence length
    "fp16": False,                     # Mixed precision (requires CUDA)
    "gradient_checkpointing": True,
    "transformer_device": "cpu"        # "cpu" | "mps" | "auto"
}
```

---

## Dataset

- **Source**: Ukrainian Twitter sentiment data
- **Classes**: 3 (positive, negative, neutral)
- **Files**:
  - `./dict/positive.json`
  - `./dict/negative.json`
  - `./dict/neutral.json`
- **Total samples**: ~9,000
- **Test set**: 1,833 samples (20%)

### Class Distribution in Test Set:
| Class | Samples |
|-------|---------|
| Positive | 1,314 |
| Negative | 419 |
| Neutral | 100 |

---

## Output Files Generated

### 1. Metrics Table (`metrics_table.csv`)
Main results table with all evaluation metrics per model.

**Columns**:
- `model` - Model identifier
- `accuracy` - Overall accuracy
- `precision_macro` - Macro-averaged precision
- `recall_macro` - Macro-averaged recall
- `f1_macro` - Macro-averaged F1 score
- `precision_micro`, `recall_micro`, `f1_micro` - Micro-averaged metrics

### 2. Confusion Matrices (`confusion_matrix_*.png`)
One grayscale confusion matrix per model showing:
- True labels (rows) vs Predicted labels (columns)
- Cell values showing sample counts
- Grid lines for readability
- Black/white friendly for printing

**Files**:
- `confusion_matrix_1_vader_generic.png`
- `confusion_matrix_2a_custom_ua_rulebased.png`
- `confusion_matrix_2b_custom_ua_hybrid.png`
- `confusion_matrix_3_tfidf_logreg.png`
- `confusion_matrix_4_xlm-roberta-base.png`
- `confusion_matrix_5_bert-base-multilingual-cased.png`

### 3. Metric Bar Charts (`metrics_bar_*.png`)
Comparative bar charts with hatching patterns (black/white friendly):
- `metrics_bar_accuracy.png`
- `metrics_bar_precision_macro.png`
- `metrics_bar_recall_macro.png`
- `metrics_bar_f1_macro.png`

**Features**:
- Different hatch patterns per model (solid, ///, \\\, xxx, ..., +++)
- Value labels on top of each bar
- Horizontal grid lines

### 4. Classification Reports (`report_*.txt`)
Detailed sklearn classification reports per model:
```
              precision    recall  f1-score   support

    negative       0.93      0.99      0.96       419
     neutral       1.00      1.00      1.00       100
    positive       1.00      0.98      0.99      1314

    accuracy                           0.98      1833
```

### 5. Predictions (`predictions_*.csv`)
Raw predictions for error analysis:
- `text` - Original Ukrainian text
- `label` - True sentiment label
- `pred_label` - Predicted sentiment label

### 6. Error Analysis Report (`error_analysis_report.txt`)
Comprehensive error analysis aligned with Section 4.5 methodology:

**Per-model analysis includes**:
- Class-wise TP, FP, FN statistics (equations 28-30)
- Precision and Recall per class
- Misclassification patterns (e.g., "negative -> positive: 6 samples (1.4%)")
- Most challenging class identification
- Sample misclassified texts with true vs predicted labels

**Cross-model summary**:
- Neutral class error rates across all models
- Key findings summary

---

## Evaluation Metrics

### Formulas Used

**Accuracy**:
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision (per class)**:
$$Precision_c = \frac{TP_c}{TP_c + FP_c}$$

**Recall (per class)**:
$$Recall_c = \frac{TP_c}{TP_c + FN_c}$$

**F1 Score (per class)**:
$$F1_c = 2 \times \frac{Precision_c \times Recall_c}{Precision_c + Recall_c}$$

**Macro-averaged** (equal weight per class):
$$F1_{macro} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

---

## Custom Sentiment Analyzer Features

The hybrid model (`2b_custom_ua_hybrid`) extracts 9 lexicon-based features:

| Feature | Description |
|---------|-------------|
| `compound` | Overall sentiment score (-1 to +1) |
| `positive` | Positive sentiment intensity |
| `negative` | Negative sentiment intensity |
| `neutral` | Neutral token ratio |
| `confidence` | Entropy-based classification confidence |
| `num_tokens` | Token count |
| `has_negation` | Binary: negation detected (0/1) |
| `num_boosters` | Count of intensity boosters |
| `emoji_score` | Sentiment contribution from emojis |

These 9 features are scaled and combined with 5000 TF-IDF features (unigrams + bigrams) using horizontal stacking, then fed to Logistic Regression.

---

## Results Summary Table

| Model | Accuracy | Precision | Recall | F1 (Macro) |
|-------|----------|-----------|--------|------------|
| TF-IDF + LogReg | **98.0%** | 97.5% | **98.8%** | **98.1%** |
| XLM-RoBERTa | 97.5% | 96.6% | 98.7% | 97.6% |
| Custom Hybrid | 97.2% | 95.0% | 98.2% | 96.5% |
| mBERT | 96.7% | 95.6% | 98.2% | 96.8% |
| Custom Rule-based | 83.1% | 76.2% | 86.0% | 80.2% |
| VADER (English) | 7.1% | 42.5% | 34.3% | 5.4% |

---

## Key Findings

1. **TF-IDF + LogReg achieves best performance** (98.0%) - classical ML remains competitive for Ukrainian sentiment

2. **Custom hybrid approach reaches 97.2%** - combining linguistic features with statistical patterns

3. **Pure rule-based achieves 83.1%** - demonstrating the value of Ukrainian-specific lexicons

4. **VADER fails completely (7.1%)** - English-oriented tools unsuitable for Ukrainian

5. **Transformers perform well but don't exceed classical ML** on this dataset size

6. **Neutral class perfectly classified by all models** (100 samples, 0% error rate)

7. **Most challenging: positive class** - models struggle with sarcasm and mixed sentiment

---

## Running the Framework

```bash
# Activate virtual environment
source venv312/bin/activate

# Run full comparison
python main.py

# Launch Streamlit UI
streamlit run app.py
```

---

## Streamlit UI Features (`app.py`)

- **Results Dashboard**: Metrics table, F1 bar chart, confusion matrices, error analysis charts
- **Test Sentiment**: Interactive testing with all models simultaneously
- **Model Details**: Per-model confusion matrix, classification report, sample predictions
- **Run Models**: Configuration display and cache management

---

## File Structure

```
nlp-comparison/
├── main.py                    # Main comparison script
├── app.py                     # Streamlit UI
├── custom_sentiment.py        # Ukrainian lexicon-based analyzer
├── dict/
│   ├── positive.json          # Positive sentiment samples
│   ├── negative.json          # Negative sentiment samples
│   ├── neutral.json           # Neutral sentiment samples
│   ├── emolex.txt             # NRC Emotion Lexicon (Ukrainian)
│   ├── polarity_score.csv     # Word polarity scores
│   ├── stopwords_ua.txt       # Ukrainian stopwords
│   ├── intensity_booster_words.txt
│   └── large_phrase_sentiment_2000.json
└── outputs/
    ├── metrics_table.csv
    ├── confusion_matrix_*.png
    ├── metrics_bar_*.png
    ├── predictions_*.csv
    ├── report_*.txt
    └── error_analysis_report.txt
```
