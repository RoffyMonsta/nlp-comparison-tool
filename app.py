#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Ukrainian Sentiment Analysis Comparison

Run with: streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Ukrainian Sentiment Analysis Comparison",
    page_icon="🇺🇦",
    layout="wide"
)

# Paths
OUTPUTS_DIR = Path("outputs")
METRICS_FILE = OUTPUTS_DIR / "metrics_table.csv"

# ================================
# Helper functions
# ================================

@st.cache_data
def load_metrics():
    """Load metrics from CSV if exists."""
    if METRICS_FILE.exists():
        df = pd.read_csv(METRICS_FILE)
        return df
    return None

@st.cache_data
def load_predictions(model_name: str):
    """Load predictions for a specific model."""
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    pred_file = OUTPUTS_DIR / f"predictions_{safe_name}.csv"
    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None

def get_confusion_matrix_path(model_name: str):
    """Get path to confusion matrix image."""
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    cm_file = OUTPUTS_DIR / f"confusion_matrix_{safe_name}.png"
    if cm_file.exists():
        return cm_file
    return None

def get_available_models():
    """Get list of models with results."""
    if not METRICS_FILE.exists():
        return []
    df = pd.read_csv(METRICS_FILE)
    return df['model'].tolist()

# ================================
# Main UI
# ================================

st.title("🇺🇦 Ukrainian Sentiment Analysis Comparison")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page:",
        ["📊 Results Dashboard", "🔍 Test Sentiment", "📈 Model Details", "⚙️ Run Models"]
    )

    st.markdown("---")
    st.markdown("### Quick Info")
    metrics_df = load_metrics()
    if metrics_df is not None:
        st.success(f"✅ {len(metrics_df)} models evaluated")
    else:
        st.warning("⚠️ No results yet. Run main.py first.")

# ================================
# Page: Results Dashboard
# ================================
if page == "📊 Results Dashboard":
    st.header("Results Dashboard")

    metrics_df = load_metrics()

    if metrics_df is None:
        st.error("No results found. Please run `python main.py` first to generate results.")
        st.code("python main.py", language="bash")
    else:
        # Main metrics table
        st.subheader("Model Comparison")

        # Format for display
        display_df = metrics_df.copy()
        display_df = display_df.sort_values('f1_macro', ascending=False)

        # Rename columns for clarity
        col_rename = {
            'model': 'Model',
            'accuracy': 'Accuracy',
            'precision_macro': 'Precision',
            'recall_macro': 'Recall',
            'f1_macro': 'F1 Score'
        }
        display_cols = ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        display_df = display_df[display_cols].rename(columns=col_rename)

        # Style the dataframe
        def highlight_best(s):
            is_max = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_max]

        styled_df = display_df.style.apply(
            highlight_best,
            subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']
        ).format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1 Score': '{:.2%}'
        })

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Bar chart
        st.subheader("F1 Score Comparison")
        chart_df = metrics_df[['model', 'f1_macro']].sort_values('f1_macro', ascending=True)
        st.bar_chart(chart_df.set_index('model'))

        # Confusion matrices
        st.subheader("Confusion Matrices")
        models = get_available_models()

        cols = st.columns(min(3, len(models)))
        for i, model in enumerate(models):
            cm_path = get_confusion_matrix_path(model)
            if cm_path:
                with cols[i % 3]:
                    st.image(str(cm_path), caption=model, use_container_width=True)

        # Error Analysis Charts
        st.subheader("📊 Error Analysis")

        error_report_path = OUTPUTS_DIR / "error_analysis_report.txt"
        if error_report_path.exists():
            # Parse error data from predictions
            error_data = []
            for model in models:
                pred_df = load_predictions(model)
                if pred_df is not None and 'label' in pred_df.columns and 'pred_label' in pred_df.columns:
                    total = len(pred_df)
                    errors = len(pred_df[pred_df['label'] != pred_df['pred_label']])
                    error_rate = errors / total if total > 0 else 0

                    # Per-class errors
                    neg_mask = pred_df['label'] == 'negative'
                    pos_mask = pred_df['label'] == 'positive'
                    neu_mask = pred_df['label'] == 'neutral'

                    neg_errors = len(pred_df[neg_mask & (pred_df['label'] != pred_df['pred_label'])])
                    pos_errors = len(pred_df[pos_mask & (pred_df['label'] != pred_df['pred_label'])])
                    neu_errors = len(pred_df[neu_mask & (pred_df['label'] != pred_df['pred_label'])])

                    neg_total = neg_mask.sum()
                    pos_total = pos_mask.sum()
                    neu_total = neu_mask.sum()

                    error_data.append({
                        'Model': model,
                        'Total Error Rate': error_rate,
                        'Negative Errors': neg_errors / neg_total if neg_total > 0 else 0,
                        'Positive Errors': pos_errors / pos_total if pos_total > 0 else 0,
                        'Neutral Errors': neu_errors / neu_total if neu_total > 0 else 0
                    })

            if error_data:
                error_df = pd.DataFrame(error_data)

                # Total error rate bar chart
                st.markdown("**Error Rate by Model**")
                chart_df = error_df[['Model', 'Total Error Rate']].set_index('Model').sort_values('Total Error Rate')
                chart_df['Total Error Rate'] = chart_df['Total Error Rate'] * 100  # Convert to percentage
                st.bar_chart(chart_df, horizontal=True)

                # Per-class error rates
                st.markdown("**Error Rate by Class**")
                class_error_df = error_df[['Model', 'Negative Errors', 'Positive Errors', 'Neutral Errors']].set_index('Model')
                class_error_df = class_error_df * 100  # Convert to percentage
                st.bar_chart(class_error_df)

                # Show raw numbers
                with st.expander("View Error Statistics Table"):
                    display_error_df = error_df.copy()
                    display_error_df['Total Error Rate'] = display_error_df['Total Error Rate'].apply(lambda x: f"{x:.1%}")
                    display_error_df['Negative Errors'] = display_error_df['Negative Errors'].apply(lambda x: f"{x:.1%}")
                    display_error_df['Positive Errors'] = display_error_df['Positive Errors'].apply(lambda x: f"{x:.1%}")
                    display_error_df['Neutral Errors'] = display_error_df['Neutral Errors'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(display_error_df, use_container_width=True, hide_index=True)
        else:
            st.info("Error analysis report not found. Run main.py to generate it.")

# ================================
# Page: Test Sentiment
# ================================
elif page == "🔍 Test Sentiment":
    st.header("Test Sentiment Analysis")

    # Text input
    text_input = st.text_area(
        "Enter Ukrainian text to analyze:",
        value="Це чудовий день! Я дуже щасливий.",
        height=100
    )

    st.markdown("---")

    # Main button to analyze with ALL models
    if st.button("🚀 Analyze with ALL Models", use_container_width=True, type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            results = {}

            # 1. VADER
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(text_input)
                compound = scores['compound']
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                results["VADER (English)"] = {
                    "label": label,
                    "compound": compound,
                    "details": f"pos={scores['pos']:.2f}, neg={scores['neg']:.2f}, neu={scores['neu']:.2f}"
                }
            except Exception as e:
                results["VADER (English)"] = {"label": "error", "compound": 0, "details": str(e)}

            # 2. Custom Ukrainian Sentiment
            try:
                from custom_sentiment import calculate_sentiment
                result = calculate_sentiment(text_input)
                compound = result.get('compound', 0)
                if compound > 0.05:
                    label = "positive"
                elif compound < -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                results["Custom UA (Lexicon)"] = {
                    "label": label,
                    "compound": compound,
                    "details": f"pos={result.get('positive', 0):.2f}, neg={result.get('negative', 0):.2f}, conf={result.get('confidence', 0):.2f}"
                }
            except Exception as e:
                results["Custom UA (Lexicon)"] = {"label": "error", "compound": 0, "details": str(e)}

            # 3. TF-IDF + LogReg (if model exists)
            try:
                import joblib
                tfidf_path = OUTPUTS_DIR / "tfidf_vectorizer.joblib"
                logreg_path = OUTPUTS_DIR / "logreg_model.joblib"
                if tfidf_path.exists() and logreg_path.exists():
                    vectorizer = joblib.load(tfidf_path)
                    clf = joblib.load(logreg_path)
                    X = vectorizer.transform([text_input])
                    pred = clf.predict(X)[0]
                    proba = clf.predict_proba(X)[0]
                    label_map = {0: "negative", 1: "neutral", 2: "positive"}
                    label = label_map.get(pred, str(pred))
                    results["TF-IDF + LogReg"] = {
                        "label": label,
                        "compound": proba[pred] if pred < len(proba) else 0,
                        "details": f"confidence={max(proba):.2f}"
                    }
                else:
                    results["TF-IDF + LogReg"] = {"label": "not trained", "compound": 0, "details": "Run main.py first"}
            except Exception as e:
                results["TF-IDF + LogReg"] = {"label": "error", "compound": 0, "details": str(e)}

            # 4. Transformers (if models exist)
            transformer_models = [
                ("XLM-RoBERTa", "transformer_xlm-roberta-base"),
                ("mBERT", "transformer_bert-base-multilingual-cased"),
            ]

            for display_name, folder_name in transformer_models:
                try:
                    model_path = OUTPUTS_DIR / folder_name / "hf_checkpoints" / "best_model"
                    if model_path.exists():
                        from transformers import AutoTokenizer, AutoModelForSequenceClassification
                        import torch

                        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                        model.eval()

                        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=128)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            probs = torch.softmax(outputs.logits, dim=1)[0]
                            pred = torch.argmax(probs).item()

                        label_map = {0: "negative", 1: "neutral", 2: "positive"}
                        label = label_map.get(pred, str(pred))
                        results[display_name] = {
                            "label": label,
                            "compound": probs[pred].item(),
                            "details": f"neg={probs[0]:.2f}, neu={probs[1]:.2f}, pos={probs[2]:.2f}"
                        }
                    else:
                        results[display_name] = {"label": "not trained", "compound": 0, "details": "Run main.py first"}
                except Exception as e:
                    results[display_name] = {"label": "error", "compound": 0, "details": str(e)[:50]}

            # Display results in a nice table
            st.subheader("Results Comparison")

            # Create columns for each result
            cols = st.columns(len(results))
            for i, (model_name, data) in enumerate(results.items()):
                with cols[i]:
                    st.markdown(f"**{model_name}**")
                    label = data["label"]
                    if label == "positive":
                        st.success(f"POSITIVE")
                    elif label == "negative":
                        st.error(f"NEGATIVE")
                    elif label == "neutral":
                        st.info(f"NEUTRAL")
                    else:
                        st.warning(f"{label.upper()}")
                    st.caption(data["details"])

            # Summary
            st.markdown("---")
            labels = [r["label"] for r in results.values() if r["label"] in ["positive", "negative", "neutral"]]
            if labels:
                from collections import Counter
                vote_counts = Counter(labels)
                majority = vote_counts.most_common(1)[0]
                st.markdown(f"**Consensus (majority vote):** {majority[0].upper()} ({majority[1]}/{len(labels)} models)")

    st.markdown("---")
    st.subheader("Individual Model Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Custom UA Details", use_container_width=True):
            try:
                from custom_sentiment import calculate_sentiment
                result = calculate_sentiment(text_input)

                st.success("Custom Ukrainian Analysis")

                # Display scores
                score_cols = st.columns(4)
                with score_cols[0]:
                    st.metric("Compound", f"{result.get('compound', 0):.3f}")
                with score_cols[1]:
                    st.metric("Positive", f"{result.get('positive', 0):.3f}")
                with score_cols[2]:
                    st.metric("Negative", f"{result.get('negative', 0):.3f}")
                with score_cols[3]:
                    st.metric("Neutral", f"{result.get('neutral', 0):.3f}")

                # Additional details
                st.markdown("**Additional Features:**")
                detail_cols = st.columns(4)
                with detail_cols[0]:
                    st.metric("Confidence", f"{result.get('confidence', 0):.3f}")
                with detail_cols[1]:
                    st.metric("Tokens", result.get('num_tokens', 0))
                with detail_cols[2]:
                    st.metric("Negation", "Yes" if result.get('has_negation', 0) else "No")
                with detail_cols[3]:
                    st.metric("Boosters", result.get('num_boosters', 0))

                # Sentiment label
                compound = result.get('compound', 0)
                if compound > 0.1:
                    st.success("**Sentiment: POSITIVE**")
                elif compound < -0.1:
                    st.error("**Sentiment: NEGATIVE**")
                else:
                    st.info("**Sentiment: NEUTRAL**")

            except ImportError:
                st.error("custom_sentiment.py not found or has errors.")

    with col2:
        if st.button("🌐 VADER Details", use_container_width=True):
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(text_input)

                st.warning("VADER Analysis (Note: Designed for English)")

                score_cols = st.columns(4)
                with score_cols[0]:
                    st.metric("Compound", f"{scores['compound']:.3f}")
                with score_cols[1]:
                    st.metric("Positive", f"{scores['pos']:.3f}")
                with score_cols[2]:
                    st.metric("Negative", f"{scores['neg']:.3f}")
                with score_cols[3]:
                    st.metric("Neutral", f"{scores['neu']:.3f}")

            except ImportError:
                st.error("NLTK/VADER not installed. Run: pip install nltk")

# ================================
# Page: Model Details
# ================================
elif page == "📈 Model Details":
    st.header("Model Details")

    models = get_available_models()

    if not models:
        st.warning("No model results available yet.")
    else:
        selected_model = st.selectbox("Select a model:", models)

        if selected_model:
            # Load predictions
            pred_df = load_predictions(selected_model)

            col1, col2 = st.columns(2)

            with col1:
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm_path = get_confusion_matrix_path(selected_model)
                if cm_path:
                    st.image(str(cm_path), use_container_width=True)
                else:
                    st.info("Confusion matrix not found.")

            with col2:
                # Classification report
                st.subheader("Classification Report")
                report_path = OUTPUTS_DIR / f"report_{selected_model.replace('/', '_')}.txt"
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        st.code(f.read())
                else:
                    st.info("Report not found.")

            # Sample predictions
            if pred_df is not None:
                st.subheader("Sample Predictions")

                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_errors = st.checkbox("Show only errors", value=False)
                with filter_col2:
                    n_samples = st.slider("Number of samples", 5, 50, 10)

                if show_errors:
                    display_pred = pred_df[pred_df['label'] != pred_df['pred_label']]
                else:
                    display_pred = pred_df

                st.dataframe(
                    display_pred[['text', 'label', 'pred_label']].head(n_samples),
                    use_container_width=True,
                    hide_index=True
                )

# ================================
# Page: Run Models
# ================================
elif page == "⚙️ Run Models":
    st.header("Run Models")

    st.markdown("""
    ### Instructions

    The main comparison script trains transformer models which takes significant time.
    Run it from the terminal for best experience:

    ```bash
    source venv312/bin/activate
    python main.py
    ```

    ### Current Configuration
    """)

    # Show current config
    try:
        from main import CONFIG
        st.json(CONFIG)
    except:
        st.info("Could not load config from main.py")

    st.markdown("---")

    # Quick run buttons for fast models
    st.subheader("Quick Analysis (Fast Models Only)")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Run VADER", use_container_width=True):
            st.info("VADER runs instantly during 'Test Sentiment' - no training needed!")

    with col2:
        if st.button("🔄 Run Custom Rule-Based", use_container_width=True):
            st.info("Custom rule-based runs instantly during 'Test Sentiment' - no training needed!")

    with col3:
        if st.button("🔄 Refresh Results", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared! Results will reload.")
            st.rerun()

    st.markdown("---")
    st.markdown("""
    ### Model Training Status

    Check terminal output for training progress. Transformers take ~1-2 hours total.

    After training completes, click **Refresh Results** to see updated metrics.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Ukrainian Sentiment Analysis Comparison Tool | "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
