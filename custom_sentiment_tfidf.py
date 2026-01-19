from typing import List, Dict, Tuple
import pandas as pd
import json
import re
from itertools import chain
import spacy

# === 1. Load lexicons ===
EMOTION_LEXICON_PATH = './dict/emolex.txt'
emotion_lexicon = pd.read_csv(EMOTION_LEXICON_PATH, sep='\t')
ukrainian_lexicon = {
    row['Ukrainian Word']: {
        'anger': row['anger'],
        'anticipation': row['anticipation'],
        'disgust': row['disgust'],
        'fear': row['fear'],
        'joy': row['joy'],
        'negative': row['negative'],
        'positive': row['positive'],
        'sadness': row['sadness'],
        'surprise': row['surprise'],
        'trust': row['trust']
    }
    for _, row in emotion_lexicon.iterrows()
}

# Booster words
BOOSTER_PATH = './dict/intensity_booster_words.txt'
with open(BOOSTER_PATH, 'r', encoding='utf-8') as f:
    booster_words = json.load(f)

# Phrase sentiment
PHRASE_SENTIMENT_PATH = './dict/large_phrase_sentiment_2000.json'
with open(PHRASE_SENTIMENT_PATH, 'r', encoding='utf-8') as f:
    phrase_sentiment = json.load(f)

# Polarity scores
POLARITY_SCORE_PATH = './dict/polarity_score.csv'
polarity_df = pd.read_csv(POLARITY_SCORE_PATH, sep=';')
polarity_score_dict = {row['word']: row['pos_neg'] for _, row in polarity_df.iterrows()}

# Stopwords
STOPWORDS_PATH = './dict/stopwords_ua.txt'
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

# Emojis with rebalanced scores
emoji_sentiment = {
    ":)": 1.5,
    ":(": -1.5,
    ":D": 2.0,
    ";)": 1.2,
    "😂": 1.8,
    "😢": -1.8,
    "❤️": 2.0,
    "🔥": 1.5,
    "😡": -1.8
}

# spaCy Ukrainian model
nlp = spacy.load("uk_core_news_sm")

# === 2. Tokenization ===
def tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b|[!?.,:;()]|[:;]-?[()D]|[\u263a-\U0001f645]', text)
    return [token for token in tokens if token not in stopwords]

# N-grams
def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Dependencies
def analyze_dependencies(text: str) -> List[Tuple[str, str, str]]:
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

# === 3. Sentiment scoring ===
def analyze_tokens(tokens: List[str]) -> List[Tuple[str, float]]:
    sentiment_scores = []
    ngrams = list(chain(tokens, generate_ngrams(tokens, 2), generate_ngrams(tokens, 3)))

    for i, token in enumerate(ngrams):
        lexicon_entry = ukrainian_lexicon.get(token, {})
        base_score = (
            1.0 * lexicon_entry.get('positive', 0)
            - 1.0 * lexicon_entry.get('negative', 0)
            + 1.0 * lexicon_entry.get('joy', 0)
            + 0.8 * lexicon_entry.get('trust', 0)
            + 0.5 * lexicon_entry.get('surprise', 0)
            - 0.8 * lexicon_entry.get('sadness', 0)
            - 0.8 * lexicon_entry.get('fear', 0)
            - 0.8 * lexicon_entry.get('disgust', 0)
            - 0.8 * lexicon_entry.get('anger', 0)
        )

        if token in phrase_sentiment:
            base_score += 1.5 * phrase_sentiment[token]
        elif token in emoji_sentiment:
            base_score += emoji_sentiment[token]
        elif token in polarity_score_dict:
            base_score += polarity_score_dict[token]
        elif any(0x1F600 <= ord(char) <= 0x1F64F for char in token):
            base_score += 1.0  # fallback for unknown emoji

        # Positional weight
        position_weight = 1.0
        if i == 0:
            position_weight = 1.2
        elif i == len(ngrams) - 1:
            position_weight = 1.1
        base_score *= position_weight

        # Boosters
        for token_boost, boost_value in booster_words.items():
            if token_boost in tokens:
                base_score *= boost_value

        sentiment_scores.append((token, base_score))
    return sentiment_scores

# === 4. Adjust with dependencies ===
def adjust_with_dependencies(tokens_with_scores: List[Tuple[str, float]], dependencies: List[Tuple[str, str, str]]) -> List[Tuple[str, float]]:
    adjusted_scores = tokens_with_scores.copy()
    for word, dep, head in dependencies:
        for i, (token, score) in enumerate(adjusted_scores):
            if token == word and dep == "neg":
                adjusted_scores[i] = (token, -score)
            elif token == word and dep == "amod":
                adjusted_scores[i] = (token, score * 1.3)
            elif token == word and dep == "punct" and head == "!":
                adjusted_scores[i] = (token, score * 1.5)
            elif token == word and dep == "punct" and head == "...":
                adjusted_scores[i] = (token, score * 0.8)
    return adjusted_scores

# === 5. Final calculate_sentiment ===
def calculate_sentiment(text: str) -> Dict[str, float]:
    tokens = tokenize_text(text)
    tokens_with_scores = analyze_tokens(tokens)
    dependencies = analyze_dependencies(text)
    adjusted_scores = adjust_with_dependencies(tokens_with_scores, dependencies)

    compound_score = sum(score for _, score in adjusted_scores)
    max_score = max(abs(compound_score), 1.0)
    compound_score = compound_score / max_score

    positive_score = sum(max(score, 0) for _, score in adjusted_scores)
    negative_score = sum(min(score, 0) for _, score in adjusted_scores)
    neutral_score = len([score for _, score in adjusted_scores if abs(score) < 0.2])

    total_tokens = len(tokens)
    if total_tokens < 5:
        compound_score *= 0.9

    has_negation = any(dep[1] == "neg" for dep in dependencies)
    num_boosters = sum(1 for booster in booster_words if booster in tokens)
    emoji_score = sum(
        emoji_sentiment.get(token, 0)
        for token in tokens
        if token in emoji_sentiment
    )

    # Mathematical confidence calculation
    # Confidence = 1 - entropy/max_entropy where entropy = -Σ(p_i * log(p_i))
    import numpy as np
    scores = [positive_score, abs(negative_score), neutral_score]
    total_score = sum(scores) + 1e-10  # Avoid division by zero
    probabilities = [s/total_score for s in scores]
    entropy = -sum(p * np.log(p + 1e-10) for p in probabilities if p > 0)
    max_entropy = np.log(3)  # log(number_of_classes)
    confidence = 1 - (entropy / max_entropy)
    
    # Statistical variance of sentiment scores
    score_variance = np.var([score for _, score in adjusted_scores]) if adjusted_scores else 0
    
    return {
        "positive": positive_score / total_tokens if total_tokens else 0,
        "negative": abs(negative_score) / total_tokens if total_tokens else 0,
        "neutral": neutral_score / total_tokens if total_tokens else 0,
        "compound": max(min(compound_score, 1), -1),
        "confidence": confidence,
        "entropy": entropy,
        "score_variance": score_variance,
        "dependencies": dependencies,
        "has_negation": int(has_negation),
        "num_boosters": num_boosters,
        "emoji_score": emoji_score,
        "num_tokens": total_tokens,
        "formula": "S = Σ(w_i * s_i * p_i * b_i) / max(|Σ|, 1)"
    }
