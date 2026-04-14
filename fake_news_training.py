"""
======================================================================================
FAKE NEWS DETECTION SYSTEM - COMPLETE PRODUCTION-READY CODE
======================================================================================
Author: AI Assistant
Date: 2024
Description: Comprehensive fake news detection using TF-IDF, BERT, and real-time verification
======================================================================================
"""

import os
import sys
import logging
import pickle
import warnings
import json
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
import hashlib
import time
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP & ML Libraries
import nltk
import re
import torch
import requests
import feedparser
from urllib.parse import quote
from urllib.error import URLError

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from imblearn.over_sampling import RandomOverSampler

# Transformers & HuggingFace
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# Sentence Transformers
from sentence_transformers import SentenceTransformer, util

# ======================================================================================
# CONFIGURATION & LOGGING SETUP
# ======================================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
 #       logging.FileHandler('fake_news_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration Dictionary
CONFIG = {
    'max_features_tfidf': 5000,
    'test_size': 0.2,
    'random_state': 42,
    'bert_max_length': 512,
    'similarity_threshold_strong': 0.65,
    'similarity_threshold_partial': 0.35,
    'bert_epochs': 2,
    'bert_batch_size': 8,
    'bert_learning_rate': 2e-5,
    'news_sample_size': 10000,
    'news_fetch_timeout': 5,
    'max_articles_per_source': 5,
    'api_rate_limit_delay': 1.0,  # seconds
    'input_max_length': 512,
    'input_min_length': 10,
}

# API Configuration
API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    logger.warning("⚠️  NEWS_API_KEY not set. News fetching will be limited to Google News RSS.")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# NLTK Downloads
nltk.download('stopwords', quiet=True)

# ======================================================================================
# TEXT PROCESSING & VALIDATION
# ======================================================================================

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def validate_input(text: str, max_length: int = CONFIG['input_max_length'],
                   min_length: int = CONFIG['input_min_length']) -> str:
    """
    Validate and sanitize user input text.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed text length
        min_length: Minimum allowed text length
    
    Returns:
        Validated and cleaned text
    
    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(text, str):
        raise ValueError("❌ Input must be text")
    
    text = text.strip()
    
    if len(text) < min_length:
        raise ValueError(f"❌ Input too short (minimum {min_length} characters)")
    
    if len(text) > max_length:
        logger.warning(f"⚠️  Input truncated to {max_length} characters")
        text = text[:max_length]
    
    # Check for mostly spam/gibberish
    if len(set(text.split())) < 3:
        raise ValueError("❌ Input appears to be spam/gibberish")
    
    return text


def clean_text(text: str) -> str:
    """
    Clean and preprocess text for model consumption.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    
    # Split and remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    
    return " ".join(words)


def clean_query(text: str) -> str:
    """
    Clean text for use as a search query.
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned query
    """
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    words = text.split()
    
    # Remove common words
    stop = ["the", "is", "a", "an", "of", "to", "for", "in", "on", "and",
            "or", "but", "not", "be", "have", "has", "do", "does"]
    keywords = [w for w in words if w not in stop and len(w) > 2]
    
    # Return top 6 keywords
    return " ".join(keywords[:6])


# ======================================================================================
# DATA LOADING & PREPARATION
# ======================================================================================

class FakeNewsDataLoader:
    """Load and prepare fake news datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_datasets(self, fake_csv: str, true_csv: str, liar_tsv: str) -> pd.DataFrame:
        """
        Load and combine multiple fake news datasets.
        
        Args:
            fake_csv: Path to fake news CSV
            true_csv: Path to true news CSV
            liar_tsv: Path to LIAR dataset TSV
        
        Returns:
            Combined and shuffled DataFrame
        """
        logger.info("📥 Loading datasets...")
        
        try:
            # Load basic datasets
            fake_df = pd.read_csv(fake_csv, nrows=None)
            true_df = pd.read_csv(true_csv, nrows=None)
            
            logger.info(f"✅ Loaded {len(fake_df)} fake news and {len(true_df)} true news articles")
            
            # Label them
            fake_df["label"] = 0  # Fake
            true_df["label"] = 1  # Real
            
            # Combine
            data = pd.concat([fake_df, true_df], ignore_index=True)
            
            # Load LIAR dataset if available
            try:
                liar_df = pd.read_csv(liar_tsv, sep='\t', header=None)
                liar_df = liar_df[[2, 1]]
                liar_df.columns = ["text", "label"]
                
                # Convert LIAR labels
                liar_df["label"] = liar_df["label"].apply(
                    lambda x: 0 if x in ["false", "pants-fire", "barely-true"] else 1
                )
                
                logger.info(f"✅ Loaded {len(liar_df)} LIAR dataset articles")
                data = pd.concat([data, liar_df], ignore_index=True)
            except FileNotFoundError:
                logger.warning("⚠️  LIAR dataset not found, skipping...")
            
            # Shuffle
            data = data.sample(frac=1, random_state=CONFIG['random_state']).reset_index(drop=True)
            
            logger.info(f"📊 Total dataset size: {len(data)} articles")
            logger.info(f"   - Real news: {(data['label'] == 1).sum()}")
            logger.info(f"   - Fake news: {(data['label'] == 0).sum()}")
            
            return data
        
        except Exception as e:
            logger.error(f"❌ Error loading datasets: {e}")
            raise
    
    @staticmethod
    def check_class_imbalance(y: pd.Series) -> bool:
        """
        Check if dataset has class imbalance.
        
        Args:
            y: Label series
        
        Returns:
            True if imbalanced (< 50% minority class)
        """
        class_ratio = y.value_counts()
        minority_ratio = class_ratio.min() / class_ratio.max()
        return minority_ratio < 0.5


# ======================================================================================
# MODEL 1: TF-IDF + LINEAR SVC
# ======================================================================================

class TFIDFModel:
    """TF-IDF + LinearSVC baseline model."""
    
    def __init__(self, max_features: int = CONFIG['max_features_tfidf']):
        """
        Initialize TF-IDF model.
        
        Args:
            max_features: Max features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        self.model = LinearSVC(
            random_state=CONFIG['random_state'],
            class_weight='balanced',
            max_iter=2000,
            verbose=0
        )
        self.model_name = "TF-IDF + LinearSVC"
        logger.info(f"✅ Initialized {self.model_name}")
    
    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """
        Train TF-IDF model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        logger.info(f"🔨 Training {self.model_name}...")
        
        try:
            X_train_vec = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_vec, y_train)
            logger.info(f"✅ {self.model_name} trained successfully")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test texts
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"📊 Evaluating {self.model_name}...")
        
        try:
            X_test_vec = self.vectorizer.transform(X_test)
            y_pred = self.model.predict(X_test_vec)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, self.model.decision_function(X_test_vec))
            }
            
            logger.info(f"📈 Metrics:")
            for key, val in metrics.items():
                logger.info(f"   {key}: {val:.4f}")
            
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            
            return metrics
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict for single text.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (prediction, confidence)
        """
        text = clean_text(text)
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        
        # Get confidence (distance from decision boundary)
        confidence = abs(self.model.decision_function(vec)[0])
        
        return ("Real 🟢", confidence) if pred == 1 else ("Fake 🔴", confidence)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            pickle.dump(self.vectorizer, open(f"{path}/vectorizer.pkl", "wb"))
            pickle.dump(self.model, open(f"{path}/tfidf_model.pkl", "wb"))
            logger.info(f"✅ Model saved to {path}")
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            self.vectorizer = pickle.load(open(f"{path}/vectorizer.pkl", "rb"))
            self.model = pickle.load(open(f"{path}/tfidf_model.pkl", "rb"))
            logger.info(f"✅ Model loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise


# ======================================================================================
# MODEL 2: DISTILBERT
# ======================================================================================

class BERTModel:
    """DistilBERT fine-tuned for fake news detection."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize BERT model.
        
        Args:
            model_name: Pretrained model name
        """
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        self.model.to(DEVICE)
        self.model_name = "DistilBERT"
        logger.info(f"✅ Initialized {self.model_name}")
    
    def prepare_dataset(self, texts: pd.Series, labels: pd.Series, test_size: float = 0.2):
        """
        Prepare dataset for BERT training.
        
        Args:
            texts: Text data
            labels: Labels
            test_size: Test set fraction
        
        Returns:
            Prepared dataset
        """
        logger.info("📥 Preparing BERT dataset...")
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        # Create HF Dataset
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(test_size=test_size, seed=CONFIG['random_state'])
        
        # Tokenize
        def tokenize_fn(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=CONFIG['bert_max_length']
            )
        
        dataset = dataset.map(tokenize_fn, batched=True)
        
        logger.info(f"✅ Dataset prepared: {len(dataset['train'])} train, {len(dataset['test'])} test")
        
        return dataset
    
    def train(self, dataset, output_dir: str = "./bert_results"):
        """
        Train BERT model.
        
        Args:
            dataset: Prepared dataset
            output_dir: Output directory for checkpoints
        """
        logger.info(f"🔨 Training {self.model_name}...")
        
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=CONFIG['bert_learning_rate'],
                per_device_train_batch_size=CONFIG['bert_batch_size'],
                per_device_eval_batch_size=CONFIG['bert_batch_size'],
                num_train_epochs=CONFIG['bert_epochs'],
                weight_decay=0.01,
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy"
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"]
            )
            
            trainer.train()
            
            logger.info(f"✅ {self.model_name} training completed")
            
            return trainer
        
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def evaluate(self, trainer, dataset) -> Dict:
        """
        Evaluate BERT model.
        
        Args:
            trainer: Trainer object
            dataset: Dataset for evaluation
        
        Returns:
            Metrics dictionary
        """
        logger.info(f"📊 Evaluating {self.model_name}...")
        
        try:
            results = trainer.evaluate(eval_dataset=dataset["test"])
            
            logger.info(f"📈 Evaluation Results:")
            for key, val in results.items():
                logger.info(f"   {key}: {val:.4f}")
            
            return results
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict for single text.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (prediction, confidence)
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=CONFIG['bert_max_length']
            )
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()
        
        return ("Real 🟢", confidence) if pred == 1 else ("Fake 🔴", confidence)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"✅ BERT model saved to {path}")
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(path)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(path)
            self.model.to(DEVICE)
            logger.info(f"✅ BERT model loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise


# ======================================================================================
# REAL-TIME NEWS VERIFICATION
# ======================================================================================

class NewsVerifier:
    """Fetch and verify news from multiple sources."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news verifier.
        
        Args:
            api_key: NewsAPI API key
        """
        self.api_key = api_key
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sim_model.to(DEVICE)
        self.request_count = 0
        self.last_request_time = 0
        logger.info("✅ Initialized NewsVerifier")
    
    def _rate_limit(self) -> None:
        """Implement rate limiting to avoid API throttling."""
        elapsed = time.time() - self.last_request_time
        if elapsed < CONFIG['api_rate_limit_delay']:
            time.sleep(CONFIG['api_rate_limit_delay'] - elapsed)
        self.last_request_time = time.time()
    
    def fetch_news_api(self, query: str) -> List[str]:
        """
        Fetch news from NewsAPI.org.
        
        Args:
            query: Search query
        
        Returns:
            List of article titles
        """
        if not self.api_key:
            logger.debug("⚠️  NewsAPI key not configured, skipping NewsAPI")
            return []
        
        try:
            self._rate_limit()
            
            query = clean_query(query)
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&apiKey={self.api_key}"
            
            response = requests.get(url, timeout=CONFIG['news_fetch_timeout'])
            
            if response.status_code != 200:
                logger.warning(f"⚠️  NewsAPI returned status {response.status_code}")
                return []
            
            data = response.json()
            articles = [article["title"] for article in data.get("articles", [])[:CONFIG['max_articles_per_source']]]
            
            logger.debug(f"✅ Fetched {len(articles)} articles from NewsAPI")
            return articles
        
        except requests.exceptions.Timeout:
            logger.warning("⚠️  NewsAPI timeout")
            return []
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  NewsAPI connection error")
            return []
        except Exception as e:
            logger.warning(f"⚠️  NewsAPI error: {e}")
            return []
    
    def fetch_news_google(self, query: str) -> List[str]:
        """
        Fetch news from Google News RSS.
        
        Args:
            query: Search query
        
        Returns:
            List of article titles
        """
        try:
            self._rate_limit()
            
            query = clean_query(query)
            query = quote(query)
            
            url = f"https://news.google.com/rss/search?q={query}"
            
            feed = feedparser.parse(url)
            
            articles = [entry.title for entry in feed.entries[:CONFIG['max_articles_per_source']]]
            
            logger.debug(f"✅ Fetched {len(articles)} articles from Google News")
            return articles
        
        except URLError:
            logger.warning("⚠️  Google News connection error")
            return []
        except Exception as e:
            logger.warning(f"⚠️  Google News error: {e}")
            return []
    
    def fetch_all_news(self, query: str) -> List[str]:
        """
        Fetch news from all available sources.
        
        Args:
            query: Search query
        
        Returns:
            Combined list of articles
        """
        news1 = self.fetch_news_api(query)
        news2 = self.fetch_news_google(query)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for article in news1 + news2:
            if article not in seen:
                seen.add(article)
                result.append(article)
        
        return result
    
    def compute_similarity(self, input_text: str, news_list: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between input and news articles.
        
        Args:
            input_text: Input text to verify
            news_list: List of news articles
        
        Returns:
            Array of similarity scores
        """
        if not news_list:
            return np.array([])
        
        try:
            embeddings = self.sim_model.encode(
                [input_text] + news_list,
                convert_to_tensor=True,
                device=DEVICE
            )
            
            input_emb = embeddings[0]
            news_embs = embeddings[1:]
            
            scores = util.cos_sim(input_emb, news_embs)
            
            return scores[0].cpu().numpy()
        except Exception as e:
            logger.error(f"❌ Similarity computation error: {e}")
            return np.array([])


# ======================================================================================
# ENSEMBLE PREDICTION SYSTEM
# ======================================================================================

class FakeNewsDetector:
    """Ensemble fake news detection system."""
    
    def __init__(self, tfidf_model: TFIDFModel, bert_model: BERTModel, 
                 news_verifier: NewsVerifier):
        """
        Initialize detector ensemble.
        
        Args:
            tfidf_model: TF-IDF model
            bert_model: BERT model
            news_verifier: News verifier
        """
        self.tfidf = tfidf_model
        self.bert = bert_model
        self.verifier = news_verifier
        self.thresholds = {
            'strong': CONFIG['similarity_threshold_strong'],
            'partial': CONFIG['similarity_threshold_partial']
        }
        logger.info("✅ Initialized FakeNewsDetector")
    
    def set_thresholds(self, strong: float = 0.65, partial: float = 0.35) -> None:
        """
        Set similarity thresholds.
        
        Args:
            strong: Threshold for strong verification
            partial: Threshold for partial verification
        """
        self.thresholds = {'strong': strong, 'partial': partial}
        logger.info(f"🔧 Thresholds updated: strong={strong}, partial={partial}")
    
    def predict(self, text: str) -> Dict:
        """
        Make ensemble prediction with verification.
        
        Args:
            text: Input news text
        
        Returns:
            Dictionary with prediction details
        """
        try:
            # Validate input
            text = validate_input(text)
        except ValueError as e:
            logger.error(f"{e}")
            return {'error': str(e)}
        
        logger.info(f"🔍 Analyzing: {text[:100]}...")
        
        result = {
            'input': text,
            'bert_prediction': None,
            'bert_confidence': None,
            'verification': None,
            'news_articles': [],
            'best_match_score': None,
            'best_match_article': None,
            'final_verdict': None
        }
        
        try:
            # BERT prediction
            bert_pred, bert_conf = self.bert.predict(text)
            result['bert_prediction'] = bert_pred
            result['bert_confidence'] = bert_conf
            
            logger.info(f"🤖 BERT: {bert_pred} (confidence: {bert_conf:.2f})")
            
            # Fetch news for verification
            news = self.verifier.fetch_all_news(text)
            result['news_articles'] = news
            
            if not news:
                result['final_verdict'] = f"{bert_pred} (⚠️  Could not verify - no news found)"
                logger.warning("⚠️  No news articles found for verification")
                return result
            
            # Compute similarity
            sim_scores = self.verifier.compute_similarity(text, news)
            
            if len(sim_scores) == 0:
                result['final_verdict'] = f"{bert_pred} (⚠️  Similarity computation failed)"
                return result
            
            best_score = max(sim_scores)
            best_idx = sim_scores.argmax()
            best_match = news[best_idx]
            
            result['best_match_score'] = best_score
            result['best_match_article'] = best_match
            
            logger.info(f"📰 Best match score: {best_score:.2f}")
            logger.info(f"📰 Best match article: {best_match[:80]}...")
            
            # Determine verification level
            if best_score > self.thresholds['strong']:
                result['verification'] = 'STRONGLY_VERIFIED'
                result['final_verdict'] = f"Real News ✅ (Strongly Verified)\nConfidence: {best_score:.2f}"
            elif best_score > self.thresholds['partial']:
                result['verification'] = 'PARTIALLY_VERIFIED'
                result['final_verdict'] = f"Possibly Real 🟡 (Partially Verified)\nConfidence: {best_score:.2f}"
            else:
                result['verification'] = 'NOT_VERIFIED'
                result['final_verdict'] = f"{bert_pred} (ML prediction only)\nConfidence: {bert_conf:.2f}"
            
            logger.info(f"✅ Verification: {result['verification']}")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            result['error'] = str(e)
            return result
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Make predictions for multiple texts.
        
        Args:
            texts: List of texts to predict
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing {i}/{len(texts)}...")
            results.append(self.predict(text))
        return results
    
    def save(self, path: str) -> None:
        """Save all models."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.tfidf.save(f"{path}/tfidf")
            self.bert.save(f"{path}/bert")
            logger.info(f"✅ All models saved to {path}")
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load all models."""
        try:
            self.tfidf.load(f"{path}/tfidf")
            self.bert.load(f"{path}/bert")
            logger.info(f"✅ All models loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise


# ======================================================================================
# VISUALIZATION & ANALYSIS
# ======================================================================================

def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Confusion matrix saved")


def plot_roc_curve(y_true, y_scores) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    logger.info("✅ ROC curve saved")


def plot_confidence_distribution(confidences: List[float]) -> None:
    """Plot confidence distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Model Confidence Distribution")
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Confidence distribution saved")


def plot_wordcloud(texts: List[str], title: str = "Word Cloud") -> None:
    """Generate and plot word cloud."""
    text = " ".join(texts)
    wc = WordCloud(width=1200, height=600, background_color='white').generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('wordcloud.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Word cloud saved")


# ======================================================================================
# MAIN TRAINING PIPELINE
# ======================================================================================

def main():
    """Main training pipeline."""
    
    logger.info("=" * 80)
    logger.info("🚀 FAKE NEWS DETECTION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # ========== 1. DATA LOADING ==========
    logger.info("\n[STEP 1/5] Loading Data...")
    
    data_loader = FakeNewsDataLoader()
    
    try:
        combined_data = data_loader.load_datasets(
            fake_csv="data/new_fake.csv",
            true_csv="data/new_true.csv",
            liar_tsv="data/train.tsv"
        )
    except FileNotFoundError:
        logger.error("❌ Dataset files not found. Please ensure fake.csv, true.csv, and train.tsv exist.")
        return
    
    # Clean text
    logger.info("🧹 Cleaning text...")
    combined_data["text"] = combined_data["text"].apply(clean_text)
    
    # Check and handle class imbalance
    if data_loader.check_class_imbalance(combined_data["label"]):
        logger.warning("⚠️  Class imbalance detected! Applying oversampling...")
        X_text = combined_data["text"].values.reshape(-1, 1)
        y = combined_data["label"].values
        
        ros = RandomOverSampler(sampling_strategy=0.7, random_state=CONFIG['random_state'])
        try:
            X_resampled, y_resampled = ros.fit_resample(X_text, y)
            combined_data = pd.DataFrame({
                'text': X_resampled.flatten(),
                'label': y_resampled
            })
            logger.info(f"✅ Rebalanced dataset size: {len(combined_data)}")
        except Exception as e:
            logger.warning(f"⚠️  Oversampling failed: {e}, continuing with original data")
    
    # ========== 2. TRAIN-TEST SPLIT ==========
    logger.info("\n[STEP 2/5] Splitting Data...")
    
    X = combined_data["text"]
    y = combined_data["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    logger.info(f"✅ Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # ========== 3. TRAIN TF-IDF MODEL ==========
    logger.info("\n[STEP 3/5] Training TF-IDF Model...")
    
    tfidf_model = TFIDFModel()
    tfidf_model.train(X_train, y_train)
    tfidf_metrics = tfidf_model.evaluate(X_test, y_test)
    
    # ========== 4. TRAIN BERT MODEL ==========
    logger.info("\n[STEP 4/5] Training BERT Model...")
    
    # Sample data for faster training
    sample_size = min(CONFIG['news_sample_size'], len(combined_data))
    sample_data = combined_data.sample(sample_size, random_state=CONFIG['random_state'])
    
    bert_model = BERTModel()
    dataset = bert_model.prepare_dataset(sample_data["text"], sample_data["label"])
    trainer = bert_model.train(dataset, output_dir="./bert_results")
    bert_metrics = bert_model.evaluate(trainer, dataset)
    
    # ========== 5. TEST ENSEMBLE SYSTEM ==========
    logger.info("\n[STEP 5/5] Testing Ensemble System...")
    
    news_verifier = NewsVerifier(api_key=API_KEY)
    detector = FakeNewsDetector(tfidf_model, bert_model, news_verifier)
    
    test_headlines = [
        "New breakthrough in quantum computing announced",
        "World leaders meet for climate summit",
        "AI researchers develop revolutionary algorithm",
        "Economic data shows mixed signals for growth",
        "Tech company announces major layoffs"
    ]
    
    logger.info(f"🧪 Testing on {len(test_headlines)} sample headlines...")
    
    for headline in test_headlines:
        result = detector.predict(headline)
        if 'error' not in result:
            logger.info(f"\n📰 Headline: {headline}")
            logger.info(f"   Verdict: {result['final_verdict']}")
    
    # ========== SAVE MODELS ==========
    logger.info("\n💾 Saving Models...")
    detector.save("./models")
    
    # ========== SUMMARY ==========
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\n📊 TF-IDF Model Metrics:")
    for key, val in tfidf_metrics.items():
        logger.info(f"   {key}: {val:.4f}")
    logger.info(f"\n📊 BERT Model Metrics:")
    for key, val in bert_metrics.items():
        logger.info(f"   {key}: {val:.4f}")
    logger.info(f"\n💾 Models saved to ./models/")
    logger.info(f"📝 Logs saved to fake_news_detector.log")


if __name__ == "__main__":
    main()
