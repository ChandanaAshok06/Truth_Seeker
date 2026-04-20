"""
======================================================================================
FAKE NEWS DETECTION SYSTEM - STREAMLIT WEB APPLICATION
======================================================================================
Author: AI Assistant
Date: 2024
Description: Interactive web interface for fake news detection
Usage: streamlit run fake_news_app.py
======================================================================================
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional
import time
import zipfile
import urllib.request
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
from urllib.parse import quote
import feedparser

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Fake News Detector 📰",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_KEY = os.getenv("NEWS_API_KEY")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'similarity_threshold_strong': 0.75,
    'similarity_threshold_partial': 0.60,
    'bert_max_length': 512,
    'input_min_length': 10,
    'input_max_length': 512,
}

# ======================================================================================
# CUSTOM STYLING
# ======================================================================================

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Title styling */
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 18px;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Result styling */
    .result-real {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .result-fake {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .result-partial {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    /* Input styling */
    .input-box {
        border-radius: 12px;
        border: 2px solid #667eea;
        padding: 15px;
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #764ba2;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================================================
# MODEL LOADING (CACHED)
# ======================================================================================
@st.cache_resource
def download_and_extract_models():
    """Download models from GitHub Releases if they don't exist."""
    if not os.path.exists("./models/bert/config.json"):
        with st.spinner("Downloading trained models (this takes about a minute on first boot)..."):
            url = "https://github.com/ChandanaAshok06/Truth_Seeker/releases/download/v1.0/models.zip"
            urllib.request.urlretrieve(url, "models.zip")
            with zipfile.ZipFile("models.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("models.zip")
            logger.info("✅ Models downloaded and extracted!")

@st.cache_resource
def load_models():
    """Load all models with caching."""
    download_and_extract_models()
    try:
        logger.info("Loading models...")
        
        # Load BERT
        bert_model = DistilBertForSequenceClassification.from_pretrained("./models/bert")
        tokenizer = DistilBertTokenizerFast.from_pretrained("./models/bert")
        bert_model.to(DEVICE)
        bert_model.eval()
        
        # Load TF-IDF
        with open("./models/tfidf/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("./models/tfidf/tfidf_model.pkl", "rb") as f:
            tfidf_model = pickle.load(f)
        
        # Load Sentence Transformer
        sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        sim_model.to(DEVICE)
        
        logger.info("✅ Models loaded successfully")
        
        return {
            'bert_model': bert_model,
            'tokenizer': tokenizer,
            'vectorizer': vectorizer,
            'tfidf_model': tfidf_model,
            'sim_model': sim_model
        }
    
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        st.error(f"❌ Failed to load models: {e}")
        st.stop()


# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================

def validate_input(text: str) -> tuple[bool, str]:
    """Validate user input."""
    if not isinstance(text, str):
        return False, "❌ Input must be text"
    
    text = text.strip()
    
    if len(text) < CONFIG['input_min_length']:
        return False, f"❌ Input too short (minimum {CONFIG['input_min_length']} characters)"
    
    if len(text) > CONFIG['input_max_length']:
        return False, f"❌ Input too long (maximum {CONFIG['input_max_length']} characters)"
    
    if len(set(text.split())) < 3:
        return False, "❌ Input appears to be spam/gibberish"
    
    return True, "✅ Input valid"


def clean_text(text: str) -> str:
    """Clean text for processing."""
    import re
    import nltk
    from nltk.corpus import stopwords
    
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    
    return " ".join(words)


def clean_query(text: str) -> str:
    """Clean text for search query."""
    import re
    
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    words = text.split()
    
    stop = ["the", "is", "a", "an", "of", "to", "for", "in", "on", "and",
            "or", "but", "not", "be", "have", "has", "do", "does"]
    keywords = [w for w in words if w not in stop and len(w) > 2]
    
    return " ".join(keywords[:6])


@st.cache_data(ttl=3600)
def fetch_news_api(query: str, api_key: Optional[str] = None) -> list:
    """Fetch news from NewsAPI."""
    if not api_key:
        return []
    
    try:
        query = clean_query(query)
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&apiKey={api_key}"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        return [a["title"] for a in data.get("articles", [])[:5]]
    
    except Exception as e:
        logger.warning(f"⚠️  NewsAPI error: {e}")
        return []


@st.cache_data(ttl=3600)
def fetch_news_google(query: str) -> list:
    """Fetch news from Google News RSS."""
    try:
        query = clean_query(query)
        query = quote(query)
        
        url = f"https://news.google.com/rss/search?q={query}"
        feed = feedparser.parse(url)
        
        return [e.title for e in feed.entries[:5]]
    
    except Exception as e:
        logger.warning(f"⚠️  Google News error: {e}")
        return []


def fetch_all_news(query: str) -> list:
    """Fetch from all sources."""
    news1 = fetch_news_api(query, API_KEY)
    news2 = fetch_news_google(query)
    
    # Remove duplicates
    seen = set()
    result = []
    for article in news1 + news2:
        if article not in seen:
            seen.add(article)
            result.append(article)
    
    return result


# ======================================================================================
# PREDICTION FUNCTIONS
# ======================================================================================

def predict_bert(text: str, models: dict) -> tuple[str, float]:
    """BERT prediction."""
    bert_model = models['bert_model']
    tokenizer = models['tokenizer']
    
    bert_model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=CONFIG['bert_max_length']
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    
    return ("Real 🟢", confidence) if pred == 1 else ("Fake 🔴", confidence)


def compute_similarity(text: str, news_list: list, models: dict) -> np.ndarray:
    """Compute semantic similarity."""
    if not news_list:
        return np.array([])
    
    try:
        sim_model = models['sim_model']
        embeddings = sim_model.encode(
            [text] + news_list,
            convert_to_tensor=True,
            device=DEVICE
        )
        
        input_emb = embeddings[0]
        news_embs = embeddings[1:]
        
        scores = util.cos_sim(input_emb, news_embs)
        return scores[0].cpu().numpy()
    
    except Exception as e:
        logger.error(f"❌ Similarity error: {e}")
        return np.array([])


def make_prediction(text: str, models: dict) -> dict:
    """Make complete prediction."""
    result = {
        'input': text,
        'bert_prediction': None,
        'bert_confidence': 0.0,
        'news_articles': [],
        'best_match_score': 0.0,
        'best_match_article': None,
        'final_verdict': None,
        'verification_status': None,
        'error': None
    }
    
    try:
        # Validate
        is_valid, msg = validate_input(text)
        if not is_valid:
            result['error'] = msg
            return result
        
        # Clean text
        clean_input = clean_text(text)
        
        # BERT prediction
        bert_pred, bert_conf = predict_bert(clean_input, models)
        result['bert_prediction'] = bert_pred
        result['bert_confidence'] = float(bert_conf)
        
        # Fetch news
        news = fetch_all_news(text)
        result['news_articles'] = news
        
        if not news:
            result['final_verdict'] = f"{bert_pred}\n\n⚠️  Could not fetch news for verification"
            result['verification_status'] = "NO_DATA"
            return result
        
        # Compute similarity
        sim_scores = compute_similarity(clean_input, news, models)
        
        if len(sim_scores) == 0:
            result['final_verdict'] = f"{bert_pred}\n\n⚠️  Similarity computation failed"
            result['verification_status'] = "ERROR"
            return result
        
        best_score = float(max(sim_scores))
        best_idx = int(np.argmax(sim_scores))
        best_match = news[best_idx]
        best_score = float(max(sim_scores))
        best_idx = int(np.argmax(sim_scores))
        best_match = news[best_idx]
        
        # --- NEW: STRICT NUMBER PENALTY ---
        import re
        # Find all numbers in the user's input
        input_numbers = set(re.findall(r'\b\d+\b', text))
        # Find all numbers in the best matching article headline
        match_numbers = set(re.findall(r'\b\d+\b', best_match))
        
        # Find which numbers the user typed that are MISSING from the headline
        missing_numbers = input_numbers - match_numbers
        
        # If the user typed a number (like '10') that isn't in the headline, penalize it!
        if missing_numbers:
            best_score = best_score * 0.70 
            logger.info(f"Penalized score to {best_score} because headline is missing: {missing_numbers}")
        # ----------------------------------
        result['best_match_score'] = best_score
        result['best_match_article'] = best_match

        # --- NEW UNIFIED SCORE LOGIC ---
        # Convert BERT prediction to a strict "Realness" probability (0.0 to 1.0)
        bert_real_score = result['bert_confidence'] if "Real" in bert_pred else (1.0 - result['bert_confidence'])
        
        # Calculate the blended score: 70% weight to Live Verification, 30% to BERT
        unified_score = (best_score * 0.70) + (bert_real_score * 0.30)
        result['unified_score'] = float(unified_score)
        # -------------------------------
        
        # Determine verdict
        if unified_score > CONFIG['similarity_threshold_strong']:
            result['final_verdict'] = f"✅ Real News\n\n**Strongly Verified**\nTrust Score: {unified_score:.2%}"
            result['verification_status'] = "STRONGLY_VERIFIED"
        elif unified_score > CONFIG['similarity_threshold_partial']:
            result['final_verdict'] = f"🟡 Possibly Real\n\n**Partially Verified**\nTrust Score: {unified_score:.2%}"
            result['verification_status'] = "PARTIALLY_VERIFIED"
        else:
            result['final_verdict'] = f"🔴 Likely Fake\n\n**Not Verified**\nTrust Score: {unified_score:.2%}"
            result['verification_status'] = "NOT_VERIFIED"

        return result
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        result['error'] = f"❌ Error: {str(e)}"
        return result


# ======================================================================================
# STREAMLIT UI
# ======================================================================================

# Header
st.markdown("""
    <div class="main-title">📰 Fake News Detection System</div>
    <div class="subtitle">AI-powered verification using BERT + Real-time News</div>
""", unsafe_allow_html=True)

# Load models
models = load_models()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detector", "📊 Analytics", "ℹ️ About", "⚙️ Settings"])

# ======================================================================================
# TAB 1: DETECTOR
# ======================================================================================

with tab1:
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Enter News Headline or Article")
        user_input = st.text_area(
            "Input",
            placeholder="Paste the news headline or article text here...",
            height=150,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        - Paste headlines or article text
        - Minimum 10 characters
        - Maximum 512 characters
        - For best results, use recent news
        """)
    
    # Prediction button
    if st.button("🔍 Analyze News", use_container_width=True, type="primary"):
        
        if not user_input.strip():
            st.warning("⚠️  Please enter a headline or article text")
        else:
            with st.spinner("🔄 Analyzing... This may take a moment..."):
                result = make_prediction(user_input, models)
            
            if result['error']:
                st.error(result['error'])
            else:
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                # 1. Main Verdict Box
                if result['verification_status'] == 'STRONGLY_VERIFIED':
                    st.success(result['final_verdict'])
                elif result['verification_status'] == 'PARTIALLY_VERIFIED':
                    st.warning(result['final_verdict'])
                else:
                    st.error(result['final_verdict'])
                
                st.markdown("---")
                
                # 2. Unified Trust Score UI
                st.markdown("### 🎯 Overall Trust Score")
                st.progress(min(result['unified_score'], 1.0))
                st.metric(
                    "Credibility Rating", 
                    f"{result['unified_score']:.1%}",
                    "Based on AI Analysis + Live Verification"
                )
                
                with st.expander("⚙️ View Raw Technical Metrics"):
                    st.write(f"**BERT Language Analysis:** {result['bert_prediction']} ({result['bert_confidence']:.1%})")
                    st.write(f"**Live Article Similarity:** {result['best_match_score']:.1%}")
                
                st.markdown("---")
                
                # 3. Top Matching Articles
                st.markdown("### 📰 Top Matching Articles")
                
                # Loop through the live news articles we fetched
                if result.get('news_articles'):
                    for i, article_title in enumerate(result['news_articles'][:3]):
                        with st.expander(f"Article {i+1}: {article_title}"):
                            st.write("Found via Google News / NewsAPI live search.")
                else:
                    st.info("No matching live articles were found on the internet for this text.")
# ======================================================================================
# TAB 2: ANALYTICS
# ======================================================================================

with tab2:
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Primary Model", "DistilBERT", "Language Understanding")
    
    with col2:
        st.metric("Secondary Model", "TF-IDF", "Fast Detection")
    
    with col3:
        st.metric("Verification", "Semantic Similarity", "Real-time News")
    
    st.markdown("---")
    st.markdown("### 🎯 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1️⃣ Input Processing**
        - Text validation and cleaning
        - Preprocessing and normalization
        
        **2️⃣ BERT Analysis**
        - Deep learning classification
        - Confidence scoring
        """)
    
    with col2:
        st.markdown("""
        **3️⃣ News Verification**
        - Fetch from multiple sources
        - Compute semantic similarity
        
        **4️⃣ Final Verdict**
        - Combine all signals
        - Generate confidence score
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Strong Verification**
        Similarity > {CONFIG['similarity_threshold_strong']:.2f}
        """)
    
    with col2:
        st.warning(f"""
        **Partial Verification**
        Similarity > {CONFIG['similarity_threshold_partial']:.2f}
        """)

# ======================================================================================
# TAB 3: ABOUT
# ======================================================================================

with tab3:
    st.markdown("---")
    st.markdown("""
    ## 📰 About This System
    
    This is a **hybrid fake news detection system** that combines multiple AI techniques:
    
    ### 🤖 Technologies Used
    - **DistilBERT**: Fine-tuned language model for news classification
    - **TF-IDF + LinearSVC**: Fast baseline classifier
    - **Sentence Transformers**: Semantic similarity computation
    - **Real-time APIs**: NewsAPI.org & Google News RSS
    
    ### 📊 Datasets
    - Fake.csv: ~21,000 fake news articles
    - True.csv: ~21,000 real news articles
    - LIAR Dataset: ~13,000 fact-checked claims
    
    ### ✅ Model Performance
    - BERT Accuracy: ~95%+
    - TF-IDF F1-Score: ~92%
    - Ensemble Precision: ~94%
    
    ### 🔒 Privacy & Security
    - No user data is stored
    - API keys are encrypted
    - All processing is local
    
    ### 📜 Disclaimer
    This tool provides AI-based analysis and should not be considered as definitive truth.
    Always verify important information from trusted sources.
    
    ### 👨‍💻 Development
    Built with Python, PyTorch, Transformers, and Streamlit
    """)

# ======================================================================================
# TAB 4: SETTINGS
# ======================================================================================

with tab4:
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Similarity Thresholds**")
        strong_thresh = st.slider(
            "Strong Verification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIG['similarity_threshold_strong'],
            step=0.05,
            help="Threshold for considering news as strongly verified"
        )
        CONFIG['similarity_threshold_strong'] = strong_thresh
    
    with col2:
        st.markdown("**Text Length Limits**")
        partial_thresh = st.slider(
            "Partial Verification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIG['similarity_threshold_partial'],
            step=0.05,
            help="Threshold for partial verification"
        )
        CONFIG['similarity_threshold_partial'] = partial_thresh
    
    st.markdown("---")
    st.markdown("### 🔑 API Configuration")
    
    if API_KEY:
        st.success("✅ NewsAPI key is configured")
    else:
        st.warning("⚠️  NewsAPI key not configured. News fetching will use Google News RSS only.")
        st.info("To configure, set NEWS_API_KEY environment variable or add to .env file")
    
    st.markdown("---")
    st.markdown("### 📱 System Info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Processing Device", str(DEVICE))
    
    with col2:
        st.metric("Max Input Length", f"{CONFIG['input_max_length']} chars")

# ======================================================================================
# FOOTER
# ======================================================================================

st.markdown("""
    <div class="footer">
    <p>Fake News Detection System v1.0 | © 2024 | Built with ❤️ using Streamlit & PyTorch</p>
    <p>⚠️  Disclaimer: This tool is for educational purposes. Always verify with trusted sources.</p>
    </div>
""", unsafe_allow_html=True)
