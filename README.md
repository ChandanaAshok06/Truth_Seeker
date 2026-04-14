# 🔍 Truth Seeker: AI-Powered Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)](https://huggingface.co/)

An AI-powered fake news detector that doesn't just guess—it verifies. This project combines deep learning (DistilBERT + TF-IDF) with real-time web scraping to fact-check claims against live news sources. Built with clean, production-ready code and wrapped in an interactive Streamlit app.

## 🎯 Features
- **AI-Powered Detection**: Uses the DistilBERT transformer model combined with TF-IDF for robust fake news classification.
- **Real-Time Web Scraping**: Integrates with live news sources to dynamically verify claims.
- **Interactive UI**: Streamlit-based web application for easy, intuitive access.
- **Production-Ready Code**: Clean, well-structured, and highly maintainable codebase.
- **Fact-Checking**: Validates claims against multiple active news sources and fact-checking databases.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- `pip` or `conda` package manager

### Installation & Setup
1. **Clone the repository:**
```bash
git clone https://github.com/ChandanaAshok06/Truth_Seeker.git
cd Truth_Seeker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

The app will automatically open in your browser at http://localhost:8501

## 📋 How to Use
- Enter a news claim or headline in the input field.
- Click "Check Veracity" to run the analysis.
- View the model's prediction result along with its confidence score.
- Review the detailed fact-checking logs retrieved from live news sources.

## 🏗️ Project Structure
```
Truth_Seeker/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── models/                # Pre-trained ML models (DistilBERT, TF-IDF)
├── utils/                 # Helper functions and utilities
├── data/                  # Sample datasets
└── README.md              # Project documentation
```

## 🤖 Model Details
- **Primary Model**: DistilBERT (via Hugging Face Transformers)
- **Feature Engineering**: TF-IDF vectorization
- **Architecture**: Hybrid ensemble approach combining deep contextual learning (BERT) with statistical baseline methods (TF-IDF).

## 🛠️ Development & Testing
To run the test suite:
```bash
python -m pytest tests/
```

## 🤝 Contributing
We welcome contributions! To contribute:
- Fork the repository
- Create a feature branch (`git checkout -b feature/AmazingFeature`)
- Commit your changes (`git commit -m 'Add AmazingFeature'`)
- Push to the branch (`git push origin feature/AmazingFeature`)
- Open a Pull Request

## ⚠️ Disclaimer
This tool is designed to assist in identifying potential misinformation for educational and analytical purposes. While it uses advanced AI models, no detection system is 100% accurate. Always cross-reference with multiple reliable primary sources before drawing definitive conclusions.

## 👤 Author
ChandanaAshok06  
GitHub: @ChandanaAshok06  

## 🌐 Live Demo
👉 https://truthseeker-06.streamlit.app/
