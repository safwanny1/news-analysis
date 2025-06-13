# News Article Bias Detection System

A machine learning-powered web application that automatically detects political bias in news articles using natural language processing and classification techniques.

## 🎯 Project Overview

This project implements an automated news bias detection system that can classify news articles into three categories:
- **Left**: Liberal/progressive bias
- **Right**: Conservative bias  
- **Neutral**: Minimal political bias

The system uses sentence embeddings and logistic regression to analyze article content and predict political lean based on training data from various news sources.

## ✨ Features

- **Automated Data Collection**: Fetches articles from major news sources using NewsAPI
- **Text Preprocessing**: Cleans and normalizes article content for analysis
- **ML Classification**: Uses SentenceTransformer embeddings with Logistic Regression
- **URL Analysis**: Can analyze bias of any news article from a URL
- **Web Interface**: Flask-based web application for easy interaction
- **Real-time Prediction**: Instant bias detection for new articles

## 🏗️ Architecture

```
Data Collection (NewsAPI) → Text Preprocessing → Feature Extraction (SentenceTransformer) 
    ↓
Model Training (Logistic Regression) → Bias Prediction → Web Interface (Flask)
```

## 📋 Requirements

### Core Dependencies
```
newsapi-python
pandas
scikit-learn
flask
sentence-transformers
tensorflow
requests
beautifulsoup4
```

### System Requirements
- Python 3.7+
- Internet connection for NewsAPI and article fetching
- Minimum 4GB RAM (for SentenceTransformer model)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd news-bias-detection
```

2. **Install dependencies**
```bash
pip install newsapi-python pandas scikit-learn flask sentence-transformers tensorflow requests beautifulsoup4
```

3. **Set up NewsAPI key**
   - Get a free API key from [NewsAPI](https://newsapi.org/)
   - Replace the hardcoded API key in the code:
   ```python
   newsapi = NewsApiClient(api_key='YOUR_API_KEY_HERE')
   ```

4. **Create templates directory** (for Flask web interface)
```bash
mkdir templates
```

5. **Create basic HTML template** (`templates/index.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <title>News Bias Detector</title>
</head>
<body>
    <h1>News Article Bias Detection</h1>
    <p>Welcome to the News Bias Detection System</p>
</body>
</html>
```

## 💻 Usage

### Basic Usage

1. **Run the application**
```bash
python your_script_name.py
```

2. **Access the web interface**
   - Open browser to `http://localhost:8000`

### Programmatic Usage

```python
# Predict bias of an article from URL
url = "https://example-news-site.com/article"
article_text = get_article_from_url(url)
bias_prediction = predict_article_bias(article_text)
print(f"Predicted bias: {bias_prediction}")

# Predict bias of raw text
text = "Your article content here..."
bias = predict_article_bias(text)
```

## 📊 Data Sources

The system is trained on articles from categorized news sources:

| Category | Sources |
|----------|---------|
| **Left** | The New York Times, The Washington Post, MSNBC |
| **Right** | Fox News, Breitbart News |
| **Neutral** | Associated Press |

## 🔧 Model Details

- **Text Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Classifier**: Logistic Regression with max 1000 iterations
- **Train/Test Split**: 80/20 with stratification
- **Text Preprocessing**: Lowercasing, URL removal, whitespace normalization

## 📈 Performance

The model performance can be evaluated using the built-in classification report:
```python
print(classification_report(y_test, y_pred))
```

## 🔒 Security Considerations

**⚠️ Important Security Notes:**
- **API Key Exposure**: The current code contains a hardcoded NewsAPI key. For production use:
  - Store API keys in environment variables
  - Use `.env` files with `python-dotenv`
  - Never commit API keys to version control

**Recommended Fix:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
```

## 📁 Project Structure

```
news-bias-detection/
├── main.py                 # Main application file
├── templates/              # Flask HTML templates
│   └── index.html         # Main web interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## ⚙️ Configuration

### Customizable Parameters

- **Source Selection**: Modify news sources in the fetch functions
- **Model Parameters**: Adjust LogisticRegression parameters
- **Text Cleaning**: Customize the `clean_text()` function
- **Embedding Model**: Change SentenceTransformer model
- **API Limits**: Modify `max_pages` and `page_size` for data collection

### Environment Variables

Create a `.env` file:
```
NEWS_API_KEY=your_newsapi_key_here
FLASK_DEBUG=True
FLASK_PORT=8000
```

## 🚨 Known Issues

1. **Limited Training Data**: Only uses specific news sources
2. **API Rate Limits**: NewsAPI has daily request limits
3. **Template Missing**: Flask app needs proper HTML templates
4. **Error Handling**: Minimal error handling for network requests
5. **Model Persistence**: Model retrains on every run

## 🔮 Future Enhancements

- [ ] **Model Persistence**: Save/load trained models
- [ ] **Enhanced Web UI**: Rich HTML interface with forms
- [ ] **Batch Processing**: Upload and analyze multiple articles
- [ ] **Confidence Scores**: Show prediction confidence
- [ ] **Source Expansion**: Include more diverse news sources
- [ ] **Real-time Training**: Continuously update model
- [ ] **API Endpoint**: RESTful API for external integration
- [ ] **Visualization**: Charts showing bias distribution trends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source. Please add an appropriate license file.

## 👥 Authors

- **Zach, Cris, Safwan** - AIIE Gen AI Project

## 🙏 Acknowledgments

- [NewsAPI](https://newsapi.org/) for news data access
- [SentenceTransformers](https://www.sbert.net/) for text embeddings
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Flask](https://flask.palletsprojects.com/) for web framework

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This project was developed as part of an AI/ML educational project. Results may vary based on training data and should be used for educational purposes.