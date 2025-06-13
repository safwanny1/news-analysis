# News Article Bias Detection System

A machine learning-powered system that analyzes news articles and predicts their political bias orientation (left, right, or neutral). This project uses natural language processing techniques and supervised learning to classify articles based on their content.

## 🎯 Project Overview

This system fetches articles from various news sources, trains a machine learning model to recognize bias patterns, and provides predictions for new articles. It can analyze articles either from a dataset or directly from URLs through web scraping.

## ✨ Features

- **Automated Data Collection**: Fetches articles from multiple news sources using NewsAPI
- **Text Preprocessing**: Cleans and normalizes article content for better analysis
- **Machine Learning Classification**: Uses sentence embeddings and logistic regression for bias detection
- **URL Analysis**: Can analyze articles directly from web URLs
- **Web Interface**: Flask-based web application for easy interaction
- **Three-Class Classification**: Categorizes articles as left-leaning, right-leaning, or neutral

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning**: scikit-learn, sentence-transformers
- **Web Framework**: Flask
- **Data Processing**: pandas, numpy
- **Web Scraping**: requests, BeautifulSoup
- **News Data**: NewsAPI
- **NLP**: TF-IDF, SentenceTransformer embeddings

## 📋 Prerequisites

Before running this project, ensure you have:

- Python 3.7 or higher
- NewsAPI key (free account at [newsapi.org](https://newsapi.org))
- Internet connection for fetching articles and embeddings

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd news-bias-detection
```

2. **Install required packages**:
```bash
pip install pandas scikit-learn flask newsapi-python sentence-transformers requests beautifulsoup4 tf-keras
```

3. **Set up NewsAPI key**:
   - Sign up at [newsapi.org](https://newsapi.org) to get your free API key
   - Replace the API key in the code:
   ```python
   newsapi = NewsApiClient(api_key='YOUR_API_KEY_HERE')
   ```

## 🚀 Usage

### Training the Model

The system automatically trains when run:

1. **Run the main script**:
```bash
newsanalysis.py
```

2. **The system will**:
   - Fetch articles from predefined news sources
   - Clean and preprocess the text
   - Generate sentence embeddings
   - Train a logistic regression classifier
   - Display model performance metrics

### Predicting Article Bias

#### Method 1: Direct Text Analysis
```python
article_text = "Your article content here..."
bias_prediction = predict_article_bias(article_text)
print(f"Predicted bias: {bias_prediction}")
```

#### Method 2: URL Analysis
```python
url = "https://example-news-site.com/article"
article_content = get_article_from_url(url)
bias_prediction = predict_article_bias(article_content)
print(f"Predicted bias: {bias_prediction}")
```

### Web Interface

1. **Start the Flask server**:
```bash
newsanalysis.py
```

2. **Access the web interface**:
   - Open your browser to `http://localhost:8000`
   - Use the web interface to analyze articles

## 📊 Data Sources

The model is trained on articles from categorized news sources:

### Left-leaning Sources
- The New York Times
- The Washington Post
- MSNBC

### Right-leaning Sources
- Fox News
- Breitbart News

### Neutral Sources
- Associated Press

## 🔍 How It Works

1. **Data Collection**: Articles are fetched from NewsAPI using predefined source categories
2. **Text Preprocessing**: Articles undergo cleaning to remove URLs, brackets, and normalize formatting
3. **Feature Extraction**: Text is converted to numerical embeddings using SentenceTransformer
4. **Model Training**: Logistic regression classifier learns patterns from labeled training data
5. **Prediction**: New articles are processed through the same pipeline to predict bias

## ⚙️ Configuration

### Modifying News Sources

To change the news sources, update these variables:
```python
left_src = 'source1,source2,source3'     # Left-leaning sources
right_src = 'source1,source2'            # Right-leaning sources  
neutral_src = 'source1'                  # Neutral sources
```

### Adjusting Data Collection

Modify the `fetch_articles` function parameters:
- `max_pages`: Number of pages to fetch per source
- `page_size`: Articles per page (max 20 for NewsAPI)

## 📈 Model Performance

The system uses stratified train-test split (80/20) and provides:
- Classification report with precision, recall, and F1-scores
- Label distribution analysis
- Confusion matrix (can be added for detailed evaluation)

## ⚠️ Limitations

- **API Limits**: NewsAPI free tier has request limitations
- **Source Bias**: Model accuracy depends on the quality of source categorization
- **Text Length**: Very short articles may not provide enough context
- **Language**: Currently optimized for English language articles
- **Temporal Bias**: Model may reflect biases present in training timeframe

## 🔒 Privacy and Ethics

- The system analyzes publicly available news content
- No personal data is collected or stored
- Bias detection is based on source categorization which may not always be accurate
- Results should be interpreted as computational estimates, not definitive judgments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your NewsAPI key is valid and not expired
2. **Import Errors**: Install missing packages using pip
3. **Memory Issues**: Reduce `max_pages` if processing too many articles
4. **Network Errors**: Check internet connection for API calls and web scraping

### Dependencies Issues
```bash
# If you encounter TensorFlow issues
pip install --upgrade tensorflow

# For sentence-transformers issues
pip install --upgrade sentence-transformers torch
```

## 📄 License

This project is created for educational purposes. Please ensure compliance with NewsAPI terms of service and respect website robots.txt when scraping.

## 👥 Authors

- Zach Arrastia
- Cris Rizzi
- Safwan Chowdhury

*AIIE Gen AI Project*

## 📞 Support

For questions or issues, please create an issue in the repository or contact the project maintainers.

---

**Note**: This tool provides computational estimates of bias and should not be considered as definitive political categorization. Always cross-reference with multiple sources and critical thinking when evaluating news content.