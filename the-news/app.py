from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import os
import joblib
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Flask app
app = Flask(__name__)

# Load model
clf = joblib.load('bias_classifier.pkl')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API
genai.configure(api_key="AIzaSyD5CEqeAJAnU_7Ul5TaO4eQ7XCSPw6ZB9U")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Clean text
def clean_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = re.sub(r'\[.*?\]', '', txt)
    txt = re.sub(r'https?://\S+', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    for keyword in ["click here", "subscribe", "sign up", "advertisement"]:
        txt = txt.replace(keyword, '')
    return txt

# Extract article from URL
def get_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
    except Exception as e:
        return f"Error extracting article: {str(e)}"

# Predict bias
def predict_article_bias(article_text):
    cleaned = clean_text(article_text)
    embedding = model.encode([cleaned])[0]
    bias_prediction = clf.predict([embedding])[0]
    probabilities = clf.predict_proba([embedding])[0]
    confidence_scores = dict(zip(clf.classes_, probabilities))
    return bias_prediction, confidence_scores, cleaned

# Generate Gemini response
def generate_gemini_analysis(text, bias, confidence, url=None):
    excerpt = text[:2000] + "..." if len(text) > 2000 else text
    prompt = f"""
You are a political media analyst.

ARTICLE EXCERPT:
{excerpt}

BIAS PREDICTION: {bias}
CONFIDENCE SCORES: {', '.join([f"{k}: {v:.2%}" for k, v in confidence.items()])}
URL: {url if url else "Not provided"}

Please:
1. Summarize the article's main points
2. Explain why it might be classified as '{bias}'
3. Point out language or framing suggesting this
4. Comment on reliability of this result
5. Offer tips on critically reading this article
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# Web UI route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    explanation = None
    confidence = None

    if request.method == "POST":
        form_type = request.form.get("form_type")
        input_text = ""

        if form_type == "file":
            file = request.files.get("fileUpload")
            if file and file.filename:
                input_text = file.read().decode("utf-8")

        elif form_type == "text":
            user_input = request.form.get("textInput")
            if user_input:
                input_text = get_article_from_url(user_input) if user_input.startswith("http") else user_input

        if input_text:
            bias, confidence, cleaned = predict_article_bias(input_text)
            explanation = generate_gemini_analysis(cleaned, bias, confidence, url=user_input if "http" in user_input else None)
            prediction = bias

    return render_template("index.html", prediction=prediction, explanation=explanation, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
