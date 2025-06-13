# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
from newsapi import NewsApiClient
import re
from sentence_transformers import SentenceTransformer
import tf_keras
import joblib


# Initializing Variables - Important for rest of the program
newsapi = NewsApiClient(api_key='4e6d0e08be6445ee927d5a5896090114')


# NewsAPI fetching articles
def fetch_articles(source, label, max_pages=5):
    all_articles = []
    for page in range(1, max_pages+1):
        resp = newsapi.get_everything(
            sources=source,
            language='en',
            sort_by='publishedAt',
            page_size=20,
            page=page
        )
        for art in resp.get('articles', []):
            all_articles.append({'text': art.get('content') or art.get('description'),
                                 'label': label,
                                 'source': source})
    return all_articles

# Example:
left_src = 'the-new-york-times,the-washington-post,msnbc'
right_src = 'fox-news,breitbart-news'
neutral_src = 'associated-press'  # mainstream center

left = fetch_articles(left_src, 'left')
right = fetch_articles(right_src, 'right')
neutral = fetch_articles(neutral_src, 'neutral')

df = pd.DataFrame(left + right + neutral).dropna(subset=['text'])
# print(df.label.value_counts(), len(df))

def clean_text(txt):
    if not isinstance(txt, str):
        return ""

    txt = txt.lower()  # normalize case
    txt = re.sub(r'\[.*?\]', '', txt)  # remove [bracketed] text
    txt = re.sub(r'https?://\S+', '', txt)  # remove links
    txt = re.sub(r'\s+', ' ', txt)  # collapse multiple whitespace into one space
    txt = txt.strip()  # remove leading/trailing whitespace

    # Optional: remove common footer junk
    footer_keywords = ["click here", "subscribe", "sign up", "advertisement"]
    for keyword in footer_keywords:
        txt = txt.replace(keyword, '')

    return txt

df['text'] = df['text'].apply(clean_text)
# df = df[df['text'].str.len() > 200]  # keep articles of reasonable length

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# Make sure it shows a non-zero count for each label.

print("Final data size:", len(df))
print("Label distribution:\n", df['label'].value_counts())

# # Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
joblib.dump(clf, 'bias_classifier.pkl')








    