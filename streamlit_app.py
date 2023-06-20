import subprocess
import sys

# Fungsi untuk menginstal dependensi menggunakan pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Menginstal dependensi nltk
try:
    import nltk
except ImportError:
    install('nltk')
    
import streamlit as st,numpy as np,time

# Import library yang diperlukan setelah dependensi terinstal
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Fungsi untuk membersihkan teks dari karakter khusus, mengubah menjadi huruf kecil, dan melakukan lemmatisasi
def preprocess_text(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('indonesian'))  # Ganti 'english' dengan 'indonesian' jika ingin bahasa Indonesia
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text


# Fungsi untuk mendeteksi sentimen pada teks
def detect_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 10:
        sentiment = 'Positive'
    elif compound_score <= -0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, sentiment_scores


# Deploy model menggunakan Streamlit
def main():
    st.title("Sentiment Analysis pada Review Novel")
    review = st.text_input("Masukkan review novel")

    if st.button("Analyze"):
        cleaned_review = preprocess_text(review)
        sentiment, sentiment_scores = detect_sentiment(cleaned_review)

        st.write("Sentimen          : ", sentiment)
        st.write("Nilai Sentimen    : ", sentiment_scores['compound'])

if __name__ == '__main__':
    main()
