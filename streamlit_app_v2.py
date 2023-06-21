import subprocess
import sys

# Fungsi untuk menginstal dependensi menggunakan pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List library yang ingin diinstal
packages = ['streamlit', 'pandas', 'scikit-learn', 'nltk']

# Menginstal library
for package in packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Library {package} tidak ditemukan. Menginstal...")
        install_package(package)

# Setelah instalasi selesai, import library yang dibutuhkan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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
import streamlit as st

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


def sentiment_analysis(review):
    preprocessed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([preprocessed_review])
    #prediction = classifier.predict(vectorized_review)
    score = classifier.decision_function(vectorized_review)[0]
    if score > 0.2:
        sentiment = 'Positive'
    elif score < -0.2:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, score

# Baca dataset review novel
dataset = pd.read_csv('./reviewnovel.csv')  # Ganti 'dataset.csv' dengan nama file dataset Anda
reviews = dataset['Review'].tolist()

# Preprocessing dan vektorisasi teks
preprocessed_reviews = [preprocess_text(review) for review in reviews]
vectorizer = TfidfVectorizer()
vectorized_reviews = vectorizer.fit_transform(preprocessed_reviews)

# Pelatihan model SVM
labels = dataset['Rating'].apply(lambda x: 'positive' if x > 3 else 'negative')
classifier = SVC(kernel='linear')
classifier.fit(vectorized_reviews, labels)

# Deploy model menggunakan Streamlit
def main():
    st.title("Sentiment Analysis pada Review Novel")
    review = st.text_input("Masukkan review novel")

    if st.button('Analisis Sentimen'):
        if review:
            sentiment, score = sentiment_analysis(review)
            st.write('Sentimen  :', sentiment)
            st.write('Score     : ', score)

if __name__ == '__main__':
    main()
