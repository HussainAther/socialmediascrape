from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the labels mapping
labels = ['Negative', 'Neutral', 'Positive']

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = text.lower().replace('[^\w\s]', '')
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join the tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['text']
        
        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)
        
        # Vectorize the preprocessed input
        vectorized_input = vectorizer.transform([preprocessed_input])
        
        # Predict the sentiment
        prediction = model.predict(vectorized_input)[0]
        sentiment = labels[prediction]
        
        return render_template('index.html', sentiment=sentiment)
    
    return render_template('index.html')


