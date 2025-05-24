import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify
import pickle



# Download NLTK stopwords (run once)
nltk.download('stopwords')

#Load the Sentiment140 dataset

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="latin-1", header=None)
#Assign column names
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
#keep only sentiment label and tweet text
df = df[['target', 'text']] 
#Convert 0 to negative(0) and  4 to positive (1)
df['target'] = df['target'].map({0:0, 4:1})
#use a subset of 10 000 samples to ensure efficiency on a standard laptop

df = df.sample(n=10000, random_state=42)
#Preprocesing the text

def preprocess_text(text):
    #Remove URLS to elimimate irrelevant links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    #remove mentions as they don't contribute to sentiment
    text = re.sub(r'@\w+', '', text)
    #Remove special characters and numbers - keep only letters and space
    text = re.sub(r'[^A-Za-z\s]', '', text)
    #Convert to lowercase for consistenly
    text = text.lower()
    words = text.split()
    #remove words that add little or not meaning to sentiment
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    #reduce words to their root verb with stemming
    stemmer = PorterStemmer()
    words= [stemmer.stem(word) for word in words]
    return ' '.join(words)

#Apply preprocessing

df['cleaned_text'] = df['text'].apply(preprocess_text)