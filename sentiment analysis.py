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

#Extract Feature with TF-IDF]

vectorizer = TfidfVectorizer(max_features=5000)
#X is the features
X = vectorizer.fit_transform(df['cleaned_text'])
#Here y is the label 0 or 1
y = df['target']

#Split data into training and test sets 80%, 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, text_size=0.2, random_state=42)

#Train the model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#EValuate the model

y_pred = model.predict(X_test)
#proportion of correct predictions
accuracy = accuracy_score(y_test, y_pred)
#accuracy of positive predictions 
precision = precision_score(y_test, y_pred)
#ability to detect all positives
recall = recall_score(y_test, y_pred)
#f1 score
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


