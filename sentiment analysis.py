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

