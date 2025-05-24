import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

#Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# define function to fetch tweet from URL using oEmbed API
def get_tweet_text_from_url(url):
    oembed_url = f"https://publish.twitter.com/oembed?url={url}"
    response = requests.get(oembed_url)
    if response.status_code == 200:
        data = response.json()
        html = data['html']
        soup = BeautifulSoup(html, 'html.parser')
        tweet_text = soup.find('p').text
        return tweet_text
    else:
        raise ValueError(f"Failed to fetch tweet: HTTP {response.status_code}")

# Function to validate tweet URL
def is_valid_tweet_url(url):
    pattern = r'^https?://twitter\.com/\w+/status/\d+$'
    return bool(re.match(pattern, url))

#Flask app for deployement
app = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


#define route for predictions

@app.route('/predict', methods=['POST'])
def predict():
    #get json data from reqest
    data = request.json
    input_text = data['text']

    #Handle URL or plain text input 
    if input_text.startswith('http'):
        if not is_valid_tweet_url(input_text):
            return jsonify({'error': 'Invalid tweet URL'}), 400
        try:
            tweet_text = get_tweet_text_from_url(input_text)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        tweet_text = input_text

    #Preprocess and predict
    cleaned_text = preprocess_text(text)
    #Convert to TF-IDF features
    features = vectorizer.transform([cleaned_text])
    #make a prediction
    prediction =  model.predict(features)[0]
    sentiment = 'Positive'  if prediction ==1 else 'Negative'
    #return as json
    return jsonify({'sentiment': sentiment})

# Run the flask app

if __name__=="__main__":
    app.run(debug=True)