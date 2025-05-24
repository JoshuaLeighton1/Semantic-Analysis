# Semantic-Analysis

Sentiment Analysis on Social Media Data with Tweet URL Support

This project performs sentiment analysis on tweets, classifying them as either positive or negative. It uses the Sentiment140 dataset for training and supports real-time analysis by accepting either tweet URLs or plain text as input. The model is deployed via a Flask API for practical use.
---------------------------------------------------------------------------------------------------------------------------------------------

#Project Overview

The project analyzes the sentiment of tweets using a machine learning pipeline:

Data: Trains on the Sentiment140 dataset (1.6M tweets, subset to 10,000 for efficiency).
Preprocessing: Cleans tweet text by removing URLs, mentions, and special characters, then applies stemming.
Model: Uses Logistic Regression with TF-IDF features for sentiment classification.
Deployment: Provides a Flask API to predict sentiment from tweet URLs or plain text.

-----------------------------------------------------------------------------------------------------------------------------------------------

To run this project, you'll need:
Python 3.x
Required libraries:
pandas
scikit-learn
nltk
flask
requests
beautifulsoup4

These are all noted in requirements.txt file.
----------------------------------------------------------------------------------------------------------------------------------------------

#Dataset
Source: Download the Sentiment140 dataset from Kaggle.
File: Save it as training.1600000.processed.noemoticon.csv in your working directory.
Details: Contains 1.6M tweets labeled as 0 (negative) or 4 (positive), mapped to 0 and 1

#Usage

Training and Evaluation:

Place the dataset in your working directory.
Run the script to train the model and evaluate its performance:

For bash: "python sentiment_analysis_with_url_fixed.py" This will also run the Flask app. 

Use curl or Postman to send a POST request to http://127.0.0.1:5000/predict.

Example usage: 
curl -X POST -H "Content-Type: application/json" -d '{"text":"https://x.com/elonmusk/status/1925977074868642071"}' http://127.0.0.1:5000/predict

or with text

curl -X POST -H "Content-Type: application/json" -d '{"text":"I love this day!"}' http://127.0.0.1:5000/predict

Example output: {"sentiment": "Positive"}


Notes
Efficiency: Uses a subset of 10,000 tweets and limits features to 5,000 for performance on standard hardware.
Extensibility: Can be extended with larger datasets or different models (e.g., BERT).
Practicality: Supports both twitter.com and x.com URLs for real-world use


