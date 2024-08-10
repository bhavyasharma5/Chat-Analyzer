import os
import re
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def load_data(filepath):
    """
    Load data from the given filepath.
    """
    return pd.read_csv(filepath)

def clean_text(text):
    """
    Clean the text data by removing unwanted characters and symbols.
    """
    text = text.replace('<Media omitted>', '').replace('This message was deleted', '').replace('\n', ' ').strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]|_', '', text)
    text = re.sub(r'([a-zA-Z])\1\1', r'\1', text)
    return text.lower()

def analyze_sentiment(chat):
    """
    Perform sentiment analysis on the chat data.
    """
    # Initialize the VADER sentiment analyzer with custom lexicon
    pos, neg = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()

    pos.lexicon.clear()
    neg.lexicon.clear()

    pos.lexicon.update(pd.read_table('lexicon/InSet/positive.tsv').set_index('word').to_dict()['weight'])
    neg.lexicon.update(pd.read_table('lexicon/InSet/negative.tsv').set_index('word').to_dict()['weight'])

    # Calculate sentiment score
    chat['sentiment'] = chat.apply(lambda x: (neg.polarity_scores(
        x['clean_msg'])['compound'] + pos.polarity_scores(
        x['clean_msg'])['compound'] + (
        0 if (score := emosent_score(x['emoji'])) == 0 else score)
        ) / ((2 if score != 0 else 1) if (neg.polarity_scores(
        x['clean_msg'])['compound'] + pos.polarity_scores(
        x['clean_msg'])['compound']) != 0 else 1), axis=1)

    # Label the sentiment
    chat['sentiment'] = chat['sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

    return chat

def save_sentiment_results(chat, filepath):
    """
    Save the sentiment analysis results to a file.
    """
    chat.to_csv(filepath, index=False)

def SentimentAnalysis():
    # Load data
    chat = load_data('whatsapp_chat.csv')

    # Clean text
    chat['clean_msg'] = chat['message'].apply(clean_text)

    # Perform sentiment analysis
    chat = analyze_sentiment(chat)

    # Save sentiment analysis results
    save_sentiment_results(chat, 'sentiment_results.csv')

if __name__ == "__main__":
    SentimentAnalysis()
