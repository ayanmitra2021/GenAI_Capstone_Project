#Define all your imports over here.
from flask import Flask, render_template, request
import joblib
import pandas as pd
import spacy
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from flask import Flask, request, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the saved models, vectorizer, and matrices
sentiment_model = joblib.load('sentiment_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
item_user_matrix = joblib.load('item_user_matrix.joblib') 
user_item_matrix = joblib.load('user_item_matrix.joblib')

# --- Preprocessing Function 
nltk.download('stopwords', quiet=True) 
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_for_app(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    else:
        return ""


# Load the original dataframe (for product names, reviews, etc.)
df = pd.read_csv('./sample30.csv')

# Make sure preprocessing is consistent
df['processed_reviews_text'] = df['reviews_text'].apply(lambda text: preprocess_text_for_app(text) if isinstance(text, str) else "") 

#define the item based recommendations function
def item_based_recommendations(item_user_matrix, username, num_recommendations=20):
    if username not in item_user_matrix.columns:
        return "User not found."

    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

    user_ratings = item_user_matrix[username]
    rated_items = user_ratings[user_ratings > 0].index.tolist()

    recommendations = {}
    for item_id in rated_items:
        similar_items = item_similarity_df[item_id].sort_values(ascending=False).index[1:]
        for similar_item_id in similar_items:
            if similar_item_id not in rated_items:
                predicted_rating = user_ratings[item_id] * item_similarity_df.loc[item_id, similar_item_id]
                recommendations[similar_item_id] = recommendations.get(similar_item_id, 0) + predicted_rating

    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
    recommended_item_ids = [item[0] for item in sorted_recommendations[:num_recommendations]]
    recommended_products = df[df['id'].isin(recommended_item_ids)]['name'].unique().tolist()
    return recommended_products[:num_recommendations]

def user_based_recommendations(user_item_matrix, username, num_recommendations=20):
    if username not in user_item_matrix.index:
        return "User not found."

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    similar_users = user_similarity_df[username].sort_values(ascending=False).index[1:]
    user_ratings = user_item_matrix.loc[username]
    rated_items = user_ratings[user_ratings > 0].index.tolist()

    recommendations = {}
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        items_to_recommend = similar_user_ratings[~similar_user_ratings.index.isin(rated_items) & (similar_user_ratings > 0)]
        for item_id, rating in items_to_recommend.items():
            recommendations[item_id] = recommendations.get(item_id, 0) + rating * user_similarity_df.loc[username, similar_user]

    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
    recommended_item_ids = [item[0] for item in sorted_recommendations[:num_recommendations]]
    recommended_products = df[df['id'].isin(recommended_item_ids)]['name'].unique().tolist()
    return recommended_products[:num_recommendations]

#define the filtered recommendation based on sentiments
def filter_recommendations_by_sentiment(recommended_products, sentiment_model, tfidf_vectorizer, review_df, top_n=5):
    product_sentiment_scores = {}
    for product_name in recommended_products:
        product_reviews = review_df[review_df['name'] == product_name]['processed_reviews_text'].tolist()
        if not product_reviews:
            product_sentiment_scores[product_name] = 0
            continue

        tfidf_reviews = tfidf_vectorizer.transform(product_reviews)
        sentiments = sentiment_model.predict(tfidf_reviews)

        positive_sentiment_count = sum(1 for sentiment in sentiments if sentiment == 1)
        sentiment_len = len(sentiments)

        #if the sentiment length > 0 then store the sentiment score
        if sentiment_len > 0:
            sentiment_score = positive_sentiment_count / sentiment_len
        else:
            sentiment_score = 0

        product_sentiment_scores[product_name] = sentiment_score

    sorted_products_by_sentiment = sorted(product_sentiment_scores.items(), key=lambda item: item[1], reverse=True)
    top_sentiment_products = [product[0] for product in sorted_products_by_sentiment[:top_n]]
    return top_sentiment_products

#define the entry point of the API
@app.route('/', methods=['GET', 'POST'])
def index():
    user_based_recommendations_list = []
    item_based_recommendations_list = []
    recommendation_type = None # To track which tab was active

    if request.method == 'POST':
        username = request.form['username']
        recommendation_type = request.form.get('recommendation_type') # Get recommendation type from form

        if recommendation_type == 'user_based':
            if username in user_item_matrix.index:
                initial_recommendations_user = user_based_recommendations(user_item_matrix, username, num_recommendations=20)
                if not isinstance(initial_recommendations_user, str) or initial_recommendations_user != "User not found.":
                    user_based_recommendations_list = filter_recommendations_by_sentiment(
                        initial_recommendations_user, sentiment_model, tfidf_vectorizer, df, top_n=5
                    )
                else:
                    user_based_recommendations_list = [initial_recommendations_user] # "User not found." message
            else:
                user_based_recommendations_list = ["Username not found in user-based data."]

        elif recommendation_type == 'item_based':
            if username in item_user_matrix.columns:
                initial_recommendations_item = item_based_recommendations(item_user_matrix, username, num_recommendations=20)
                if not isinstance(initial_recommendations_item, str) or initial_recommendations_item != "User not found.":
                    item_based_recommendations_list = filter_recommendations_by_sentiment(
                        initial_recommendations_item, sentiment_model, tfidf_vectorizer, df, top_n=5
                    )
                else:
                    item_based_recommendations_list = [initial_recommendations_item] # "User not found." message
            else:
                item_based_recommendations_list = ["Username not found in item-based data."]

    return render_template(
        'index.html',
        user_based_recommendations=user_based_recommendations_list,
        item_based_recommendations=item_based_recommendations_list,
        recommendation_type=recommendation_type # Pass recommendation type to template
    )

#run the application
if __name__ == '__main__':
    app.run(debug=True) # Turn debug=False for production