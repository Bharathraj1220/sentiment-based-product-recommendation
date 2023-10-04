import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle

class SentimentRecommendationModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "Sentiment-Based-Classification-XGBoost-Model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "final-user-rating.pkl"
    CLEANED_DATA = "cleaned-final-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommendationModel.ROOT_PATH + SentimentRecommendationModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommendationModel.ROOT_PATH + SentimentRecommendationModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommendationModel.ROOT_PATH + SentimentRecommendationModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("dataset/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommendationModel.ROOT_PATH + SentimentRecommendationModel.CLEANED_DATA, 'rb'))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""

    def getSentimentRecommendations(self, user):
        if (user in self.user_final_rating.index):
            # get the product recommedation using the trained ML model
            recommendations = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(
                recommendations)]
            X = self.vectorizer.transform(
                filtered_data["lemmatized_review_text"].values.astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
                temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
            sorted_products = temp_grouped.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])

        else:
            print(f"User {user} doesn't exist")
            return None

    """function to classify the sentiment - positive or negative(1 or 0) - using the trained ML model"""

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    """function to preprocess the text before it's sent to ML model"""

    def preprocess_text(self, document):

        # tokenize into words
        words = word_tokenize(document)

        # cleaning the review text 
        words = list(map(str.lower,words))
        document = ' '.join(s for s in words if not any(c.isdigit() for c in s))
        res = re.sub(r'[^\w\s]', '', document)

        # remove stop-words and convert it to lemma
        text = self.lemmatize(res)
        return text

    """function to get the pos tag to derive the lemma form"""

    def get_words_tag(self,tag):
        if tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    """function to remove the stop words from the text"""

    def remove_stop_words(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    """function to derive the base lemma form of the text using the pos tag"""

    def lemmatize(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(self.remove_stop_words(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], self.get_words_tag(tag[1])) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)
