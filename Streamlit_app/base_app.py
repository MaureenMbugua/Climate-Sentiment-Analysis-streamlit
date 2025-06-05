"""

    Simple Streamlit webserver application for serving developed classification
	models.

    This file is used to launch a minimal streamlit web
	application. 
    
"""
# Streamlit dependencies
from bleach import clean
import streamlit as st
import joblib, os

# Data and Plot dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import pickle
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pprint import pprint
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Preprocessing
stemmer= PorterStemmer()
def processed_tweet(message):
    """
    Returns a list of stemmed words from a list of input words
    Parameters:
        message (str): the string to be tokenised and stemmed
    Returns:
        message (str): stemmed string
    """
    stopwords_list= stopwords.words('english')
    message= message.lower().strip()
    message= word_tokenize(message)
    message= " ".join(message)
    message= re.sub(r"https\S+|www\S+https\S+", '', message, flags=re.MULTILINE)
    message= re.sub(r'\@w+|\#', '', message)
    message= re.sub(r'[^\w\s]','', message)
    token_words= word_tokenize(message)
    filtered_list= [word for word in token_words if word not in stopwords_list]
    filtered_message= " ".join(filtered_list)
    message= " ".join([stemmer.stem(word) for word in word_tokenize(filtered_message)])

    return message

# create a wordcloud of most frequently occuring words per Sentiment
# def sentIment_wordcloud(df, sent):
# 	sentiment= dictionary[sent]
# 	sentiment_text= " ". join([word for word in df['message'][df['sentiment']== sentiment]])
# 	wordcloud= WordCloud(max_words=500, width=1600, height=800).generate(sentiment_text)
# 	fig, ax= plt.subplots(figsize=(30,20))
# 	ax.imshow(wordcloud)
# 	plt.axis('off')
# 	return fig


# Vectorizer
vect = open("resources/predict_logreg_vect.pkl","rb")
tweet_cv = joblib.load(vect) # loading your vectorizer from the pkl file
dictionary= {-1: 'Anti-Climate Change', 0: 'Neutral', 1: 'Pro-Climate Change', 2: 'News Fact on climate Change'}

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	
		# Building out a word cloud
		st.subheader("Word Cloud")
		# options= st.selectbox('select topic',['Anti Climate Change','Neutral','Pro Climate Change','News'])
		# if options== 'Anti Climate Change':
		# 	st.pyplot(sentIment_wordcloud(raw, options)) 
		if st.checkbox('Show Word Cloud'):
			# sentiment= dictionary[sent]
			st.text("This shows the most common used words of all tweets\n in the dataframe regardless of sentiment")
			sentiment_text= " ". join([word for word in raw['message']])
			wordcloud= WordCloud(max_words=500, width=1600, height=800).generate(sentiment_text)
			fig, ax= plt.subplots(figsize=(40,40))
			ax.imshow(wordcloud)
			plt.axis('off')
			st.pyplot(fig)

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		cleaned_tweet= processed_tweet(tweet_text)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([cleaned_tweet]).toarray()
			# Load your .pkl file with the model to make predictions
			
			predictor = joblib.load(open(os.path.join("resources/predict_logreg_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
            
			st.success("Text Categorized as: {}".format(dictionary[list(prediction)[0]]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
