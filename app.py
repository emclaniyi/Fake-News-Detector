import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

# load vectorizers
count_vectorizer = pickle.load(open('models/count_vectorizer.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# load models
count_vectorizer_model = pickle.load(open('models/countvector_final_model.pkl', 'rb'))
tfidf_vectorizer_model = pickle.load(open('models/tfidf_final_model.pkl', 'rb'))


def clean_text(text):
    pattern = r"[?|$|.!'{}:<>\-(#/\")&,+=]"
    text = re.sub(pattern, '', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


prediction_labels = {
    'FAKE': 0,
    'REAL': 1
}


def main():
    st.title('Fake News Detector Web Application')
    st.subheader('NLP Supervised Learning Classifier')
    st.text_input('Title')
    news_text = st.text_area('News Text', 'Enter text here')

    vectorizer_options = ['Count Vectorizer', 'Tfidf Vectorizer']
    vectorizer_choice = st.sidebar.selectbox('Choose Text Vectorizer', vectorizer_options)

    if vectorizer_choice == 'Count Vectorizer':

        if st.button('Detect'):
            cleaned_text = clean_text(news_text)
            st.text('Cleaned Text:: \n{}'.format(news_text))

            vect_text = count_vectorizer.transform([cleaned_text]).toarray()
            prediction = count_vectorizer_model.predict(vect_text)
            st.write(prediction)
            final_result = get_keys(prediction, prediction_labels)
            st.success('News Detected as:: {}'.format(final_result))
    else:
        if st.button('Detect'):
            cleaned_text = clean_text(news_text)
            st.text('Cleaned Text:: \n{}'.format(news_text))

            vect_text = tfidf_vectorizer.transform([cleaned_text]).toarray()
            prediction = tfidf_vectorizer_model.predict(vect_text)
            st.write(prediction)
            final_result = get_keys(prediction, prediction_labels)
            st.success('News Detected as:: {}'.format(final_result))


if __name__ == '__main__':
    main()
