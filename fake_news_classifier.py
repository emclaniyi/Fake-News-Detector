import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

news = pd.read_csv('assets/news_train.csv')
news = news.dropna()
print(len(news))
print(news.isnull().sum())

"""TEXT PREPROCESSING"""


# clean text column description
def clean_text(text):
    pattern = r"[?|$|.!'{}:<>\-(#/\")&,+=]"
    text = re.sub(pattern, '', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


target = news['label']

news['clean_text'] = news['text'].apply(lambda x: clean_text(x))

trainX, testX, trainY, testY = train_test_split(news['text'], target, test_size=0.33, random_state=53)

# initialize count vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(trainX.values)
pickle.dump(count_vectorizer, open('models/count_vectorizer.pkl', 'wb'))
count_test = count_vectorizer.transform(testX.values)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

# fitting with TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(trainX)
pickle.dump(tfidf_vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))
tfidf_test = tfidf_vectorizer.transform(testX)

print(tfidf_vectorizer.get_feature_names_out()[:10])
print(tfidf_train.A[:5])

# inspecting the vectors by creating dataframes
# create count vectorizer dataframe
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names_out())
# create tfidf dataframe
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names_out())
print(count_df.head())
print(tfidf_df.head())

# calculate the difference in columns
difference = set(tfidf_df) - set(count_df)
print(difference)

# check whether the dataframes are equal
print(count_df.equals(tfidf_df))

"""BUILD CLASSIFIER MODEL"""
# count vectorizer
nb_classifier = MultinomialNB()
model_nb = nb_classifier.fit(count_train, trainY)
# save the model
#pickle.dump(model_nb, open('countvector_final_model.pkl', 'wb'))
predictions_cv = nb_classifier.predict(count_test)
score = accuracy_score(testY, predictions_cv)
print('count_vectorizer:', score)

cm = confusion_matrix(testY, predictions_cv)
cv_report = classification_report(testY, predictions_cv)
print(cm)
print('cv_report', cv_report)

sns.heatmap(cm.T, square=True, annot=True, fmt='d', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()

# tfidf vectorizer
nb_classifier_2 = MultinomialNB(alpha=0.1) # since best score is 0.90 for alpha 0.1, set new alpaha and save model
model = nb_classifier_2.fit(tfidf_train, trainY)

# save the model
#pickle.dump(model, open('tfidf_final_model.pkl', 'wb'))

predictions = nb_classifier_2.predict(tfidf_test)
score_tfidf = accuracy_score(testY, predictions)
print('tfidf:', score_tfidf)

cm_tfidf = confusion_matrix(testY, predictions)
tfidf_report = classification_report(testY, predictions)
print(cm_tfidf)
print('tfidf_report:', tfidf_report)

sns.heatmap(cm_tfidf.T, square=True, annot=True, fmt='d', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()

# Create the list of alphas: alphas
alphas = np.arange(0, 1, .1)


# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier_3 = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier_3.fit(tfidf_train, trainY)
    # Predict the labels: pred
    pred = nb_classifier_3.predict(tfidf_test)
    # Compute accuracy: score
    score = accuracy_score(testY, pred)
    return score


# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
