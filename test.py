import pandas as pd
import numpy as np
import urllib.parse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import logging
import os
import pickle

# Load the data from files
def load_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    # Unquote URL-encoded data
    return [urllib.parse.unquote(line.strip()) for line in lines]

# Load bad and good queries from text files
bad_queries = load_file('badqueries.txt')
good_queries = load_file('goodqueries.txt')

# Remove duplicate queries
bad_queries = list(set(bad_queries))
good_queries = list(set(good_queries))

# Create labels for bad and good queries (1 for bad, 0 for good)
labels = [1] * len(bad_queries) + [0] * len(good_queries)

# Combine the data and labels into a DataFrame
data = pd.DataFrame({
    'query': bad_queries + good_queries,
    'label': labels
})

# Prepare dataset
queries = data['query']
labels = data['label']

# Prepare vectorizers
logging.info("Preparing CountVectorizer and TfidfVectorizer...")
# Bag of Words (CountVectorizer)
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_count = count_vectorizer.fit_transform(queries)

# TF-IDF (TfidfVectorizer)
tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(queries)

# Split the dataset into training and testing sets for both vectorizers
X_train_count, X_test_count, y_train, y_test = train_test_split(X_count, labels, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Train models

# Train a Decision Tree model using CountVectorizer features
logging.info("Training Decision Tree model with CountVectorizer features...")
model_count = DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
model_count.fit(X_train_count, y_train)

# Evaluate the model with CountVectorizer features
predictions_count = model_count.predict(X_test_count)
accuracy_count = accuracy_score(y_test, predictions_count)
f1_score_count = f1_score(y_test, predictions_count)

print("CountVectorizer Model - Classification Report:")
print(classification_report(y_test, predictions_count))
print("CountVectorizer Model - Confusion Matrix:\n", confusion_matrix(y_test, predictions_count))
print(f"Accuracy: {accuracy_count:.4f}")
print(f"F1 Score: {f1_score_count:.4f}")

# Train a Decision Tree model using TfidfVectorizer features
logging.info("Training Decision Tree model with TfidfVectorizer features...")
model_tfidf = DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
model_tfidf.fit(X_train_tfidf, y_train)

# Evaluate the model with TfidfVectorizer features
predictions_tfidf = model_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
f1_score_tfidf = f1_score(y_test, predictions_tfidf)

print("\nTfidfVectorizer Model - Classification Report:")
print(classification_report(y_test, predictions_tfidf))
print("TfidfVectorizer Model - Confusion Matrix:\n", confusion_matrix(y_test, predictions_tfidf))
print(f"Accuracy: {accuracy_tfidf:.4f}")
print(f"F1 Score: {f1_score_tfidf:.4f}")

# Save the trained models and vectorizers for later use
logging.info("Saving models and vectorizers...")
pickle.dump(model_count, open('model_count_dt.pkl', 'wb'))
pickle.dump(count_vectorizer, open('vect_count_dt.pkl', 'wb'))
pickle.dump(model_tfidf, open('model_tfidf_dt.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vect_tfidf_dt.pkl', 'wb'))

logging.info("Process completed successfully.")
