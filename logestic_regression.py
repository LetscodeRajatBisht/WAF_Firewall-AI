import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import urllib.parse

# Function to load data from a file and decode URL-encoded lines
def load_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    data = list(set(lines))  # Remove duplicates
    data = [urllib.parse.unquote(line.strip()) for line in data]
    return data

# Load data
bad_queries = load_file('badqueries.txt')
good_queries = load_file('goodqueries.txt')

# Prepare dataset
queries = bad_queries + good_queries

# Define labels: 1 for bad queries, 0 for good queries
y = [1] * len(bad_queries) + [0] * len(good_queries)

# Prepare vectorizers
# Bag of Words (CountVectorizer)
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_count = count_vectorizer.fit_transform(queries)

# TF-IDF (TfidfVectorizer)
tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(queries)

# Split the dataset into training and testing sets for both vectorizers
X_train_count, X_test_count, y_train, y_test = train_test_split(X_count, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train and evaluate models

# Train a Logistic Regression model using CountVectorizer features
print("Training Logistic Regression model with CountVectorizer features...")
model_count = LogisticRegression(max_iter=1000)
model_count.fit(X_train_count, y_train)

print("Evaluating model...")
# Evaluate the model with CountVectorizer features
predictions_count = model_count.predict(X_test_count)
print("CountVectorizer Model - Classification Report:")
print(classification_report(y_test, predictions_count))
print("CountVectorizer Model - Confusion Matrix:\n", confusion_matrix(y_test, predictions_count))

# Train a Logistic Regression model using TfidfVectorizer features
print("\nTraining Logistic Regression model with TfidfVectorizer features...")
model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)

print("Evaluating model...")
# Evaluate the model with TfidfVectorizer features
predictions_tfidf = model_tfidf.predict(X_test_tfidf)
print("\nTfidfVectorizer Model - Classification Report:")
print(classification_report(y_test, predictions_tfidf))
print("TfidfVectorizer Model - Confusion Matrix:\n", confusion_matrix(y_test, predictions_tfidf))

# Save the trained models and vectorizers for later use
print("Saving models and vectorizers...")
pickle.dump(model_count, open('model_count_logistic.pkl', 'wb'))
pickle.dump(count_vectorizer, open('vect_count_logistic.pkl', 'wb'))
pickle.dump(model_tfidf, open('model_tfidf_logistic.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vect_tfidf_logistic.pkl', 'wb'))

print("Process completed successfully.")
