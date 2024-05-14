import os
import pickle
import logging
import urllib.parse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data from a file and decode URL-encoded lines
def load_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    data = list(set(lines))  # Remove duplicates
    data = [urllib.parse.unquote(line.strip()) for line in data]
    return data

logging.info("Loading data...")
# Load data
bad_queries = load_file('badqueries.txt')
good_queries = load_file('goodqueries.txt')

# Prepare dataset
queries = bad_queries + good_queries

# Define labels: 1 for bad queries, 0 for good queries
labels = [1] * len(bad_queries) + [0] * len(good_queries)

# Assign labels to y
y = labels

# Prepare vectorizers
logging.info("Preparing CountVectorizer and TfidfVectorizer...")

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

# Train a SVM model using CountVectorizer features
logging.info("Training SVM model with CountVectorizer features...")
model_count = SVC(kernel='linear', probability=True, random_state=42)
model_count.fit(X_train_count, y_train)

logging.info("Evaluating model...")
# Evaluate the model with CountVectorizer features
predictions_count = model_count.predict(X_test_count)
print("CountVectorizer Model - Classification Report:")
print(classification_report(y_test, predictions_count))
print("CountVectorizer Model - Confusion Matrix:\n", confusion_matrix(y_test, predictions_count))

# Calculate F1 score, accuracy, and ROC AUC for CountVectorizer model
f1_count = f1_score(y_test, predictions_count)
accuracy_count = accuracy_score(y_test, predictions_count)
roc_auc_count = roc_auc_score(y_test, predictions_count)
print(f"CountVectorizer Model - F1 Score: {f1_count:.5f}, Accuracy: {accuracy_count:.5f}, ROC AUC: {roc_auc_count:.5f}")

# Train a SVM model using TfidfVectorizer features
logging.info("Training SVM model with TfidfVectorizer features...")
model_tfidf = SVC(kernel='linear', probability=True, random_state=42)
model_tfidf.fit(X_train_tfidf, y_train)

logging.info("Evaluating model...")
# Evaluate the model with TfidfVectorizer features
predictions_tfidf = model_tfidf.predict(X_test_tfidf)
print("\nTfidfVectorizer Model - Classification Report:")
print(classification_report(y_test, predictions_tfidf))
print("TfidfVectorizer Model - Confusion Matrix:\n", confusion_matrix(y_test, predictions_tfidf))

# Calculate F1 score, accuracy, and ROC AUC for TfidfVectorizer model
f1_tfidf = f1_score(y_test, predictions_tfidf)
accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
roc_auc_tfidf = roc_auc_score(y_test, predictions_tfidf)
print(f"TfidfVectorizer Model - F1 Score: {f1_tfidf:.5f}, Accuracy: {accuracy_tfidf:.5f}, ROC AUC: {roc_auc_tfidf:.5f}")

# Save the trained models and vectorizers for later use
logging.info("Saving models and vectorizers...")
pickle.dump(model_count, open('model_count_svm.pkl', 'wb'))
pickle.dump(count_vectorizer, open('vect_count_svm.pkl', 'wb'))
pickle.dump(model_tfidf, open('model_tfidf_svm.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vect_tfidf_svm.pkl', 'wb'))

logging.info("Process completed successfully.")
