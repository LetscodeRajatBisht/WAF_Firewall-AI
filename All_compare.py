import pandas as pd
import numpy as np
import urllib.parse
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the data from files
def load_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    # Unquote URL-encoded data
    return [urllib.parse.unquote(line.strip()) for line in lines]

# Load bad and good queries from text files
bad_queries = load_file('badqueries.txt')
good_queries = load_file('goodqueries.txt')

# Combine bad and good queries into a single list
queries = bad_queries + good_queries

# Count occurrences of each query
query_counts = Counter(queries)

# Find duplicate queries (queries appearing more than once)
duplicate_queries = [query for query, count in query_counts.items() if count > 1]

# Display duplicate queries
if duplicate_queries:
    print("\nDuplicate queries found in the dataset:")
    for query in duplicate_queries:
        print(query)
else:
    print("\nNo duplicate queries found in the dataset.")

# Count the total number of good and bad queries
total_bad_queries = len(bad_queries)
total_good_queries = len(good_queries)
# Create labels: 1 for bad queries, 0 for good queries
labels = [1] * total_bad_queries + [0] * total_good_queries

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(queries, labels, test_size=0.2, random_state=42)

# Define pipelines for each classifier
logistic_regression = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000))  # Increase max_iter
])

decision_tree = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('classifier', DecisionTreeClassifier())
])

linear_svc = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)),
    ('classifier', LinearSVC(dual=False))  # Set dual=False
])

random_forest = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('classifier', RandomForestClassifier())
])

# List of pipelines for easy iteration
pipelines = [logistic_regression, decision_tree, linear_svc, random_forest]

# Dictionary of pipeline names for easy reference
pipeline_names = ['Logistic Regression', 'Decision Tree', 'Linear SVC', 'Random Forest']

# Initialize lists to store evaluation metrics
accuracies = []
f1_scores = []
auc_scores = []
confusion_matrices = []

# Train and evaluate each model
for pipe, name in zip(pipelines, pipeline_names):
    print(f"Training and evaluating {name}...")
    # Train the model
    pipe.fit(X_train, y_train)
    # Predict on the test set
    y_pred = pipe.predict(X_test)
    # Calculate F1 score and accuracy
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"{name} - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")

