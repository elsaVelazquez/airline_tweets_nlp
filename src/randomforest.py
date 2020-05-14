from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from naive_bayes import print_model_metrics

from helpers import sw

data = pd.read_csv("data/Clean_T_Tweets.csv", index_col=0)

X = data['text']
y = data['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

count_vect = CountVectorizer(stop_words=sw, analyzer='word')
X_train_vec = count_vect.fit_transform(X_train)
X_test_vec = count_vect.transform(X_test)

rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=1000)

rf_pipeline = Pipeline([
                        ('vect', count_vect),
                        ('model', rf)
                        ])

rf_pipeline.fit(X_train, y_train)

print(f"Baseline Cross Val score = {cross_val_score(rf_pipeline, X_train, y_train, cv=5).mean()}")

y_preds = rf_pipeline.predict(X_test)

print_model_metrics(y_test, y_preds)

# print("Cross Val Score: ",cross_val_score(rf_pipeline, X_train, y_train, cv=5).mean())
# print("Train Score: ", round(rf_pipeline.score(X_train, y_train), 4))
# print("Test Score: ", round(rf_pipeline.score(X_test, y_test), 4))