from src.d2v_custom import CustomDoc2Vec
from src.helpers import print_model_metrics

import pandas as pd
import numpy as np
from string import punctuation
import re
import random
from joblib import dump, load
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


if __name__ == "__main__":

    data = pd.read_csv("data/Clean_T_Tweets_wo_Users.csv")
    X = np.asarray(data['text'])
    y = np.asarray(data['airline_sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    np.random.seed(123)
    random.seed(123)

    d2v = CustomDoc2Vec(
                            seed=123,
                            dm=0,
                            vector_size=50,
                            epochs=5,
                            window=20,
                            alpha=0.025,
                            min_alpha=0.001
                        )

    pipe = Pipeline([
        ('doc2vec', d2v),
        ('model', RandomForestClassifier(
                            random_state=42,
                            n_jobs=-1,
                            n_estimators=400,
                            max_features='auto',
                            oob_score=True,
                            class_weight='balanced_subsample',
                            min_samples_split=10,
                            min_samples_leaf=1,
                        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print(f"Testing F1 score: {f1_score(y_test, y_pred, average='macro')}")

    print_model_metrics(y_test, y_pred)

    pipe.fit(X, y)

    if not os.path.exists("models/"):
        os.mkdir('models/')
    dump(pipe, 'models/d2v_randomforest.joblib')
