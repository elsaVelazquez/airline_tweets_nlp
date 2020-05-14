from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.feature_extraction.text
import nltk
from string import punctuation
import numpy as np
from nltk.stem import WordNetLemmatizer 
from helpers import sw, remove_punctuation, additional_lemmatizate_dict
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def wordnet_lemmetize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    words = remove_punctuation(text).split()
    tokens = []
    for word in words:
        if word not in sw:
            if word in additional_lemmatizate_dict:
                clean_word = additional_lemmatizate_dict[word]
            else:
                clean_word = lemmatizer.lemmatize(word)
            tokens.append(clean_word)
    return tokens


def scorer(y_true, y_predicted):
    pred_negatives = y_predicted == 'negative'                                                                             
    actual_negatives = y_true=='negative'                                                                              
    score = f1_score(actual_negatives, pred_negatives)  
    return score

scoring_func = make_scorer(scorer, greater_is_better=True)

data = pd.read_csv("data/Clean_T_Tweets.csv", index_col=0)

X = data['text']
y = data['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

param_grid = {
    'vect__min_df': [1, 12, 24, 48, 96, 1000, 5000, 10000], 
    'vect__max_df': [None, 0.1, 0.2, 0.3, 0.4, 0.5], 
    'model__alpha': [0.1, 0.2, 0.5, 1, 10, 100, 1000], 
    'vect__max_features': [None, 10, 100, 1000, 10000] 
}  


count_vect = CountVectorizer(tokenizer=wordnet_lemmetize_tokenize,
                            analyzer='word', min_df=12, max_df=0.9,
                            max_features=None)

tfidf_transformer = TfidfTransformer(use_idf=True)

nb_model = MultinomialNB(alpha=10, fit_prior=True)

nb_pipeline = Pipeline([
                        ('vect', count_vect),
                        ('tfidf', tfidf_transformer),
                        ('model', nb_model),
                        ])


nb_pipeline.fit(X_train, y_train)
y_preds = nb_pipeline.predict(X_test)

search = GridSearchCV(nb_pipeline, param_grid, scoring=scoring_func, n_jobs=-1, verbose=1, cv=3)
search.fit(X_train, y_train)
print("Best parameter (f1 score of Negative Class=%0.3f):" % search.best_score_)
print(search.best_params_)

# best params:
# {'model__alpha': 0.1, 'vect__max_df': 0.2, 'vect__max_features': 1000, 'vect__min_df': 1}

