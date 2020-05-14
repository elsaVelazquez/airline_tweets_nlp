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

def print_model_metrics(y_test, y_preds):
    print(classification_report(y_test, y_preds))
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_preds, labels = ['negative', 'neutral', 'positive']), 
        index=['true:negative', 'true:neutral', 'true:positive'], 
        columns=['pred:negative', 'pred:neutral', 'pred:positive']
    )
    print(cmtx)
    return

if __name__ == '__main__':
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
    # {'model__alpha': 0.1, 'vect__max_df': 0.2, 'vect__max_features': 1000, 'vect__min_df': 1}


    count_vect = CountVectorizer(tokenizer=wordnet_lemmetize_tokenize,
                                analyzer='word', min_df=1, max_df=0.2,
                                max_features=1000)

    tfidf_transformer = TfidfTransformer(use_idf=True)

    nb_model = MultinomialNB(alpha=0.1, fit_prior=True)

    nb_pipeline = Pipeline([
                            ('vect', count_vect),
                            ('tfidf', tfidf_transformer),
                            ('model', nb_model),
                            ])


    nb_pipeline.fit(X_train, y_train)
    y_preds = nb_pipeline.predict(X_test)

    print_model_metrics(y_test, y_preds)

    target_names = np.unique(y_train)
    n=30 # number of top words to include
    feature_words = count_vect.get_feature_names()
    for cat in range(len(np.unique(y_train))):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_prob = nb_model.feature_log_prob_[cat]
        i_topn = np.argsort(log_prob)[::-1][:n]
        features_topn = [feature_words[i] for i in i_topn]
        print(f"Top {n} tokens: ", features_topn)