from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.feature_extraction.text
import nltk
from string import punctuation
import numpy as np
from nltk.stem import WordNetLemmatizer
from helpers import (
            sw, remove_punctuation,
            additional_lemmatize_dict,
            plot_feature_importances,
            define_axis_style,
            print_model_metrics
        )
import itertools
import matplotlib.pyplot as plt
from joblib import dump, load


def wordnet_lemmetize_tokenize(text):
    '''
    Custom tokenizer object that applies WordNetLemmatizer
    Intended to be passed into CountVectorizer as a tokenizer object
    '''
    lemmatizer = WordNetLemmatizer()
    words = remove_punctuation(text).split()
    tokens = []
    for word in words:
        if word not in sw:
            if word in additional_lemmatize_dict:
                clean_word = additional_lemmatize_dict[word]
            else:
                clean_word = lemmatizer.lemmatize(word)
            tokens.append(clean_word)
    return tokens


if __name__ == '__main__':

    # load data
    data = pd.read_csv("data/Clean_T_Tweets_wo_Users.csv", index_col=0)

    X = data['text']
    y = data['airline_sentiment']

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Initialize count vectorizer
    count_vect = CountVectorizer(
                        tokenizer=wordnet_lemmetize_tokenize,
                        analyzer='word',
                        min_df=1,
                        max_df=0.2,
                        stop_words=['wa', 'u'],
                        max_features=1000
                    )

    # Initialize TF-IDF transformer
    tfidf_transformer = TfidfTransformer(use_idf=True)

    # Initialize MultinomialNB
    nb_model = MultinomialNB(alpha=0.1, fit_prior=True)

    # Assemble pipeline
    nb_pipeline = Pipeline([
                            ('vect', count_vect),
                            ('tfidf', tfidf_transformer),
                            ('model', nb_model),
                            ])

    # Fit, predict, and print performance metrics
    nb_pipeline.fit(X_train, y_train)
    y_preds = nb_pipeline.predict(X_test)
    print_model_metrics(y_test, y_preds)

    # Fit on total training data and export
    nb_pipeline.fit(X, y)
    dump(nb_pipeline, 'models/naivebayes.joblib')

    # Print and plot feature importances for each class
    target_names = np.unique(y)
    n = 15  # number of top words to include
    feature_words = count_vect.get_feature_names()
    for cat in range(len(np.unique(y))):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_probs = nb_model.feature_log_prob_[cat]
        top_idx = np.argsort(log_probs)[::-1][:n]
        features_top_n = [feature_words[i] for i in top_idx]
        print(f"Top {n} tokens: ", features_top_n)
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Make it pretty/consistent
        define_axis_style(
                    ax=ax,
                    title=f'Niave Bayes Top {n} Features: {target_names[cat]}',
                    x_label=None,
                    y_label="Log Probability"
                )

        plot_feature_importances(
                            ax=ax,
                            feat_importances=nb_model.feature_log_prob_[cat],
                            feat_std_deviations=[],
                            feat_names=feature_words,
                            n_features=n,
                            outfilename=f'images/nb_top_features_{target_names[cat]}'
                        )
