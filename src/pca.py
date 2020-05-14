import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os
import imageio
import glob
from helpers import create_stop_words, define_axis_style

if __name__ == '__main__':

    more_sw = [
        "im",
        "fly",
        "wa",
        "airline",
        "fleek",
        "got"
    ]

    sw = create_stop_words(more_sw)

    data = pd.read_csv("data/Tweets.csv")
    X_raw = data['text']

    count_vect = CountVectorizer(tokenizer=None, stop_words=sw,
                                analyzer='word', min_df=10,
                                max_features=None)

    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_vec = count_vect.fit_transform(X_raw)
    X_tfidf = tfidf_transformer.fit_transform(X_vec)

    # scree plot
    # don't define n features
    pca = PCA()
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # scree plot
    cum_variance = np.cumsum(pca.explained_variance_)
    total_variance = np.sum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance

    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(prop_var_expl, color='blue', linewidth=2, label='Explained variance')
    ax.axhline(0.9, label=r'90% Variance Explained', linestyle='--', color="grey", linewidth=1)
    define_axis_style(
                    ax=ax,
                    title="Proportion of Explained Variance",
                    x_label='Number of Principal Components',
                    y_label='Cumulative Proportion of Explained Variance',
                    legend=True
                )
    plt.savefig("images/scree_plot.png")

    