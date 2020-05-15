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

    # add a few words to SKLearn stop words list
    sw = create_stop_words(more_sw)

    # load clean data
    data = pd.read_csv("data/Clean_T_Tweets_wo_Users.csv")
    X_raw = data['text']

    # Fit/transorom CountVectorizer and TF-IDF matrix
    count_vect = CountVectorizer(
                            tokenizer=None,
                            stop_words=sw,
                            analyzer='word', min_df=10,
                            max_features=None
                        )

    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_vec = count_vect.fit_transform(X_raw)
    X_tfidf = tfidf_transformer.fit_transform(X_vec)

    # Create PCA object with max features
    pca = PCA()
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # Create SCREE plot to determine appropriate number of principal components
    cum_variance = np.cumsum(pca.explained_variance_)
    total_variance = np.sum(pca.explained_variance_)
    prop_var_expl = cum_variance/total_variance

    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        prop_var_expl,
        color='blue',
        linewidth=2,
        label='Explained variance'
    )

    ax.axhline(
        0.8,
        label=r'80% Variance Explained',
        linestyle='--', color="grey",
        linewidth=1
    )

    define_axis_style(
                    ax=ax,
                    title="Proportion of Explained Variance",
                    x_label='Number of Principal Components',
                    y_label='Cumulative Proportion of Explained Variance',
                    legend=True
                )

    plt.savefig("images/scree_plot.png")
