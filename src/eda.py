import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from helpers import clean_df_column, define_axis_style, sw


def group_plot_pos_neg_dist(
                    df, ax, title, groupby_col,
                    normalize=False, outfilepath=None
                ):
    '''
    Create a plot describing sentiments grouped by a specified column
    '''
    x_vals = np.array(df[groupby_col].unique())
    y_vals_total = np.array([sum(df[groupby_col] == val) for val in x_vals])
    sort_idx = np.argsort(y_vals_total)[::-1]

    y_vals_pos = np.array([])
    y_vals_neutral = np.array([])

    for val in x_vals:
        pos_count = sum((df[groupby_col] == val) & (df['airline_sentiment'] == 'positive'))
        y_vals_pos = np.append(y_vals_pos, pos_count)
        neut_count = sum((df[groupby_col] == val) & (df['airline_sentiment'] == 'neutral'))
        y_vals_neutral = np.append(y_vals_neutral, neut_count)

    if normalize:
        normalize_factors = y_vals_total[sort_idx]
        y_label = 'Proportion of Tweets'
        legend_bool = False
    else:
        normalize_factors = 1
        y_label = 'Number of Tweets'
        legend_bool = True

    neg_heights = (y_vals_total[sort_idx]) / normalize_factors
    neu_heights = (y_vals_total[sort_idx] - y_vals_pos[sort_idx]) / normalize_factors
    pos_heights = (y_vals_total[sort_idx] - y_vals_pos[sort_idx] - y_vals_neutral[sort_idx]) / normalize_factors

    ax.bar(
        x_vals[sort_idx], neg_heights,
        label='Positive Tweets',
        color='seagreen',
        edgecolor='black',
        linewidth=1
    )

    ax.bar(
        x_vals[sort_idx], neu_heights,
        label='Neutral Tweets',
        color='gold',
        edgecolor='black',
        linewidth=1
    )

    ax.bar(
        x_vals[sort_idx], pos_heights,
        label='Negative Tweets',
        color='firebrick',
        edgecolor='black',
        linewidth=1
    )

    define_axis_style(ax, title, x_label=None, y_label=y_label, legend=legend_bool)
    if outfilepath:
        plt.savefig(outfilepath)
        plt.close('all')
    return


def plot_total_sentiment_dist(df, ax, title, outfilepath=None):
    '''
    Create a plot describing sentiments across all data
    '''
    x_vals = np.array(["All Airlines"])
    y_vals_total = np.array([len(df)])

    y_vals_pos = np.array([])
    y_vals_neutral = np.array([])

    pos_count = sum(df['airline_sentiment'] == 'positive')
    y_vals_pos = np.append(y_vals_pos, pos_count)
    neut_count = sum(df['airline_sentiment'] == 'neutral')
    y_vals_neutral = np.append(y_vals_neutral, neut_count)

    barwidth = 1.6

    ax.bar(
        x_vals, y_vals_total,
        label='Positive Tweets',
        color='seagreen',
        edgecolor='black',
        linewidth=1,
        width=barwidth
    )

    ax.bar(
        x_vals, y_vals_total - y_vals_pos,
        label='Neutral Tweets',
        color='gold',
        edgecolor='black',
        linewidth=1,
        width=barwidth
    )

    ax.bar(
        x_vals, y_vals_total - y_vals_pos - y_vals_neutral,
        label='Negative Tweets',
        color='firebrick',
        edgecolor='black',
        linewidth=1,
        width=barwidth
    )

    ax.set_xlim(-1, 2)
    define_axis_style(ax, title, x_label=None, y_label='Number of Tweets', legend=True)
    if outfilepath:
        plt.savefig(outfilepath)
        plt.close('all')
    return


def create_word_cloud(text, title, additional_stop_words, outfilepath):
    '''
    Create and save a wordcloud with given text
    '''
    wordcloud = WordCloud(stopwords=additional_stop_words, width=1600, height=800).generate(text)
    plt.figure(figsize=(10, 6), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontdict={"fontsize": 22})
    plt.tight_layout()
    plt.savefig(outfilepath, bbox_inches='tight')
    plt.close('all')
    return


if __name__ == '__main__':
    # set style and load data
    plt.style.use("seaborn")
    raw_df = pd.read_csv("data/Tweets.csv")

    # Create wordcloud for raw data
    create_word_cloud(
                text=" ".join(raw_df['text']),
                title="Raw Tweets",
                additional_stop_words=[],
                outfilepath=f"images/raw_wordcloud.png"
            )

    # plot distribution of sentiment confidence
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(raw_df['airline_sentiment_confidence'],
            bins=np.linspace(0, 1, 11), edgecolor='black',
            linewidth=1)
    ax.set_xticks(np.linspace(0, 1, 11))

    # make it pretty/consistent
    define_axis_style(
                        ax=ax,
                        title="Distribution of Sentiment Confidence",
                        x_label="Sentiment Confidence",
                        y_label="Number of Tweets"
                    )

    plt.savefig('images/sentiment_conf_dist.png')
    plt.close('all')

    # drop rows with confidence < 0.5
    high_conf_df = raw_df[raw_df['airline_sentiment_confidence'] >= 0.5]

    # plot overall sentiment
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_total_sentiment_dist(
                        df=high_conf_df,
                        ax=ax,
                        title="Tweet Sentiment in Training Data",
                        outfilepath="images/overall_tweet_sentiment.png"
                    )

    # plot sentiment grouped by airline
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    group_plot_pos_neg_dist(
                        df=high_conf_df,
                        ax=ax,
                        title="Tweet Sentiment by Airline in Training Data",
                        groupby_col='airline',
                        normalize=False,
                        outfilepath="images/airline_tweet_sentiment.png"
                    )

    # plot sentiment grouped by airline (normalized)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    group_plot_pos_neg_dist(
                        df=high_conf_df,
                        ax=ax,
                        title="Tweet Sentiment by Airline in Training Data (Normalized)",
                        groupby_col='airline',
                        normalize=True,
                        outfilepath="images/airline_tweet_sentiment_normal.png"
                    )

    # load cleaned data
    clean_data = pd.read_csv("data/Clean_T_Tweets.csv")

    clean_data = clean_df_column(clean_data, 'text')

    # add a few stop words to make it prettier
    stop_words = sw + ["amp", "flight", "flights", "fly", "airline", "airport", "just", "was"]

    # create a word cloud for all tweets
    all_text_sent = clean_data['text']
    create_word_cloud(
                text=" ".join(all_text_sent),
                title="All Tweets",
                additional_stop_words=stop_words,
                outfilepath=f"images/total_wordcloud.png"
            )

    # create wordclouds for each sentiment
    for sentiment in np.unique(clean_data['airline_sentiment']):
        clean_text_sent = clean_data[clean_data['airline_sentiment'] == sentiment]['text']
        create_word_cloud(
                    text=" ".join(clean_text_sent),
                    title=f"{sentiment.capitalize()} Tweets",
                    additional_stop_words=stop_words,
                    outfilepath=f"images/{sentiment}_wordcloud.png"
                )
