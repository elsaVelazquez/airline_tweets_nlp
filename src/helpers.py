import sklearn.feature_extraction.text
import nltk
from string import punctuation
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# stop words
sw = [
    'my', 'is', 'in', 'it', 'no', 'of', 'not',
    'your', 'me', 'hour', 'have', 'wa', 'that',
    'to', 'the', 'i', 'you', 'u', 'on', 'a', 'do',
    'for', 'at', 'so', 'and', 'be', 'now', 'with',
    'just', 'get', 'our', 'we', 'an', 'are', 'this',
    'but', 'will', 'fleek', 'im', 'if', 'it', 'u', 'or'
]

# additional lemmatization terms
additional_lemmatize_dict = {
    "cancelled": "cancel",
    "cancellation": "cancel",
    "cancellations": "cancel",
    "delays": "delay",
    "delayed": "delay",
    "baggage": "bag",
    "bags": "bag",
    "luggage": "bag",
    "dms": "dm"
}


def print_model_metrics(y_test, y_preds):
    '''
    Print classification matrix and confusion matrix for a given prediction
    '''
    class_rept_dict = classification_report(y_test, y_preds, output_dict=True)
    class_rept_df = pd.DataFrame(class_rept_dict).transpose()
    print(class_rept_df.to_markdown())
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_preds, labels=[
                         'negative', 'neutral', 'positive']),
        index=['true:negative', 'true:neutral', 'true:positive'],
        columns=['pred:negative', 'pred:neutral', 'pred:positive']
    )
    print("\n")
    print(cmtx.to_markdown())
    return


def create_stop_words(additional_stopwords=None):
    '''
    Combine SKLearn stop words with addition list (optional)
    '''

    sk_stop_words = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
    if additional_stopwords:
        return sk_stop_words.union(additional_stopwords)
    else:
        return sk_stop_words


def remove_punctuation(string, punc=punctuation):
    '''
    Remove all punctuation from a string
    '''
    for character in punc:
        string = string.replace(character, '')
    return string


def remove_hashtags(string, keep_text=False):
    '''
    Remove hashtags from a string

    if keep_text == True: remove the '#' symbol, not the text itself
    '''
    if keep_text:
        return re.sub(r"\#([^\s\#]+)", r"\1", string)
    else:
        return re.sub(r"\#\S+", "", string)


def remove_tagged_users(string):
    '''
    remove strings beginning with '@' symbols

    ex.
    remove_tagged_users("I hate @southwestairlines")
    >>> "I hate "
    '''
    return re.sub(r"\@\S+", "", string)


def remove_line_breaks(string):
    return re.sub(r"\n", " ", string)


def clean_whitespace(string):
    return re.sub(r"\s+", " ", string.strip())


def clean_df_column(df, col):
    '''
    Apply series of helper scripts to clean a text based column in a DataFrame
    '''
    df[col] = df[col].apply(remove_hashtags, keep_text=True)
    df[col] = df[col].apply(remove_tagged_users)
    df[col] = df[col].apply(remove_line_breaks)
    df[col] = df[col].apply(clean_whitespace)
    df[col] = df[col].str.strip()
    return df


def define_axis_style(ax, title, x_label, y_label, legend=False):
    '''
    Function to define labels/title font sizes for consistency across plots
    '''
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(labelsize=14)
    if legend:
        ax.legend(fontsize=16)
    return


def plot_feature_importances(
                    ax, feat_importances,
                    feat_std_deviations,
                    feat_names, n_features,
                    outfilename
                ):
    '''
    Plot feature importances for an NLP model

    feat_importances : Array of feature importances
    feat_std_deviations : Standard deviations of feature importances
                                intended for RandomForest) **OPTIONAL
    feat_names : Array of feature names
    n_features : Number of top features to include in plot
    outfilename : Path to save file
    '''
    feat_importances = np.array(feat_importances)
    feat_names = np.array(feat_names)
    sort_idx = feat_importances.argsort()[::-1][:n_features]
    if len(feat_std_deviations) > 0:
        feat_std_deviations = feat_std_deviations[sort_idx]
    else:
        feat_std_deviations = None
    ax.bar(feat_names[sort_idx], feat_importances[sort_idx], color='slateblue',
           edgecolor='black', linewidth=1, yerr=feat_std_deviations)
    ax.set_xticklabels(feat_names[sort_idx], rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(outfilename)
    plt.close('all')
    return
